import torch
import datasets
import json
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from src.data import BERTDataModule, LabelModelDataModule
from src.models.BERT import BERT

def get_conf_inds(labels, features, coverage, device='cuda',
                  K=20, scores=None, return_scores=False):
    N = labels.shape[0]
    if scores is None:
        features = torch.FloatTensor(features).to(device)
        labels = torch.LongTensor(labels).to(device)

        # move to CPU for memory issues on large dset
        pairwise_dists = torch.cdist(features, features, p=2).to('cpu')

        dists_sorted = torch.argsort(pairwise_dists)
        neighbors = dists_sorted[:,:K]
        dists_nn = pairwise_dists[torch.arange(N)[:,None], neighbors]
        weights = 1/(1 + dists_nn)

        neighbors = neighbors.to(device)
        dists_nn = dists_nn.to(device)
        weights = weights.to(device)

        cut_vals = (labels[:,None] != labels[None,:]).long()
        cut_neighbors = cut_vals[torch.arange(N)[:,None], neighbors]
        Jp = (weights * cut_neighbors).sum(dim=1)

        weak_counts = torch.bincount(labels)
        weak_pct = weak_counts / weak_counts.sum()

        prior_probs = weak_pct[labels]
        mu_vals = (1-prior_probs) * weights.sum(dim=1)
        sigma_vals = prior_probs * (1-prior_probs) * torch.pow(weights, 2).sum(dim=1)
        sigma_vals = torch.sqrt(sigma_vals)
        normalized = (Jp - mu_vals) / sigma_vals
        normalized = normalized.cpu()
    else:
        normalized = scores

    inds_sorted = torch.argsort(normalized)
    N_select = int(coverage * N)
    conf_inds = inds_sorted[:N_select]
    conf_inds = list(set(conf_inds.tolist()))
    if return_scores:
        return conf_inds, normalized
    else:
        return conf_inds


def get_conf_inds_per_class(labels, features, num_per_class, device='cuda',
                            K=20, scores=None, return_scores=False,
                            ref_labels=None, ref_features=None):
    N = labels.shape[0]
    uniq_labels, counts = np.unique(labels.numpy(), return_counts=True)
    if scores is None:
        features = torch.FloatTensor(features).to(device)
        labels = torch.LongTensor(labels).to(device)

        # move to CPU for memory issues on large dset
        pairwise_dists = torch.cdist(features, features, p=2).to('cpu')

        dists_sorted = torch.argsort(pairwise_dists)
        neighbors = dists_sorted[:,:K]
        dists_nn = pairwise_dists[torch.arange(N)[:,None], neighbors]
        weights = 1/(1 + dists_nn)

        neighbors = neighbors.to(device)
        dists_nn = dists_nn.to(device)
        weights = weights.to(device)

        cut_vals = (labels[:,None] != labels[None,:]).long()
        cut_neighbors = cut_vals[torch.arange(N)[:,None], neighbors]
        Jp = (weights * cut_neighbors).sum(dim=1)

        weak_counts = torch.bincount(labels)
        weak_pct = weak_counts / weak_counts.sum()

        prior_probs = weak_pct[labels]
        mu_vals = (1-prior_probs) * weights.sum(dim=1)
        sigma_vals = prior_probs * (1-prior_probs) * torch.pow(weights, 2).sum(dim=1)
        sigma_vals = torch.sqrt(sigma_vals)
        normalized = (Jp - mu_vals) / sigma_vals
        normalized = normalized.cpu()
    else:
        normalized = scores

    conf_inds = []
    all_inds = torch.arange(labels.shape[0])

    for l in uniq_labels:
        l_inds = all_inds[labels == l]
        scores_l = normalized[labels == l]
        sorted_inds_l = l_inds[torch.argsort(scores_l)]
        N_select = min(len(l_inds), num_per_class)
        conf_inds.extend(sorted_inds_l[:N_select].tolist())

    conf_inds = list(set(conf_inds))
    if return_scores:
        return conf_inds, normalized
    else:
        return conf_inds


# select a minimum percentage per class,
# then select the rest from the unstratified ranking.
def get_conf_inds_minppc(labels, features, coverage, min_ppc, device='cuda',
                         K=20, return_scores=False):
    N = labels.shape[0]
    features = torch.FloatTensor(features).to(device)
    uniq_labels, _ = np.unique(labels.numpy(), return_counts=True)
    labels = torch.LongTensor(labels).to(device)

    # move to CPU for memory issues on large dset
    pairwise_dists = torch.cdist(features, features, p=2).to('cpu')

    dists_sorted = torch.argsort(pairwise_dists)
    neighbors = dists_sorted[:,:K]
    dists_nn = pairwise_dists[torch.arange(N)[:,None], neighbors]
    weights = 1/(1 + dists_nn)

    neighbors = neighbors.to(device)
    dists_nn = dists_nn.to(device)
    weights = weights.to(device)

    cut_vals = (labels[:,None] != labels[None,:]).long()
    cut_neighbors = cut_vals[torch.arange(N)[:,None], neighbors]
    Jp = (weights * cut_neighbors).sum(dim=1)

    weak_counts = torch.bincount(labels)
    weak_pct = weak_counts / weak_counts.sum()

    prior_probs = weak_pct[labels]
    mu_vals = (1-prior_probs) * weights.sum(dim=1)
    sigma_vals = prior_probs * (1-prior_probs) * torch.pow(weights, 2).sum(dim=1)
    sigma_vals = torch.sqrt(sigma_vals)
    normalized = (Jp - mu_vals) / sigma_vals
    normalized = normalized.cpu()

    conf_inds = []
    all_inds = torch.arange(labels.shape[0])
    N_select = int(coverage*N)

    # first select the top min_ppc*Nselect for each label.
    for l in uniq_labels:
        l_inds = all_inds[labels == l]
        scores_l = normalized[labels == l]
        sorted_inds_l = l_inds[torch.argsort(scores_l)]
        min_select = min(len(l_inds), int(min_ppc*N_select))
        conf_inds.extend(sorted_inds_l[:min_select].tolist())

    # now select the rest from the pooled (non-stratified) ranking
    remaining_inds = list(set(all_inds).difference(set(conf_inds)))
    remaining_inds = torch.tensor(remaining_inds)

    scores = normalized[remaining_inds]
    inds_sorted = torch.argsort(remaining_inds)
    global_inds_sorted = remaining_inds[inds_sorted]
    num_select = N_select - len(conf_inds)
    conf_inds.extend(global_inds_sorted[:num_select])
    conf_inds = list(set(conf_inds))
    return conf_inds



add_special_tokens=True
def extract_psl_and_features(example, prompttokenizer=None, promptmodel=None, list_templates=None, template_idx=None, device='cuda'):
    promptmodel.eval()
    with torch.no_grad():
        if template_idx is not None:
            template = list_templates[template_idx]
        else:
            template = np.random.choice(list_templates)

        input_str, target_str = template.apply(example)
        answer_choices = template.get_answer_choices_list(example)
        if isinstance(input_str, list):
            input_ids = torch.cat(
                [
                    prompttokenizer(
                        input_field, return_tensors="pt", truncation=True, add_special_tokens=False
                    ).input_ids.squeeze(0)
                    for input_field in input_str[:-1]
                ]
                + [
                    prompttokenizer(
                        input_str[-1], return_tensors="pt", truncation=True, add_special_tokens=add_special_tokens
                    ).input_ids.squeeze(0)
                ]
            )
        else:
            tok_outputs = prompttokenizer(
                input_str, return_tensors="pt", truncation=True, add_special_tokens=add_special_tokens
            )
            input_ids = tok_outputs.input_ids.squeeze(0)
            tok_outputs = {k:v.to('cuda') for (k,v) in tok_outputs.items()}

        target_ids = prompttokenizer(
            target_str, return_tensors="pt", truncation=True, add_special_tokens=add_special_tokens
        ).input_ids.squeeze(0)
        answer_choices_ids = [
            prompttokenizer(
                answer_choice, return_tensors="pt", truncation=True, add_special_tokens=add_special_tokens
            ).input_ids.squeeze(0)
            for answer_choice in answer_choices
        ]

        target_ids = target_ids.unsqueeze(0)
        input_ids = input_ids.unsqueeze(0)
        answer_choices_ids = [answer_choices_ids]

        flat_answer_choice_ids = [choice for list_choices in answer_choices_ids for choice in list_choices]
        num_choice = [len(list_choices) for list_choices in answer_choices_ids]
        if max(num_choice) != min(num_choice):
            raise NotImplementedError("The collate_fn is not implmented for variable number of choices")
        flat_answer_choices_ids = torch.nn.utils.rnn.pad_sequence(
            flat_answer_choice_ids, batch_first=True, padding_value=prompttokenizer.pad_token_id
        )
        answer_choices_ids = flat_answer_choices_ids.view(len(answer_choices_ids), max(num_choice), -1).contiguous()

        input_ids, choices_ids, labels = input_ids, answer_choices_ids, target_ids
        bs, num_choices = choices_ids.size()[:2]

        flat_choices_ids = choices_ids.flatten(0, 1)
        attention_mask = (input_ids != prompttokenizer.pad_token_id).float()  # [bs, max_seq_len]

        input_ids=input_ids.to(device)
        attention_mask = attention_mask.to(device)

        encoder_hidden_states = promptmodel.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        encoder_features = encoder_hidden_states[0].mean(dim=0)

        encoder_hidden_states = encoder_hidden_states.unsqueeze(dim=1).repeat(1, num_choices, 1, 1).flatten(0, 1)

        attention_mask = attention_mask.unsqueeze(dim=1).repeat(1, num_choices, 1).flatten(0, 1)
        decoder_input_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
        decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
        lm_target = flat_choices_ids - 100 * (flat_choices_ids == prompttokenizer.pad_token_id).long()

        model_output = promptmodel(
            attention_mask=attention_mask,
            encoder_outputs=[encoder_hidden_states],
            decoder_input_ids=decoder_input_ids.to(device),
            decoder_attention_mask=decoder_attention_mask.to(device),
            output_hidden_states=True
        )

        choices_scores = (
            F.cross_entropy(model_output.logits.flatten(0, 1), lm_target.flatten(0, 1).to(device), reduction="none")
            .view(bs, num_choices, -1)
            .sum(dim=-1)
        )

        choice = choices_scores[0].argmin()
        example['pseudolabel'] = choice.cpu().numpy().item()

        decoder_features = model_output.decoder_hidden_states[-1][0,0]
        features = decoder_features # this is what we use in the icml paper

        example['features'] = features.cpu().numpy().tolist()
        return example


def get_dsdict_prompt(config, dataset_reader, prompttokenizer, promptmodel):
    train = dataset_reader.read_orig_dataset('train')

    # for super_glue, use public validation like a test set, for our purposes.
    test_key = 'validation' if config.validation_is_test else 'test'
    test = dataset_reader.read_orig_dataset(test_key)

    templates = dataset_reader.get_template(-1)
    list_templates = templates

    ds_dict = dataset_reader.get_full_orig_dataset()

    if config.train_template_idx == -1:
        template_idx = np.random.choice(len(list_templates))
    else:
        template_idx = config.train_template_idx

    train = train.map(extract_psl_and_features, batched=False,
                      fn_kwargs={'prompttokenizer': prompttokenizer,
                                 'promptmodel': promptmodel,
                                 'list_templates': list_templates,
                                 'template_idx': template_idx})

    test = test.map(extract_psl_and_features, batched=False,
                    fn_kwargs={'prompttokenizer': prompttokenizer,
                               'promptmodel': promptmodel,
                               'list_templates': list_templates,
                               'template_idx': template_idx})


    lab = np.array(test['label'])
    psl = np.array(test['pseudolabel'])
    acc = (lab == psl).astype(float).mean()
    print(f"T0 TRUE test accuracy: {acc}")

    accumulated = {"prediction": psl, "label": lab}
    metrics = dataset_reader.compute_metric(accumulated)
    for key, value in accumulated.items():
        if key.startswith("log."):
            metrics[key.replace("log.", "")] = mean(value)

    result_str = json.dumps(metrics) + "\n"
    t0_path = f"{config.dev_score_file}.t0"
    with open(t0_path, "a+") as f:
        f.write(result_str)
    print("\n" + result_str)

    train_subset, val_subset, test = make_subset(config, train, test)
    ds_dict['train'] = train_subset
    ds_dict['validation'] = val_subset
    ds_dict['test'] = test
    return ds_dict




def extract_psl_and_features_bert(example, bert):
    bert.eval()
    with torch.no_grad():
        example.pop('labels', None)
        outputs = bert(**{k: v.to('cuda').unsqueeze(0) for (k,v) in example.items()},
                        output_hidden_states=True)
        hs = outputs.hidden_states[-1]
        avg_rep = hs[:,0,:].cpu().numpy().tolist()[0] # cls token
        psl = outputs.logits.argmax(dim=1).item()
        example = {'features': avg_rep, 'pseudolabel': psl}
        return example

def get_dsdict_bert(config, dataset_reader, bertmodule):
    ds_dict = dataset_reader.get_full_orig_dataset()

    datamodule = BERTDataModule(
        bertmodule.model_name_or_path,
        bertmodule.task_name,
        ds_dict
    )
    datamodule.setup('fit') # run tokenization
    train = datamodule.dataset['train']

    # for super_glue, use public validation like a test set, for our purposes.
    test_key = 'validation' if config.validation_is_test else 'test'
    test = datamodule.dataset[test_key]

    model = bertmodule.model.to('cuda')
    train = train.map(extract_psl_and_features_bert, batched=False,
                      fn_kwargs={'bert': model})
    test = test.map(extract_psl_and_features_bert, batched=False,
                      fn_kwargs={'bert': model})

    train = train.rename_column("labels", "label")
    test = test.rename_column("labels", "label")

    lab = np.array(test['label'])
    psl = np.array(test['pseudolabel'])
    acc = (lab == psl).astype(float).mean()
    print(f"BERT TRUE test accuracy: {acc}")


    accumulated = {"prediction": psl, "label": lab}
    metrics = dataset_reader.compute_metric(accumulated)
    for key, value in accumulated.items():
        if key.startswith("log."):
            metrics[key.replace("log.", "")] = mean(value)

    result_str = json.dumps(metrics) + "\n"
    bert_path = f"{config.dev_score_file}.bert"
    with open(bert_path, "a+") as f:
        f.write(result_str)
    print("\n" + result_str)

    # need to zip train and test back up with the originals
    # since datamodule setup() adds stuff we don't want later (BERT input ids, etc).
    # we only need the text fields when we train T0.

    # add the new columns from the mapped versions to the
    # original sets.

    train_subset, val_subset, test = make_subset(config, train, test)

    # calling from_dict(ds.to_dict()) seems required to re-wrap everything correctly.
    ds_dict['train'] = datasets.Dataset.from_dict(train_subset.to_dict())
    ds_dict['validation'] = datasets.Dataset.from_dict(val_subset.to_dict())
    ds_dict['test'] = datasets.Dataset.from_dict(test.to_dict())
    return ds_dict


def extract_psl_and_features_labelmodel(example, model):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(example['feat'])[None,...]
        inputs = inputs.to(model.device)
        pred = model(inputs)
        psl = pred.argmax(dim=1).item()
        feat = example['feat']
        example = {'features': inputs.view(-1).cpu().tolist(), 'pseudolabel': psl, 'scores': pred.view(-1)}
        return example


def get_dsdict_labelmodel(config, dataset_reader, model, bert_name=None, bert_model=None):
    ds_dict = dataset_reader.get_full_orig_dataset()

    datamodule = LabelModelDataModule(
        config,
        ds_dict
    )
    datamodule.setup('fit') # convert to torch tensors

    # for these datasets, i've already renamed (the superglue) "validation" to "test"
    train = datamodule.dataset['train']
    test_key = 'validation' if config.validation_is_test else 'test'
    test = datamodule.dataset[test_key]

    model = model.to('cuda')
    train = train.map(extract_psl_and_features_labelmodel, batched=False,
                      fn_kwargs={'model': model})
    test = test.map(extract_psl_and_features_labelmodel, batched=False,
                    fn_kwargs={'model': model})

    # sets 'features' column to the bert features to use in the
    # data selection.
    if bert_name is not None:
        bert_dm = BERTDataModule(
            bert_name,
            config.dataset,
            dataset_reader.get_full_orig_dataset()
        )
        bert_dm.setup('fit')
        if bert_model is None:
            bert = BERT(
                model_name_or_path=bert_name,
                num_labels=bert_dm.num_labels,
                task_name=bert_dm.task_name,
                dataset_reader=dataset_reader,
                warmup_steps=500,
                weight_decay=config.bert_wd,
                learning_rate=config.bert_lr
            )
        else:
            bert = bert_model

        bert = bert.model.to('cuda')
        trainbert = bert_dm.dataset['train'].map(extract_psl_and_features_bert, batched=False,
                                                 fn_kwargs={'bert': bert})
        testbert = bert_dm.dataset[test_key].map(extract_psl_and_features_bert, batched=False,
                                                 fn_kwargs={'bert': bert})

        train = train.remove_columns(["features"])
        test = test.remove_columns(["features"])
        train = train.add_column("features", trainbert['features'].tolist())
        test = test.add_column("features", testbert['features'].tolist())

    print(train)
    lab = np.array(test['label'])
    psl = np.array(test['pseudolabel'])
    acc = (lab == psl).astype(float).mean()
    print(f"LABELMODEL TRUE test accuracy: {acc}")

    accumulated = {"prediction": psl, "label": lab}
    metrics = dataset_reader.compute_metric(accumulated)
    for key, value in accumulated.items():
        if key.startswith("log."):
            metrics[key.replace("log.", "")] = mean(value)

    result_str = json.dumps(metrics) + "\n"
    lm_path = f"{config.dev_score_file}.lm"
    with open(lm_path, "a+") as f:
        f.write(result_str)
    print("\n" + result_str)

    train_subset, val_subset, test = make_subset(config, train, test)

    # calling from_dict(ds.to_dict()) seems required to re-wrap everything correctly.
    ds_dict['train'] = datasets.Dataset.from_dict(train_subset.to_dict())
    ds_dict['validation'] = datasets.Dataset.from_dict(val_subset.to_dict())
    ds_dict['test'] = datasets.Dataset.from_dict(test.to_dict())
    return ds_dict



def make_subset(config, train, test):
    # make confident subsets.
    # we're going to pick 80/20 split of the confident data for training + (pseudo)val,
    # so use a slightly larger beta here.

    train_inds = list(range(len(train)))
    train_inds, val_inds = train_test_split(train_inds, test_size=0.2)
    train_subset = train.select(train_inds)
    val_subset = train.select(val_inds)

    #actual_beta = config.cotrain_beta / 0.8
    #conf_inds_train, conf_inds_val = train_test_split(conf_inds, test_size=0.2)

    if config.cotrain_min_pct_per_class > 0:
        conf_inds_train = get_conf_inds_minppc(
            torch.LongTensor(train_subset['pseudolabel']),
            train_subset['features'],
            config.cotrain_beta,
            config.cotrain_min_pct_per_class
        )
        conf_inds_val = get_conf_inds_minppc(
            torch.LongTensor(val_subset['pseudolabel']),
            val_subset['features'],
            config.cotrain_beta,
            config.cotrain_min_pct_per_class
        )

    else:
        conf_inds_train = get_conf_inds(torch.LongTensor(train_subset['pseudolabel']),
                                        train_subset['features'], config.cotrain_beta)

        conf_inds_val = get_conf_inds(torch.LongTensor(val_subset['pseudolabel']),
                                      val_subset['features'], config.cotrain_beta)

    train_subset = train_subset.select(conf_inds_train)
    val_subset = val_subset.select(conf_inds_val)


    lab = np.array(train_subset['label'])
    psl = np.array(train_subset['pseudolabel'])
    print("pseudolabel metrics on confident set:")
    print({"accuracy": accuracy_score(lab, psl), "balanced_accuracy": balanced_accuracy_score(lab, psl)})
    print(torch.bincount(torch.LongTensor(psl)))

    lab = np.array(val_subset['label'])
    psl = np.array(val_subset['pseudolabel'])
    print("pseudolabel metrics on confident set, val:")
    print({"accuracy": accuracy_score(lab, psl), "balanced_accuracy": balanced_accuracy_score(lab, psl)})

    lab = np.array(train['label'])
    psl = np.array(train['pseudolabel'])
    print("pseudolabel metrics on full set:")
    print({"accuracy": accuracy_score(lab, psl), "balanced_accuracy": balanced_accuracy_score(lab, psl)})

    # REPLACE LABELS WITH PSEUDOLABELS
    train_subset = train_subset.remove_columns(["label", "features"])
    train_subset = train_subset.rename_column("pseudolabel", "label")
    val_subset = val_subset.remove_columns(["label", "features"])
    val_subset = val_subset.rename_column("pseudolabel", "label")
    test = test.remove_columns(["label", "features"])
    test = test.rename_column("pseudolabel", "label")

    return train_subset, val_subset, test
