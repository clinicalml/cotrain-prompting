import torch
import numpy as np
from pytorch_lightning import LightningDataModule
import datasets
from typing import Optional
from transformers import AutoTokenizer

class FinetuneDataModule(LightningDataModule):
    def __init__(self, config, tokenizer, dataset_reader):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        if self.config.few_shot:
            _ = self.dataset_reader.read_few_shot_dataset()

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if self.config.few_shot:
            self.train_dataset = self.dataset_reader.read_few_shot_dataset()
        else:
            self.train_dataset = self.dataset_reader.read_orig_dataset("train")
        self.dev_dataset = self.dataset_reader.read_orig_dataset("validation")
        self.train_dataset = FinetuneDatasetWithTemplate(
            self.train_dataset, self.dataset_reader.get_train_template(), self.tokenizer
        )
        self.dev_dataset = FinetuneDatasetWithTemplate(
            self.dev_dataset, self.dataset_reader.get_eval_template(), self.tokenizer
        )
        print(f"Train size {len(self.train_dataset)}")
        print(f"Eval size {len(self.dev_dataset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            drop_last=True,
            num_workers=min([self.config.batch_size, self.config.num_workers]),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
        )


# for use *after* using a datasetreader to create a ds_dict
class CotrainDataModule(LightningDataModule):
    def __init__(self, config, tokenizer, ds_dict, dataset_reader):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.ds_dict = ds_dict
        self.dataset_reader = dataset_reader

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        self.train_dataset = self.ds_dict['train']
        self.dev_dataset = self.ds_dict['validation']
        if 'test' in self.ds_dict:
            self.test_dataset = self.ds_dict['test']
            self.test_dataset = FinetuneDatasetWithTemplate(
                self.test_dataset, self.dataset_reader.get_eval_template(), self.tokenizer
            )

        self.train_dataset = FinetuneDatasetWithTemplate(
            self.train_dataset, self.dataset_reader.get_train_template(), self.tokenizer
        )
        self.dev_dataset = FinetuneDatasetWithTemplate(
            self.dev_dataset, self.dataset_reader.get_eval_template(), self.tokenizer
        )

        print(f"Train size {len(self.train_dataset)}")
        print(f"Eval size {len(self.dev_dataset)}")
        print(f"Test size {len(self.test_dataset)}")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            drop_last=True,
            num_workers=min([self.config.batch_size, self.config.num_workers]),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
        )




class FinetuneDatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, templates, tokenizer, add_special_tokens=True):
        super().__init__()
        self.dataset = dataset
        self.templates = templates
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(self.templates, list):
            template = np.random.choice(self.templates)
        else:
            template = self.templates
        example = self.dataset[key]
        input_str, target_str = template.apply(example)

        answer_choices = template.get_answer_choices_list(example)
        if isinstance(input_str, list):
            input_ids = torch.cat(
                [
                    self.tokenizer(
                        input_field, return_tensors="pt", truncation=True, add_special_tokens=False
                    ).input_ids.squeeze(0)
                    for input_field in input_str[:-1]
                ]
                + [
                    self.tokenizer(
                        input_str[-1], return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
                    ).input_ids.squeeze(0)
                ]
            )
        else:
            input_ids = self.tokenizer(
                input_str, return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
            ).input_ids.squeeze(0)
        target_ids = self.tokenizer(
            target_str, return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
        ).input_ids.squeeze(0)
        answer_choices_ids = [
            self.tokenizer(
                answer_choice, return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
            ).input_ids.squeeze(0)
            for answer_choice in answer_choices
        ]
        label = torch.LongTensor([example["label"]])
        idx = torch.LongTensor([example["idx"]])
        return input_ids, target_ids, answer_choices_ids, label, idx


class PretrainDataModule(LightningDataModule):
    def __init__(self, config, tokenizer, dataset_reader):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader

    def setup(self, stage):
        self.train_datasets = self.dataset_reader.read_orig_dataset("train")
        self.base_templates = self.dataset_reader.get_template()
        self.train_datasets_withtemplate = []
        for index, train_dataset in enumerate(self.train_datasets):
            self.train_datasets_withtemplate.append(
                PretrainDatasetWithTemplate(train_dataset, self.base_templates[index], self.tokenizer)
            )

        self.train_dataset = torch.utils.data.ConcatDataset(self.train_datasets_withtemplate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=True),
            drop_last=True,
            num_workers=min([self.config.batch_size, self.config.num_workers]),
        )


class PretrainDatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, templates, tokenizer):
        super().__init__()
        self.dataset = dataset
        self.templates = templates
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(self.templates, list):
            template = np.random.choice(self.templates)
        else:
            template = self.templates
        example = self.dataset[key]
        input_target_str = template.apply(example)
        if len(input_target_str) == 2:
            input_str, target_str = input_target_str
            if target_str == "":
                target_str = "<NO LABEL>"
        else:
            input_str = "<NO INPUT>"
            target_str = "<NO LABEL>"
        input_ids = self.tokenizer(input_str, return_tensors="pt", truncation=True).input_ids.squeeze(0)
        target_ids = self.tokenizer(target_str, return_tensors="pt", truncation=True).input_ids.squeeze(0)
        return input_ids, target_ids


def create_collate_fn(pad_token_id, pretrain):
    def collate_fn(batch):
        if not pretrain:
            input_ids, target_ids, answer_choices_ids, labels, idx = zip(*batch)
        else:
            input_ids, target_ids = zip(*batch)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=pad_token_id)
        output_batch = {
            "input_ids": input_ids,
            "target_ids": target_ids,
        }

        if not pretrain:
            flat_answer_choice_ids = [choice for list_choices in answer_choices_ids for choice in list_choices]
            num_choice = [len(list_choices) for list_choices in answer_choices_ids]
            if max(num_choice) != min(num_choice):
                raise NotImplementedError("The collate_fn is not implmented for variable number of choices")
            flat_answer_choices_ids = torch.nn.utils.rnn.pad_sequence(
                flat_answer_choice_ids, batch_first=True, padding_value=pad_token_id
            )
            answer_choices_ids = flat_answer_choices_ids.view(len(answer_choices_ids), max(num_choice), -1).contiguous()
            labels = torch.cat(labels)
            idx = torch.cat(idx)
            output_batch.update(
                {
                    "answer_choices_ids": answer_choices_ids,
                    "labels": labels,
                    "idx": idx,
                }
            )

        return output_batch

    return collate_fn


# taken and modified from Lightning docs
class BERTDataModule(LightningDataModule):

    task_text_field_map = {
        "rte": ["premise", "hypothesis"],
        "boolq": ["passage", "question"],
        "cb": ["premise", "hypothesis"],
        "gpt-rte": ["sentence1", "sentence2"],
        "gpt-cb": ["premise", "hypothesis"],
        "gpt-trec": ["text"],
    }

    glue_task_num_labels = {
        "rte": 2,
        "boolq": 2,
        "cb": 3,
        "gpt-rte": 2,
        "gpt-cb": 3,
        "gpt-trec": 6
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "rte",
        ds_dict: dict = {},
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.dataset = ds_dict

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]

        # override for mnli-pretrained models
        if 'mnli' in model_name_or_path:
            self.num_labels = 3

        print("loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        print('loaded tokenizer')

    # called on every proc in DDP
    def setup(self, stage: str):
        print("in setup")
        for split in self.dataset.keys():
            if 'input_ids' not in self.dataset[split].features:
                self.dataset[split] = self.dataset[split].map(
                    self.convert_to_features,
                    batched=True,
                    remove_columns="label",
                )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        print(f"Train size {len(self.dataset['train'])}")
        if 'validation' in self.dataset:
            print(f"Eval size {len(self.dataset['validation'])}")

        #self.dataset = datasets.load_dataset("super_glue", self.task_name)

    # called on 1 proc in DDP
    def prepare_data(self):
        pass
        #datasets.load_dataset("super_glue", self.task_name)
        #AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset["train"],
                                           batch_size=self.train_batch_size,
                                           shuffle=True,
                                           num_workers=8)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset["validation"],
                                           batch_size=self.eval_batch_size,
                                           num_workers=8)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset["test"], batch_size=self.eval_batch_size,
                                           num_workers=8)

    def convert_to_features(self, example_batch, indices=None):
        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, padding='max_length', truncation=True
        )

        # todo jj: this will probably give a key error
        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = example_batch["label"]

        return features


class LabelModelDataModule(LightningDataModule):
    def __init__(
        self,
        config,
        ds_dict: dict = {},
    ):
        super().__init__()
        self.config = config
        self.dataset = ds_dict

    # called on every proc in DDP
    def setup(self, stage: str):
        print("in setup")
        for split in self.dataset.keys():
            self.columns = [c for c in self.dataset[split].column_names]
            self.dataset[split] = self.dataset[split].map(
                self.reshape_features,
                batched=False
            )

        print(f"Train size {len(self.dataset['train'])}")
        if 'validation' in self.dataset:
            print(f"Eval size {len(self.dataset['validation'])}")

        #self.dataset = datasets.load_dataset("super_glue", self.task_name)

    # called on 1 proc in DDP
    def prepare_data(self):
        pass
        #datasets.load_dataset("super_glue", self.task_name)
        #AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def reshape_features(self, example):
        example['feat'] = torch.tensor(example['feat']).reshape(self.config.gpt_num_prompts, -1).tolist()
        return example

    def create_collate_fn(self):
        def collate(batch):
            feat = torch.tensor([item['feat'] for item in batch])
            labels = torch.tensor([item['label'] for item in batch])
            return {'feat': feat, 'labels': labels}

        return collate

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["train"],
            collate_fn=self.create_collate_fn(),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["validation"],
            collate_fn=self.create_collate_fn(),
            batch_size=self.config.batch_size,
            num_workers=0,
            shuffle=False
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset["test"],
            collate_fn=self.create_collate_fn(),
            batch_size=self.config.batch_size,
            num_workers=0,
            shuffle=False
        )
