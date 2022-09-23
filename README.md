# Co-train large language models

This repo contains the code for our ICML 2022 paper [Co-training Improves Prompt-based Learning for Large Language Models](https://arxiv.org/abs/2202.00828) and updates / extensions, including tuning based on  [T-Few](https://github.com/r-three/t-few).

This code is useful for:
  - boosting the zero-shot and few-shot performance of large language models
  - distilling large models like GPT-3 and T0 into smaller task-specific models.

Large parts of the repo are built on top of the excellent [T-Few](https://github.com/r-three/t-few) repository.

If you find this code useful, please consider citing our paper:
```
@inproceedings{lang2022co,
  title={Co-training improves prompt-based learning for large language models},
  author={Lang, Hunter and Agrawal, Monica N and Kim, Yoon and Sontag, David},
  booktitle={International Conference on Machine Learning},
  pages={11985--12003},
  year={2022},
  organization={PMLR}
}
```

## Setup
```
conda create -n cotrain
conda activate cotrain
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

## Zero-shot Co-training BERT and T0-3B
### With T-Few
Since the publication of our ICML paper, T-Few has emerged as a better technique for fine-tuning T0 than soft prompt tuning.
We have included code for co-training T0 (using T-Few) with BERT (using regular head tuning).

| Method/Model | RTE | CB |
| ----  | ---- | ---- |
| T0-3B (no training) | 62.1 | 51.8 |
| T0-3B + **co-training** | 86.1 (0.6) | 78.9 (9.5) |
| DeBERTa-large + **co-training** | 87.1 (0.3) | 79.3 (9.4) |

The median performances of cotrained T0 and BERT on CB are 82.1 and 85.7, respectively. The large standard deviation is because 2/5 seeds get stuck at 67% accuracy for both models; even these low seeds are near the best performance of cotraining with soft-prompt tuning.

#### Reproducing
To run co-training for all seeds and datasets:
```
CUDA_VISIBLE_DEVICES=0 ./bin/cotrain_tfew.sh
```
Once this is finished, we can look in ```dev_scores.json.<model>``` to get mean performance across seeds after 5 iterations:
```
cat exp_out/cotrain_ia3_rte_seed*_round5/dev_scores.json.bert | cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
cat exp_out/cotrain_ia3_rte_seed*_round5/dev_scores.json.t0 | awk 'NR % 2 == 0' |cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
cat exp_out/cotrain_ia3_cb_seed*_round5/dev_scores.json.bert | cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
cat exp_out/cotrain_ia3_cb_seed*_round5/dev_scores.json.t0 | awk 'NR % 2 == 0' |cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
```

To just do one run:
```
CUDA_VISIBLE_DEVICES=0 dataset=rte seed=0 t03b.json+${dataset}.json+ia3.json -k exp_name=cotrain_ia3_${dataset}_seed${seed} seed=${seed} few_shot=False allow_skip_exp=True train_template_idx=0 eval_template_idx=0 bert_name=microsoft/deberta-large-mnli bert_epochs=40 eval_epoch_interval=1
```

**Note**: The performance is sensitive to the choice of prompt for T0 (`train_template_idx` / `eval_template_idx`), since this dictates the quality of the initial pseudo-labeled data used for co-training. By default the code uses the first template.

### With Soft Prompt Tuning
This is the original method from our ICML paper, which used soft prompt tuning since T-Few had not been released.

| Method/Model | RTE | CB |
| ----  | ---- | ---- |
| T0-3B + **co-training** | 84.8 (0.8) | 64.6 (2.4) |
| DeBERTa-large + **co-training** | 86.4 (0.7) | 72.9 (1.3) |

#### Reproducing
To run all seeds and datasets:
```
CUDA_VISIBLE_DEVICES=0 ./bin/cotrain_spt.sh
```
Once this is finished, we can look in ```dev_scores.json.<model>``` to get mean performance across seeds after 5 iterations:
```
cat exp_out/cotrain_spt_rte_seed*_round5/dev_scores.json.bert | cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
cat exp_out/cotrain_spt_rte_seed*_round5/dev_scores.json.t0 | awk 'NR % 2 == 0' |cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
cat exp_out/cotrain_spt_cb_seed*_round5/dev_scores.json.bert | cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
cat exp_out/cotrain_spt_cb_seed*_round5/dev_scores.json.t0 | awk 'NR % 2 == 0' |cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
```

To just do one run:
```
CUDA_VISIBLE_DEVICES=0 dataset=rte seed=0 python -m src.cotrain -c t03b.json+${dataset}.json+prompt_tuning-10_prompts.json -k exp_name=cotrain_spt_${dataset}_seed${seed} seed=${seed} few_shot=False allow_skip_exp=True train_template_idx=0 eval_template_idx=0 bert_name=microsoft/deberta-large-mnli bert_epochs=40 eval_epoch_interval=1 prompt_tuning_num_prefix_emb=20 prompt_tuning_decoder=False num_steps=30000 prompt_tuning_init_with_pad=True cotrain_load_best=True batch_size=16 grad_accum_factor=2
```

The large number of steps for soft prompt tuning here is key to obtaining good performance.
Replacing SPT with T-Few thus maintains (or improves) the performance while being much more efficient due to requiring fewer steps.

### Using your own dataset
1. Create a dataset reader for your dataset in `src/data/dataset_readers.py` inheriting from `BaseDatasetReader`. Your dataset reader should set `self.templates` with appropriate templates to use with T0. See `HSwagReader` for a good example to follow. **Note**: the code uses `validation` as the name of the test split because, following other work, we report test performance on the public SuperGLUE validation sets. Make sure your test split is called `validation`. The co-training code samples a separate validation set for you already.
2. Add your reader to the `get_dataset_reader` function in `src/data/dataset_readers.py`
3. Add a config file in `configs/<your-dataset-name>.json`. `configs/rte.json` is a good one to copy.
4. Tell BERT how to tokenize your data by setting `task_text_field_map` for your task in `src/data/dataset_modules.py`


## Co-training BERT and GPT-3
This code is useful for distilling the outputs of GPT-3 into a smaller performant model.

| Method/Model                | RTE        | CB         | TREC       |
| ----                        | ----       | ----       | ---        |
| Label model (no cotrain)    | 62.8       | 76.8       | 77.2       |
| Label model **+ cotrain**   | 67.2 (1.3) | 82.1 (2.3) | 79.2 (1.8) |
| DeBERTa-large **+ cotrain** | 80.1 (4.2) | 84.6 (1.4) | 81.6 (1.6) |

These results differ from Table 1 in the paper because we [replaced](https://github.com/clinicalml/cotrain-prompting/blob/7747e6a4092713f6f9b9e724889aa51c1cba7e7c/src/cotrain_gpt.py#L93) the more sensitive confidence-based data selection for the label model by using the cut statistic on the BERT representations in each iteration. This selects higher-quality pseudo-labeled training data based on the label model pseudolabels and removes the need to set a constraint on the minimum label frequency.

#### Reproducing
To run all seeds and datasets:
```
CUDA_VISIBLE_DEVICES=0 ./bin/cotrain_gpt.sh
```
Once this is finished, we can look in ```dev_scores.json.<model>``` to get mean performance across seeds after 5 iterations:
```
cat exp_out/cotrain_gpt_rte_seed*_round5/dev_scores.json.bert | cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
cat exp_out/cotrain_gpt_rte_seed*_round5/dev_scores.json.lm | awk 'NR % 2 == 0' |cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
cat exp_out/cotrain_gpt_cb_seed*_round5/dev_scores.json.bert | cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
cat exp_out/cotrain_gpt_cb_seed*_round5/dev_scores.json.lm | awk 'NR % 2 == 0' |cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
cat exp_out/cotrain_gpt_trec_seed*_round5/dev_scores.json.bert | cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
cat exp_out/cotrain_gpt_trec_seed*_round5/dev_scores.json.lm | awk 'NR % 2 == 0' |cut -d':' -f 2 | cut -d ',' -f 1 | jq -s add/length;
```



### Using your own dataset
1. Get GPT-3 (or other LLM) probabilities for each output token in your desired vocabulary (i.e., the feature set of tokens you want to use for the label model). For each input example, you should have a `num_prompts x num_tokens` matrix. Turn this into a vector with `.reshape(-1)` and add it as a new column to your Huggingface dataset. **Note:** make sure the initial verbalizer tokens are the first columns (see paper Figure 2).
3. Obtain calibrate-before-use output matrix for each prompt and add it to `CBU_MAT` in `src/cotrain_gpt.py`. This should be `num_prompts x num_initial_verbalizer_tokens`. Each row corresponds to the diagonal of the initial calibration matrix.
4. Add a config file for you dataset to `configs/gpt-<your-dataset-name>.json`. You can copy `gpt-trec`, but update the config with the number of prompts you used.
5. Add your dataset to `get_dataset_reader` in `src/data/dataset_readers.py` (map it to `GPTReader`)
6. Tell BERT how to tokenize your data by setting `task_text_field_map` for your task in `src/data/dataset_modules.py`

