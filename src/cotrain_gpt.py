import os
import gc
import torch
import argparse
import datasets
import logging
import numpy as np
import time
import copy
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from src.data import (
    CotrainDataModule,
    get_dataset_reader,
    PretrainDataModule,
    BERTDataModule,
    LabelModelDataModule
)
from src.models.LabelModel import LabelModel
from src.models.BERT import BERT
from src.utils.Config import Config
from src.utils.util import ParseKwargs, set_seeds
from src.utils.cotrain_utils import (
    get_conf_inds,
    get_conf_inds_per_class,
    get_dsdict_prompt,
    get_dsdict_bert,
    get_dsdict_labelmodel
)

# probabilities obtained from calibrate before use
# for our 4 1-shot prompts.
# should be num_prompts x num_label_tokens.
CBU_MAT = {
    'gpt-rte': np.array([[0.58553599, 0.41446401],
                     [0.54688961, 0.45311039],
                     [0.67715019, 0.32284981],
                     [0.52186163, 0.47813837]]),
    'gpt-cb': np.array([[0.48230783, 0.37372809, 0.14396408],
                    [0.43594281, 0.2865988 , 0.27745839],
                    [0.35556505, 0.27506816, 0.3693668 ],
                    [0.42172063, 0.22625599, 0.35202338]]),
    'gpt-trec': np.array([[0.06482382, 0.02168737, 0.40624225, 0.03669339, 0.22755683, 0.24299634],
                      [0.0810809 , 0.01194075, 0.5113626 , 0.07378576, 0.1730886 , 0.14874139],
                      [0.08388882, 0.02699317, 0.44079022, 0.25573598, 0.16026982, 0.03232199],
                      [0.02862431, 0.04297541, 0.37211708, 0.12663035, 0.21889563, 0.21075722]]),
}


def main(config):
    # make sure we have the calibration parameters set correctly
    assert(CBU_MAT[config.dataset].shape[0] == config.gpt_num_prompts)

    dataset_reader = get_dataset_reader(config)

    # load the dataset once so we can compute the number of features
    # we need to use in the label model.
    dm = LabelModelDataModule(config, dataset_reader.get_full_orig_dataset())
    dm.setup('fit')
    num_feat = torch.tensor(dm.dataset['train'][0]['feat']).shape[1]

    # look up initialization parameters
    # from CBU based on config.dataset.
    cbu_params = CBU_MAT[config.dataset]
    model = LabelModel(config,
                       num_feat=num_feat,
                       cbu_params=cbu_params,
                       dataset_reader=dataset_reader,
    )

    original_exp_dir = config.exp_dir
    original_exp_name = config.exp_name
    original_beta = config.cotrain_beta

    bert = None

    for t in range(5):
        config.exp_dir = f'{original_exp_dir}_round{t+1}'
        config.exp_name = f'{original_exp_name}_round{t+1}'
        config.set_exp_dir() # calls mkdir

        # todo: make 0.1 a configurable amt
        config.cotrain_beta = original_beta + 0.1*t

        # get confidently-pseudolabeled data from GPT-3 outputs
        ds_dict = get_dsdict_labelmodel(
            config, dataset_reader,
            model.to('cuda'),
            bert_name=config.bert_name, # use bert features for cutstat
            bert_model=bert, # use fine-tuned model for iter > 1
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

        ### train BERT model ####
        ### then switch back to Prompt model ####

        # wrap up this pseudolabeled data in a data module for BERT
        dm = BERTDataModule(config.bert_name, config.dataset, ds_dict)

        bert = BERT(
            model_name_or_path=config.bert_name,
            num_labels=dm.num_labels,
            task_name=dm.task_name,
            dataset_reader=dataset_reader,
            warmup_steps=500,
            weight_decay=config.bert_wd,
            learning_rate=config.bert_lr
        )

        ckpt_dir=os.path.join(config.exp_dir, 'bert')
        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                              save_top_k=1,
                                              monitor="val_balanced_acc",
                                              mode='max')
        trainer = Trainer(
            max_epochs=config.bert_epochs,
            accelerator="gpu",
            devices=1,
            callbacks=[checkpoint_callback],
            check_val_every_n_epoch=1,
        )
        print("loaded trainer, starting fit")
        trainer.fit(bert, datamodule=dm)

        # load the best model
        print(f"GOT BEST CHECKPOINT PATH {trainer.checkpoint_callback.best_model_path}")

        if config.cotrain_load_best:
            bert = BERT.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            )

        # get confidently-pseudolabeled data from bert model
        ds_dict = get_dsdict_bert(
            config, dataset_reader, bert
        )

        # reset the label model every iter before training it
        model = LabelModel(
            config,
            num_feat=num_feat,
            cbu_params=cbu_params,
            dataset_reader=dataset_reader,
        )

        dm = LabelModelDataModule(config, ds_dict)

        ckpt_dir=os.path.join(config.exp_dir, 'labelmodel')
        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                              save_top_k=1,
                                              monitor="val_balanced_acc",
                                              mode='max')

        # fine-tune label model
        trainer = Trainer(
            max_epochs=config.num_epochs,
            accelerator='gpu',
            devices=1,
            check_val_every_n_epoch=config.eval_epoch_interval,
            callbacks=[checkpoint_callback],
        )

        trainer.fit(model, dm)

        print(f"got best model path {trainer.checkpoint_callback.best_model_path}")

        if config.cotrain_load_best:
            load_path = trainer.checkpoint_callback.best_model_path
            del model
            del trainer
            del dm
            gc.collect()

            torch.cuda.empty_cache()
            model = LabelModel.load_from_checkpoint(load_path)

    # get the ds dict for the label model one last time so that
    # we write out the metrics for the last run.
    ds_dict = get_dsdict_labelmodel(
        config, dataset_reader,
        model.to('cuda')
    )
    # loop is done.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    datasets.disable_caching()

    # configure logging at the root level of Lightning
    #logging.getLogger("pytorch_lightning").setLevel(logging.DEBUG)

    config = Config(args.config_files, args.kwargs)
    print(f"Start experiment {config.exp_name}")
    # Setup config
    assert config.compute_strategy in ["none", "ddp", "deepspeed_stage_3_offload", "deepspeed_stage_3"]
    if config.fishmask_mode == "create":
        print("Detecting fishmask_mode=create, override batch_size, num_step, fishmask_path")
        config.batch_size = 1
        config.num_steps = config.num_shot
        config.eval_before_training = False
        config.fishmask_path = None

    print(config.to_json())

    if config.allow_skip_exp and os.path.exists(config.finish_flag_file):
        print(f"Skip finished experiment {config.exp_name}")
    else:
        print(f"Mark experiment {config.exp_name} as claimed")
        with open(config.finish_flag_file, "a+") as f:
            f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + "\n")
        set_seeds(config.seed)
        main(config)
