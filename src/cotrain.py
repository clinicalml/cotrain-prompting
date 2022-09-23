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
from src.data import CotrainDataModule, get_dataset_reader, PretrainDataModule, BERTDataModule
from src.models.EncoderDecoder import EncoderDecoder
from src.models.BERT import BERT
from src.models.modify_model import modify_transformer
from src.utils.Config import Config
from src.utils.util import ParseKwargs, set_seeds
from src.utils.cotrain_utils import (
    get_conf_inds,
    get_conf_inds_per_class,
    get_dsdict_prompt,
    get_dsdict_bert,
)


def get_transformer(config, modify=True):
    print(config.origin_model)
    tokenizer = AutoTokenizer.from_pretrained(config.origin_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.origin_model, low_cpu_mem_usage=True)

    tokenizer.model_max_length = config.max_seq_len
    if modify:
        model = modify_transformer(model, config)
    return tokenizer, model




def main(config):
    """
    Trains the model

    :param config:
    :return:
    """


    """
    # todo: remove. for debuggin.
    ds_dict = dataset_reader.get_full_orig_dataset()
    print(ds_dict)
    bertname = 'bert-base-uncased'
    dm = BERTDataModule(bertname, 'cb', ds_dict)
    #dm.prepare_data()
    #dm.setup('fit')
    bert = BERT(
        model_name_or_path=bertname,
        num_labels=dm.num_labels,
        task_name=dm.task_name,
    )
    trainer = Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1
    )
    trainer.fit(bert, datamodule=dm)
    ds_dict = get_dsdict_bert(
        config, args, dataset_reader, bert
    )
    # end todo remove
    """

    # in the prompt-tuning case, don't call modify() on
    # the model for the very first step because adding the noisy prompt
    # terms messes up the outputs.
    # in the paper we add [PAD] instead of doing this but that's not necessary.

    if config.model_modifier == 'prompt-tuning' and not config.prompt_tuning_init_with_pad:
        tokenizer, model = get_transformer(config, modify=False)
    else:
        tokenizer, model = get_transformer(config, modify=True)

    dataset_reader = get_dataset_reader(config)

    # this wrapper only uses the dataset reader for metrics
    model = EncoderDecoder(config, tokenizer, model, dataset_reader)

    original_exp_dir = config.exp_dir
    original_exp_name = config.exp_name
    original_beta = config.cotrain_beta

    for t in range(5):
        config.exp_dir = f'{original_exp_dir}_round{t+1}'
        config.exp_name = f'{original_exp_name}_round{t+1}'
        config.set_exp_dir() # calls mkdir

        # todo: make 0.1 a configurable amt
        config.cotrain_beta = original_beta + 0.1*t

        # get confidently-pseudolabeled data from T0
        ds_dict = get_dsdict_prompt(
            config, dataset_reader,
            tokenizer, model.to('cuda').model,
        )

        #import pdb
        #pdb.set_trace()
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
            learning_rate=config.bert_lr,
            weight_decay=config.bert_wd,
            train_batch_size=16,
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
            val_check_interval=0.5,
        )
        print("loaded trainer, starting fit")
        trainer.fit(bert, datamodule=dm)

        # todo: load the best model
        print(f"GOT BEST CHECKPOINT PATH {trainer.checkpoint_callback.best_model_path}")

        if config.cotrain_load_best:
            bert = BERT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        # get confidently-pseudolabeled data from bert model
        ds_dict = get_dsdict_bert(
            config, dataset_reader, bert
        )

        # get rid of bert model and clear cuda cache
        del trainer
        del bert
        gc.collect()
        torch.cuda.empty_cache()

        # reset the t0 model every iter before training it
        tokenizer, model = get_transformer(config)
        dataset_reader = get_dataset_reader(config)

        # wrap up pseudolabeled data in a data module for T0
        datamodule = CotrainDataModule(config, tokenizer, ds_dict, dataset_reader)

        # this wrapper only uses the dataset reader for metrics
        model = EncoderDecoder(config, tokenizer, model, dataset_reader,
                               track_metric="balanced_accuracy")

        """
        ckpt_dir = os.path.join(config.exp_dir, 't0')

        checkpoint_callback = ModelCheckpoint(dirpath=ckpt_dir,
                                              save_top_k=1,
                                              monitor="val_balanced_acc",
                                              mode='max')
        """

        # fine-tune T0
        logger = TensorBoardLogger(config.exp_dir, name="log")
        trainer = Trainer(
            enable_checkpointing=False,
            accelerator='gpu',
            devices=torch.cuda.device_count(),
            precision=config.compute_precision,
            amp_backend="native",
            strategy=config.compute_strategy if config.compute_strategy != "none" else None,
            logger=logger,
            log_every_n_steps=4,
            max_steps=config.num_steps,
            min_steps=config.num_steps,
            num_sanity_val_steps=-1 if config.eval_before_training else 0,
            check_val_every_n_epoch=config.eval_epoch_interval,
            accumulate_grad_batches=config.grad_accum_factor,
            gradient_clip_val=config.grad_clip_norm,
            #callbacks=[checkpoint_callback]
        )

        trainer.fit(model, datamodule)

        cfg_with_ckpt = copy.deepcopy(config)
        cfg_with_ckpt.load_weight = os.path.join(config.exp_dir, "best.pt")

        print(f"GOT BEST metric {model.best_metric_val}")

        if config.cotrain_load_best:
            del model
            del tokenizer
            del trainer
            del datamodule
            gc.collect()
            torch.cuda.empty_cache()

            # load the best model to use for the next iteration.
            # according to (pseudo)-validation performance.
            tokenizer, model = get_transformer(cfg_with_ckpt)
            model = EncoderDecoder(cfg_with_ckpt, tokenizer, model, dataset_reader)

    # get the ds dict for the prompt model one last time so that
    # we write out the metrics for this run.
    ds_dict = get_dsdict_prompt(
        config, dataset_reader,
        tokenizer, model.to('cuda').model,
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
