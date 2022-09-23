import os
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from pytorch_lightning import LightningModule
from typing import Optional
from torch.optim import AdamW
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
import datasets
from datetime import datetime

# taken and modified from lightning docs
class BERT(LightningModule):
    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        dataset_reader,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model_name_or_path = model_name_or_path
        self.num_labels = num_labels
        self.task_name = task_name
        print('loading model')
        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)

        if 'roberta' in model_name_or_path:
            module = self.model.roberta
            pooler = None
        elif 'deberta' in model_name_or_path:
            module = self.model.deberta
            pooler = self.model.pooler
        else:
            module = self.model.bert
            pooler = module.pooler

        # turn off everything (including embeddings)

        for param in module.parameters():
            param.requires_grad = False

        # comment these two blocks for linear-only
        # turn on the last layer and the pooler again.
        for param in module.encoder.layer[-1].parameters():
            param.requires_grad = True

        if pooler is not None:
            for param in pooler.parameters():
                param.requires_grad = True


        print('loaded model')
        self.dataset_reader = dataset_reader
        #self.metric = datasets.load_metric(
        #    "super_glue", self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        #)

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        accumulated = {'prediction': preds, 'label': labels}
        metrics = self.dataset_reader.compute_metric(accumulated)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_balanced_acc", metrics['balanced_accuracy'], prog_bar=True)
        self.log_dict(metrics)


    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        all_parameters = [(n,p) for (n,p) in model.named_parameters() if p.requires_grad]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in all_parameters if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in all_parameters if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def on_train_epoch_end(self, **kwargs):
        pass
