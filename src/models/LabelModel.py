import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from src.utils.get_scheduler import get_scheduler
from statistics import mean
from transformers import get_linear_schedule_with_warmup

# taken and modified from lightning docs
class LabelModel(LightningModule):
    def __init__(
        self,
        config,
        num_feat: int,
        cbu_params: torch.Tensor,
        dataset_reader,
    ):
        super().__init__()

        self.config = config

        self.num_feat = num_feat
        self.dataset_reader = dataset_reader

        self.sgm = torch.nn.Sigmoid()
        self.sgm_scale = 1

        nprompts, nlabels = cbu_params.shape
        self.num_prompts = nprompts
        self.num_labels = nlabels

        self.save_hyperparameters()

        self.prompt_weights = torch.nn.Parameter(torch.ones(nprompts),
                                                 requires_grad=True)
        self.linears = torch.nn.ModuleList()
        for l in range(nprompts):
            self.linears.append(torch.nn.Linear(self.num_feat,
                                                nlabels,
                                                bias=True))


        for l in range(nprompts):
            self.linears[l].weight.data.zero_()
            self.linears[l].weight.data[torch.arange(nlabels), torch.arange(nlabels)] = torch.tensor(1 / cbu_params[l]).float()
            self.linears[l].bias.data.zero_()


    def forward(self, x):
        # x is N x nprompt x nfeat
        promptouts = []
        for l in range(self.num_prompts):
            out = self.linears[l](x[:,l,:])
            promptouts.append(out[:,:,None]) # N x ncls x 1
        out = torch.cat(promptouts, dim=2) # N x ncls x nprompts
        out = self.sgm(self.sgm_scale * out)
        out = out * self.prompt_weights[None, None, :]
        out = out.sum(dim=2)
        return out


    def training_step(self, batch, batch_idx):
        logits = self(batch['feat'])
        labels = batch['labels']
        loss = F.cross_entropy(logits, labels)
        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self(batch['feat'])
        labels = batch['labels']
        val_loss = F.cross_entropy(logits, labels)

        if self.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.num_labels == 1:
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
        optimizer = AdamW(self.parameters(), lr=self.config.lr,
                         weight_decay=self.config.weight_decay)

        """
        scheduler = get_scheduler(optimizer, self.config)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
        """

        """
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        """
        return optimizer #[optimizer], [scheduler]

    def on_train_epoch_end(self, **kwargs):
        pass
