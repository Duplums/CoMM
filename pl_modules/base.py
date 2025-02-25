from pytorch_lightning import LightningModule
from torch import Tensor
from typing import Tuple, Dict
from abc import ABC, abstractmethod
import torch
import math
import sys
from utils import set_weight_decay_per_param, LinearWarmupCosineAnnealingLR


class BaseModel(ABC, LightningModule):
    """
        Base model for Self-Supervised Learning (SSL), Vision-Language (VL) or Language-Guided (LG) models.
        We expect any `BaseModel` to implement a features extractor.
    """

    def __init__(self, optim_kwargs: Dict):
        super().__init__()
        self.optim_kwargs = optim_kwargs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            set_weight_decay_per_param(
                self, weight_decay=self.optim_kwargs["weight_decay"]),
            lr=self.optim_kwargs["lr"])

        if "lr_scheduler" in self.optim_kwargs:
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.optim_kwargs["lr_scheduler"]["warmup_epochs"],
                max_epochs=self.trainer.max_epochs,
                warmup_start_lr=self.optim_kwargs["lr_scheduler"]["start_warmup_value"],
                eta_min=self.optim_kwargs["lr_scheduler"]["final_value"]
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        return optimizer

    def training_step(self, batch, batch_idx):
        outputs = self.forward(*batch)
        out_dict = self.loss(outputs)
        loss = out_dict['loss']
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
        self.log_dict(out_dict, on_step=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(*batch)
        out_dict = self.loss(outputs)
        val_loss = out_dict['loss']
        self.log_dict({"val_%s"%k: v for k, v in out_dict.items()}, on_epoch=True, sync_dist=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        outputs = self.forward(*batch)
        out_dict = self.loss(outputs)
        test_loss = out_dict['loss']
        self.log_dict({"test_%s"%k: v for k, v in out_dict.items()}, on_epoch=True, sync_dist=True)
        return test_loss

    @abstractmethod
    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs) \
            -> Tuple[Tensor, Tensor]:
        """
        Extract global average pooled visual features.
        Args:
            loader: Dataset loader to serve ``(image, label)`` tuples.
        Returns:
            Pair (X,y) corresponding to extracted features and corresponding labels
        """
        pass

