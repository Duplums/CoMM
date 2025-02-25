import torch
import math
import sys
from torch import nn
from typing import Dict, List
# local imports
from pl_modules.base import BaseModel
from losses.cmc_loss import CMCLoss


class CMC(BaseModel):
    """Contrastive Multiview Coding [1] for multi-modal representation learning.
     It learns using cross-modality constraints between all possible pairs of modalities (2 among `n`).

     [1] Contrastive Multiview Coding, Tian et al., ECCV 2020
     """

    def __init__(self,
                 encoders: List[nn.Module],
                 heads: List[nn.Module],
                 optim_kwargs: Dict,
                 loss_kwargs: Dict):
        """
        Args:
            encoders: List of encoders (e.g. ViT, ResNet50, Transformer)
            heads: List of projection heads (one for each encoder)
            optim_kwargs: Optimization hparams to train CrossSelf
            loss_kwargs: kwargs including `temperature` hparam for the InfoNCE losses.
        """
        super().__init__(optim_kwargs)
        self.encoders = nn.ModuleList(encoders)
        self.heads = nn.ModuleList(heads)
        self.loss = CMCLoss(**loss_kwargs)

    def training_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        out_dict = self.loss(outputs)
        loss = out_dict['loss']
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
        self.log_dict(out_dict, on_step=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        out_dict = self.loss(outputs)
        val_loss = out_dict['loss']
        self.log_dict({"val_%s"%k: v for k, v in out_dict.items()}, on_epoch=True, sync_dist=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        out_dict = self.loss(outputs)
        test_loss = out_dict['loss']
        self.log_dict({"test_%s"%k: v for k, v in out_dict.items()}, on_epoch=True, sync_dist=True)
        return test_loss

    def forward(self, x: List[torch.Tensor]):
        outputs = [head(enc(xi)) for (enc, head, xi) in zip(self.encoders, self.heads, x)]
        return outputs

    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        """
           Extract features from both modalities
            Args:
               loader: Dataset loader to serve ``(X, y)`` tuples.
            Returns:
                Pair (z, y) corresponding to extracted features and corresponding labels
        """
        X, y = [], []
        for X_, y_ in loader:
            y_ = y_.to(self.device)
            X_ = [x.to(self.device) if isinstance(x, torch.Tensor) else x for x in X_]
            n = len(X_[0])
            with torch.inference_mode():
                # compute output
                output = [enc(xi).view(n, -1) for (enc, xi) in zip(self.encoders, X_)]
                output = torch.cat(output, dim=-1)
                X.extend(output.detach().cpu())
                y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)

