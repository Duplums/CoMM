import torch
from torch import nn
from typing import Dict
from pl_modules.base import BaseModel
from losses.cross_self_loss import CrossSelfLoss


class CrossSelf(BaseModel):
    """Multi-modal model representation learning for 2 modalities using
    cross-modality constraints + SSL constraint on each one based on InfoNCE. """

    def __init__(self,
                 enc1: nn.Module,
                 enc2: nn.Module,
                 head1: nn.Module,
                 head2: nn.Module,
                 optim_kwargs: Dict,
                 loss_kwargs: Dict):
        """
        Args:
            enc1: 1st encoder (e.g. ViT or ResNet50)
            enc2: 2dn encoder (e.g. Transformer)
            optim_kwargs: Optimization hyper-parameters to train CrossSelf
            head1: projection head for SSL constraint on 1st encoder
            head2: projection head for SSL constraint on 2nd encoder
            loss_kwargs: kwargs including `ssl_scale` (scaling hparams between SSL + cross-modality constraints)
            and `temperature` hparam for InfoNCE loss.
        """
        super().__init__(optim_kwargs)

        self.enc1 = enc1
        self.enc2 = enc2
        self.head1 = head1
        self.head2 = head2
        self.loss = CrossSelfLoss(**loss_kwargs)

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
                output = [self.enc1(X_[0]).view(n, -1), self.enc2(X_[1]).view(n, -1)]
                output = torch.cat(output, dim=-1)
                X.extend(output.detach().cpu())
                y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)

    def forward(self, mod1, mod2, mod1_aug, mod2_aug):
        outputs = {
            "mod1_embed": self.head1(self.enc1(mod1)),
            "mod2_embed": self.head2(self.enc2(mod2)),
            "mod1_aug1_embed": self.head1(self.enc1(mod1_aug[0])),
            "mod1_aug2_embed": self.head1(self.enc1(mod1_aug[1])),
            "mod2_aug1_embed": self.head2(self.enc2(mod2_aug[0])),
            "mod2_aug2_embed": self.head2(self.enc2(mod2_aug[1])),
        }
        return outputs

