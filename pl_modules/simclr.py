from torch import nn
import torch
from collections import OrderedDict
from typing import Dict
# Local imports
from pl_modules.base import BaseModel
from losses.infonce import InfoNCE


class Siamese(BaseModel):
    """ Abstract Siamese network (objective function is unspecified)
    Build a DNN model with a 3-layers MLP as projector as in [1].
    It encodes a pair of data and outputs a pair of representations through
    a single encoder (hence `Siamese` model).

    [1] A Simple Framework for Contrastive Learning of Visual Representations, Chen et al., ICML 2020
    """

    def __init__(self,
                 visual: nn.Module,
                 visual_projection: nn.Module,
                 optim_kwargs: Dict):
        """
        Args:
            visual: Vision encoder (e.g. ViT or ResNet50)
            visual_projection: MLP projector to the latent space
            optim_kwargs: Optimization hyper-parameters for training
        """
        super(Siamese, self).__init__(optim_kwargs)

        # create the encoder
        self.encoder = visual

        # build a 3-layers projector
        self.head = visual_projection

    @staticmethod
    def _build_mlp(in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def extract_features(self, loader: torch.utils.data.DataLoader, **kwargs):
        """
           Extract global average pooled visual features.
           Args:
               loader: Dataset loader to serve ``(X, y)`` tuples.
               kwargs: given to `encoder.forward()`
           Returns: Pair (Z,y) corresponding to extracted features and corresponding labels
        """
        X, y = [], []
        for X_, y_ in loader:
            if isinstance(X_, list):
                X_ = X_[0]
            X_ = X_.to(self.device)
            y_ = y_.to(self.device)
            with torch.inference_mode():
                # compute output
                output = self.encoder(X_, **kwargs)
                X.extend(output.view(len(output), -1).detach().cpu())
                y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)

    def forward(self, x1, x2):
        """
        Input:
            x1, x2: batch of tensors
        Output:
            z1, z2: latent representations
        """

        # compute features for all images
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        z1 = self.head(z1)
        z2 = self.head(z2)
        return {'aug1_embed': z1,
                'aug2_embed': z2}


class SimCLR(Siamese):
    def __init__(self,
                 visual: nn.Module,
                 visual_projection: nn.Module,
                 optim_kwargs: Dict,
                 loss_kwargs: Dict):
        """
        Args:
            visual: Vision encoder (e.g. ViT or ResNet50)
            visual_projection: MLP projector to the latent space
            optim_kwargs: Optimization hyper-parameters for training
            loss_kwargs: Hyper-parameters for the InfoNCE loss.
        """
        super().__init__(visual, visual_projection, optim_kwargs)
        self.loss = InfoNCE(**loss_kwargs)

