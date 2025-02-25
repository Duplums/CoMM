import torch
import math
from torch import nn
from typing import List
from itertools import combinations
# local imports
from losses.infonce import InfoNCE

class CMCLoss(nn.Module):
    """ Contrastive Multiview Coding model for multi-modal representation learning

    """
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.infonce = InfoNCE(temperature=temperature)

    def forward(self, inputs: List[torch.Tensor]):
        """
        :param inputs: list of embeddings (one per modality)
        :return: cross-modality loss between all pairs of embeddings
        """
        out_dict = dict(loss=0.)
        n = len(inputs)
        for (i1, i2) in combinations(range(n), 2):
            loss = self.infonce(dict(aug1_embed=inputs[i1], aug2_embed=inputs[i2]))
            out_dict["loss"] += loss["loss"]
            out_dict[f"ssl_acc_{i1}_{i2}"] = loss["ssl_acc"]
        out_dict["loss"] /= math.comb(n, 2)
        return out_dict