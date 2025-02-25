import torch.nn.functional as func
import torch
import torch.nn as nn
from utils import all_gather_batch_with_grad


class InfoNCE(nn.Module):
    """
    Normalized Temperature Cross-Entropy Loss for Contrastive Learning
    Refer for instance to:
    Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
    A Simple Framework for Contrastive Learning of Visual Representations, ICML 2020
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.INF = 1e8

    def forward(self, outputs):
        """
        :param outputs: Dict
            Dictionary with keys:
                - `aug1_embed`, shape (bsize, feature_dim), 1st aug. embedded view
                - `aug2_embed`, shape (bsize, feature_dim), 2nd aug. embedded view
        :return: {"loss": torch.Tensor(float), "ssl_acc": torch.Tensor(float)}
        """
        z1, z2 = outputs["aug1_embed"], outputs["aug2_embed"]
        z1 = func.normalize(z1, p=2, dim=-1) # dim [N, D]
        z2 = func.normalize(z2, p=2, dim=-1) # dim [N, D]
        z1, z2 = all_gather_batch_with_grad([z1, z2])
        N = len(z1)
        sim_zii= (z1 @ z1.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zjj = (z2 @ z2.T) / self.temperature # dim [N, N] => Upper triangle contains incorrect pairs
        sim_zij = (z1 @ z2.T) / self.temperature # dim [N, N] => the diag contains the correct pairs (i,j)
        # 'Remove' the diag terms by penalizing it (exp(-inf) = 0)
        sim_zii = sim_zii - self.INF * torch.eye(N, device=z1.device)
        sim_zjj = sim_zjj - self.INF * torch.eye(N, device=z1.device)
        sim_Z = torch.cat([
            torch.cat([sim_zij, sim_zii], dim=1),
            torch.cat([sim_zjj, sim_zij.T], dim=1)], dim=0)
        log_sim_Z = func.log_softmax(sim_Z, dim=1)
        loss = - torch.diag(log_sim_Z).mean()
        # compute SSL accuracy
        with torch.no_grad():
            pred = torch.argmax(sim_zij, dim=1)
            correct = pred.eq(torch.arange(N, device=z1.device)).sum()
            acc = 100 * correct / N

        return {"loss": loss, "ssl_acc": acc}

    def __str__(self):
        return "{}(temp={})".format(type(self).__name__, self.temperature)