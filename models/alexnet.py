from torchvision.models import AlexNet
import torch.nn as nn
import torch

class AlexNetEncoder(AlexNet):
    """AlexNet backbone for features representation learning."""
    def __init__(self, latent_dim: int = 512, dropout: float = 0.5, global_pool: str = "avg"):
        assert global_pool in {"avg", ""}
        super().__init__(dropout=dropout)
        self.classifier = nn.Linear(256 * 6 * 6, latent_dim)
        self.global_pool = global_pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_pool == "avg":
            return super().forward(x)
        return self.features(x)