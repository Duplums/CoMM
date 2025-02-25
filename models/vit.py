import timm
import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    """Pre-trained vision transformers supported by timm library. """

    def __init__(self, model_name: str,
                 pretrained: bool = True,
                 freeze: bool = False,
                 output_value: str = "embedding"):
        super().__init__()
        assert output_value in {'embedding', "token_embeddings"}

        if output_value == "token_embeddings":
            self.model = timm.create_model(model_name, global_pool="", pretrained=pretrained)
            self.model.head = nn.Identity() # get token embeddings
        else:
            self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model_name = model_name
        self.freeze = freeze
        self.pretrained = pretrained

        if freeze: # no grad computed
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor):
        if self.freeze:
            with torch.no_grad():
                return self.model(x)
        return self.model(x)