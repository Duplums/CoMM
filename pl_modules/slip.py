# Modified from github.com/facebookresearch/SLIP
from torch import nn
from typing import Dict
from pl_modules.clip import CLIP
from losses.slip_loss import SLIPLoss
from losses.infonce import InfoNCE


class SLIP(CLIP):
    def __init__(self, visual_projection: nn.Module, loss_kwargs: Dict, **kwargs):
        super().__init__(**kwargs)
        self.head = visual_projection
        ssl_loss = InfoNCE(loss_kwargs["temperature"])
        ssl_scale = loss_kwargs["ssl_scale"]
        self.loss = SLIPLoss(ssl_loss, ssl_scale)

    def forward(self, image, text, aug1, aug2):
        aug1_embed = self.head(self.visual(aug1))
        aug2_embed = self.head(self.visual(aug2))

        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text)

        return {'image_embed': image_embed,
                'text_embed': text_embed,
                'logit_scale': self.logit_scale.exp(),
                'aug1_embed': aug1_embed,
                'aug2_embed': aug2_embed}

