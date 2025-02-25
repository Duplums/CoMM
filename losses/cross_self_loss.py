from torch import nn
from losses.infonce import InfoNCE


class CrossSelfLoss(nn.Module):

    def __init__(self, temperature: float = 0.1, ssl_scale: float = 1.0):
        super().__init__()
        self.ssl_scale = ssl_scale
        self.infonce = InfoNCE(temperature=temperature)

    def forward(self, inputs):
        cross = self.infonce(dict(aug1_embed=inputs["mod1_embed"],
                               aug2_embed=inputs["mod2_embed"]))
        ssl1 = self.infonce(dict(aug1_embed=inputs["mod1_aug1_embed"],
                              aug2_embed=inputs["mod1_aug2_embed"]))
        ssl2 = self.infonce(dict(aug1_embed=inputs["mod2_aug1_embed"],
                              aug2_embed=inputs["mod2_aug2_embed"]))
        loss = cross['loss'] + self.ssl_scale * 0.5 * (ssl1['loss'] + ssl2['loss'])
        out_dict = dict(loss=loss)
        for o, name in zip((cross, ssl1, ssl2), ("cross", "mod1", "mod2")):
            out_dict.update({k + f"_{name}": v for k, v in o.items()})
        return out_dict