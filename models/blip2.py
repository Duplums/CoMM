import torch
import torch.nn as nn
from typing import List
from lavis.models import load_model
from utils import TextMasking


class Blip2VisionTransformer(nn.Module):
    """ BLIP2 pretrained vision encoder"""

    def __init__(self, output_value: str = 'token_embeddings'):
        super().__init__()
        assert output_value in {'embedding', 'token_embeddings'}
        self.output_value = output_value
        self.model = load_model(name="blip2_feature_extractor", model_type="pretrain")
        # Freeze all weights (no fine-tuning)
        for params in self.model.parameters():
            params.requires_grad = False

    def forward(self, x: torch.Tensor):
        features = self.model.extract_features(dict(image=x), mode="image")
        if self.output_value == "embedding":
            return features["image_embeds_proj"][:, 0, :] # shape (B, 256)
        else:
            return features["image_embeds"] # shape (B, L, 768)


class Blip2LanguageTransformer(nn.Module):
    """ BLIP-2 pretrained text encoder that implements random masking of input text.
    """

    def __init__(self,
                 output_value: str = 'token_embeddings',
                 mask_prob: float = 0.0
                 ):
        """
        :param output_value:  Default "token_embeddings", to get wordpiece token embeddings with shape (N, L, 768)
            where N == batch size, L == # tokens
            Can be set to "embedding" to get sentence embeddings with shape (N, 768).
        :param mask_prob: probability of randomly masking input tokens with mask tokens.
        """

        super().__init__()
        assert output_value in {"token_embeddings", "embedding"}

        self.model = load_model(name="blip2_feature_extractor", model_type="pretrain")
        # Feeze all weights (no fine-tuning)
        for params in self.model.parameters():
            params.requires_grad = False

        mask_ignore_token_ids = [self.model.tokenizer.pad_token_id,
                                 self.model.tokenizer.cls_token_id,
                                 self.model.tokenizer.sep_token_id]
        mask_token_id = self.model.tokenizer.mask_token_id
        self.mask = TextMasking(mask_prob, mask_token_id, mask_ignore_token_ids)
        self.output_value = output_value

    def forward(self, x: List[str]):
        text = self.model.tokenizer(x, return_tensors="pt",
                                    padding=True, truncation=True).to(self.model.device)
        text["input_ids"] = self.mask(text["input_ids"]).to(self.model.device)

        # return text features
        with torch.no_grad():
            text_output = self.model.Qformer.bert(
                    text.input_ids,
                    attention_mask=text.attention_mask,
                    return_dict=True,
                )
        text_embeds = text_output.last_hidden_state
        if self.output_value == "embedding":
            return text_embeds[:, 0, :]
        return text_embeds
