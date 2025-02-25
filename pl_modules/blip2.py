import torch
import torch.nn.functional as F
from typing import List
from lavis.models import load_model
from lavis.models.blip2_models.blip2_qformer import BlipOutputFeatures
from pl_modules.base import BaseModel


class Blip2(BaseModel):
    """Blip2 model for vision-language representation learning.
     Current implementation does not support training.
     """

    def __init__(self, **kwargs):
        super().__init__(optim_kwargs=dict())
        self.model = load_model(name="blip2_feature_extractor", model_type="pretrain", is_eval=True)

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def forward(self, image: torch.Tensor, text: List[str]):
        samples = dict({"image": image, "text_input": text})
        features = self._extract_features(samples, mode="multimodal")[f"multimodal_embeds"][:, 0, :]
        return features

    def extract_features(self, loader: torch.utils.data.DataLoader, mode="multimodal"):
        """
        Extract BLIP2 features (from vision, language, multimodal or stacking)
        Args:
            loader: Dataset loader to serve ``(X, y)`` tuples.
            mode: str, in {'image', 'text', 'multimodal', 'stacking'}
                Which modality to encode:
                If 'multimodal', returns a multimodal representation.
                If 'stacking', returns a stack of image + text representations
        Returns: Pair (X,y) corresponding to extracted features and corresponding labels
        """
        X, y = [], []

        for X_, y_ in loader:
            images, text = None, None
            if isinstance(X_, list):  # first modality == image, second modality == text (convention)
                if mode in {"image", "multimodal", "stacking"}:
                    images = X_[0].to(self.device)
                if mode in {"text", "multimodal", "stacking"}:
                    text = X_[1]
                    if isinstance(text, torch.Tensor):
                        text = text.to(self.device)
            else:
                assert mode == "image" or mode == "text", "`mode` must be either 'image' or 'text'"
                if mode == "image":
                    images = X_.to(self.device)
                if mode == "text":
                    text = X_.to(self.device)
            samples = dict({"image": images, "text_input": text})
            # compute output
            if mode == "stacking":
                features_image = self._extract_features(samples, mode="image")[f"image_embeds"][:, 0, :]
                features_text = self._extract_features(samples, mode="text")[f"text_embeds"][:, 0, :]
                features = torch.cat((features_image, features_text), dim=-1)
            else:
                features = self._extract_features(samples, mode=mode)[f"{mode}_embeds"][:, 0, :]
            X.extend(features.detach().cpu())
            y.extend(y_.detach().cpu())
        torch.cuda.empty_cache()
        return torch.stack(X, dim=0).to(self.device), torch.stack(y, dim=0).to(self.device)

    @torch.no_grad()
    def _extract_features(self, samples, mode="multimodal"):
        """ This function is a re-implementation of:
                `lavis.models.blip2_models.blip2_qformer.Blip2Qformer.extract_features()`

        It allows to handle arbitrarily long text sequences. Current official implementation is limited to 512.
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.model.maybe_autocast():
                image_embeds_frozen = self.model.ln_vision(self.model.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.model.device)
            query_tokens = self.model.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.model.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.model.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.model.tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(
                self.model.device
            )

            text_output = self.model.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.model.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.model.maybe_autocast():
                image_embeds_frozen = self.model.ln_vision(self.model.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.model.device)
            query_tokens = self.model.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.model.device
            )

            text = self.model.tokenizer(caption, return_tensors="pt", padding=True, truncation=True).to(
                self.model.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.model.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )



