import torch
import os
import pickle
import abc
import torch.nn as nn
from einops import rearrange
from torch.nn.init import xavier_uniform_
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
#Local import
from utils import TextMasking
from models.input_adapters import build_1d_sincos_posemb


class Transformer(nn.Module):
    """Extends nn.Transformer as defined in FactorCL for MultiBench data."""

    def __init__(self, n_features: int, dim: int,
                 max_seq_length: int = 50,
                 return_seq: bool = True,
                 positional_encoding: bool = True,
                 pad_value: Optional[float] = None):
        """Initialize Transformer object.

        Args:
            n_features: Number of features in the input.
            dim: Dimension which to embed upon / Hidden dimension size.
            max_seq_length: Maximum expected sequence length.
            return_seq: If True, returns a sequence of encoded features.
                Otherwise, returns an embedding.
            positional_encoding: Whether to add positional embedding to input tokens.
            pad_value: Padding values added to the end of each sequence (used to compute attn mask)
        """
        super().__init__()
        self.embed_dim = dim
        self.conv = nn.Conv1d(n_features, self.embed_dim,
                              kernel_size=1, padding=0, bias=False)
        self.return_seq = return_seq
        self.use_positional_embedding = positional_encoding
        self.pad_value = pad_value
        self.positional_embedding = nn.Parameter(
            build_1d_sincos_posemb(max_seq_length, self.embed_dim), requires_grad=False)
        layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=5, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=5)

    def forward(self, x):
        """Apply Transformer to Input.

        Args:
            x (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        # From features to tokens
        x = self.conv(rearrange(x, 'b l n -> b n l'))
        x = rearrange(x, 'b n l -> b l n')
        if self.use_positional_embedding:
            x = x + self.positional_embedding[:, :x.size(1)]
        # Encode tokens
        x = self.transformer(x)
        # Get the embedding
        if not self.return_seq:
            x = x[:, -1]
        return x


class LanguageEncoder(nn.Module):
    """Pre-trained text model that implements a caching system for fast embedding retrieval when no
    fine-tuning is required. It currently accepts all models defined by `sentence_transformers` library.
    """

    def __init__(self, model_name: str,
                 freeze: bool = True,
                 output_value: str = 'sentence_embedding',
                 mask_prob: float = 0.0,
                 normalize_embeddings: bool = True,
                 use_dataset_cache: bool = True,
                 cache_file: Optional[str] = None):
        """

        :param model_name: Text encoder name, see https://www.sbert.net/docs/pretrained_models.html
        :param freeze: whether the text encoder is fine-tuned or not
        :param output_value:  Default "sentence_embedding", to get sentence embeddings with shape (N, E)
            where N == batch size, E == embedding dimension
            Can be set to "token_embeddings" to get wordpiece token embeddings with shape (N, L, E)
            and attention mask with shape (N, L) where N == batch size, L == # tokens, E == embedding dimension.
        :param mask_prob: probability of randomly masking input tokens with mask tokens.
        :param normalize_embeddings: whether text embeddings are l2-normalized or not
            only if output_value == "sentence_embedding"
        :param use_dataset_cache: Cache the text embeddings computed in `forward` pass.
        :param cache_file: File name of cache to dump on disk.
        """

        super().__init__()
        assert output_value in {"token_embeddings", "sentence_embedding"}

        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.use_dataset_cache = use_dataset_cache
        mask_ignore_token_ids = [self.model.tokenizer.pad_token_id,
                                 self.model.tokenizer.cls_token_id,
                                 self.model.tokenizer.sep_token_id]
        mask_token_id = self.model.tokenizer.mask_token_id
        self.mask = TextMasking(mask_prob, mask_token_id, mask_ignore_token_ids)
        self.freeze = freeze
        self.output_value = output_value
        self.cache_file = cache_file or ""
        self._cache = dict()
        if self.use_dataset_cache:
            self._cache = self._load_cache()

        if freeze: # no grad computed
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x: List[str]):
        if self.use_dataset_cache:
            if self.model_name in self._cache:
                embed = []
                for txt in x:
                    if txt in self._cache[self.model_name]:
                        embed.append(self._cache[self.model_name][txt])
                if len(embed) == len(x):
                    x = torch.stack(embed, dim=0).cuda()
                    if self.normalize_embeddings:
                        x = torch.nn.functional.normalize(x, p=2, dim=1)
                    return x

        features = self.model.tokenize(x) # automatically truncate too large sentences
        if self.training: # only apply masking in training mode
            features["input_ids"] = self.mask(features["input_ids"])
        features = batch_to_device(features, self.model.device)
        if self.freeze:
            with torch.no_grad():
                out_features = self.model.forward(features)
                if self.output_value == "sentence_embedding":
                    embeddings = out_features["sentence_embedding"]
                    embeddings = embeddings.detach()
                elif self.output_value == "token_embeddings":
                    for name in out_features:
                        out_features[name] = out_features[name].detach()
                    assert "attention_mask" in features
                    # !! Important: 'True' should indicate NOT attended positions (torch convention in attn layers)
                    out_features["attention_mask"] = ~features["attention_mask"].bool()
                    return out_features
        else:
            out_features = self.model.forward(features)
            if self.output_value == "sentence_embedding":
                embeddings = out_features["sentence_embedding"]
            elif self.output_value == "token_embeddings":
                assert "attention_mask" in features
                # !! Important: 'True' should indicate NOT attended positions (torch convention in attn layers)
                out_features["attention_mask"] = ~features["attention_mask"].bool()
                return out_features

        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        if self.use_dataset_cache:
            if self.model_name not in self._cache:
                self._cache[self.model_name] = dict()
            for i, txt in enumerate(x):
                self._cache[self.model_name][txt] = embeddings[i].cpu().detach()
        return embeddings

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self._cache = pickle.load(f)
            except FileNotFoundError:
                pass
        return self._cache

    def dump_cache(self):
        def update(d, u):  # updated nested dict
            for k, v in u.items():
                if isinstance(v, abc.Mapping):
                    d[k] = update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    cache = pickle.load(f)
                assert isinstance(cache, dict)
                update(self._cache, cache)
            except FileNotFoundError:
                pass
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump(self._cache, f)
        except Exception as e:
            print("Impossible to dump cache: %s" % e)
            return False
        return True
