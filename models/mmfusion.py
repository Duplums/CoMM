import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Union
from einops import repeat
from collections import OrderedDict
# Local import
from models.mlp import MLP

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualCrossAttentionBlock(nn.Module):
    """Cross-attention module between 2 inputs. """
    def __init__(self, d_model: int, n_heads: int,
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_heads, add_bias_kv=add_bias_kv,
                                          dropout=dropout,  batch_first=batch_first)
        self.ln_1x = nn.LayerNorm(d_model)
        self.ln_1y = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor = None,
                  attn_mask: torch.Tensor = None):
        return self.attn(x, y, y, need_weights=False, key_padding_mask=key_padding_mask, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, y: torch.Tensor, key_padding_mask: torch.Tensor = None,
                attn_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1x(x), self.ln_1y(y), key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, d_model: int, n_head: int,
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, add_bias_kv=add_bias_kv,
                                          dropout=dropout,  batch_first=batch_first)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        return self.attn(x.clone(), x, x, need_weights=False, key_padding_mask=key_padding_mask)[0]

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class FusionTransformer(nn.Module):
    """Fusion of features from multiple modalities using attention.
    in_shape: (N, L1, E), (N, L2, E), out_shape: (N, E)
    We use either:
        - "concat": concatenation over tokens + self-attention module
        - "x-attn": cross-attention between two sets of tokens + concatenation over tokens
    An attention mask can be applied eventually for each modality with shape (N, Li) for modality i.
    """
    def __init__(self, width: int,
                 n_heads: int,
                 n_layers: int,
                 fusion: str = "concat",
                 pool: str = "cls",
                 add_bias_kv: bool = False,
                 dropout: float = 0.,
                 batch_first: bool = True):
        """
        :param width: embedding size
        :param n_heads: number of heads in multi-head attention blocks
        :param n_layers: number of attention blocks
        :param fusion: "concat" or "x-attn"
        :param pool: "cls" or "pool"
        :param add_bias_kv: If specified, adds bias to the key and value sequences at dim=0.
        :param dropout: Dropout probability on `attn_output_weights`
        :param batch_first: input tensor is either (batch, tokens, features) if `True` or (tokens, batch, features)
        """
        super().__init__()

        self.fusion = fusion
        self.width = width
        self.layers = n_layers
        self.norm = nn.LayerNorm(width)
        self.token_dim = 1 if batch_first else 0
        self.pool = pool
        self.cls_token = nn.Parameter(torch.randn(1, 1, width)) if self.pool == "cls" else None
        if fusion == "concat":
            self.resblocks = nn.Sequential(*[
                ResidualAttentionBlock(width, n_heads, add_bias_kv=add_bias_kv,
                                       dropout=dropout, batch_first=batch_first)
                for _ in range(n_layers)])
        elif fusion == "x-attn":
            self.resblocks = [
                nn.Sequential(*[
                    ResidualCrossAttentionBlock(width, n_heads, add_bias_kv=add_bias_kv,
                                                dropout=dropout, batch_first=batch_first)
                    for _ in range(n_layers)])
                for _ in range(2)]
        else:
            raise ValueError("Unknown fusion %s" % fusion)
        self.initialize()

    def initialize(self):
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, x: List[torch.Tensor], key_padding_mask: List[torch.Tensor] = None):
        """
        :param x: input tensors
        :param key_padding_mask: torch mask of type bool. `True` indicates unattended tokens.
        :return:
        """
        # Concatenate over tokens + self-attention
        if self.fusion == "concat":
            x = torch.cat(x, dim=self.token_dim)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(key_padding_mask, dim=self.token_dim)
            if self.pool == "cls": # append cls token at the beginning
                cls_token = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
                x = torch.cat((cls_token, x), dim=self.token_dim)
                if key_padding_mask is not None:
                    key_padding_mask = torch.cat(
                        (torch.zeros_like(cls_token[:, :, 0]), key_padding_mask), dim=self.token_dim)

            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.masked_fill(key_padding_mask.bool(), float("-inf")).float()

            for layer in self.resblocks:
                x = layer(x, key_padding_mask=key_padding_mask)

            x = self.norm(x)

            if self.pool == "cls":
                x = x[:, 0] if self.token_dim == 1 else x[0]
            else:
                x = x.mean(dim=self.token_dim)
            return x
        # Cross-attention + concatenate over tokens
        elif self.fusion == "x-attn":
            if self.pool == "cls":
                raise ValueError("Only `mean` pool is implemented for cross-attention.")
            if len(x) != 2:
                raise ValueError("Only 2 modalities are currently accepted for cross-attention")
            if key_padding_mask is not None:
                raise NotImplementedError()
            x1, x2 = x
            x = torch.cat([self.resblocks[0](x1, x2, key_padding_mask),
                           self.resblocks[1](x2, x1, key_padding_mask)], dim=self.token_dim)
            x = self.norm(x).mean(dim=self.token_dim)
            return x


class MMFusion(nn.Module):
    def __init__(self,
                 encoders: List[nn.Module],
                 input_adapters: List[nn.Module],
                 embed_dim: int = 512,
                 fusion: str = "concat",
                 pool: str = "cls",
                 n_heads: int = 8,
                 n_layers: int = 1,
                 add_bias_kv: bool = False,
                 dropout: float = 0.):
        """ Multi-Modal (MM) fusion model using `FusionTransformer` in the latent space.
        It can handle an arbitrary number of input modalities.
        Each modality is encoded through either a:
            - Transformer (e.g. for text or audio) -> no adapters
            - CNN (e.g. for images) -> `PatchedInputAdapter` for tokenization
            - MLP (e.g. tabular data) -> `FeaturesInputAdapter` for tokenization
        Once each modality is encoded and tokenized, it then goes to `FusionTransformer` to output
        the final embedding.

        :param encoders: List of Torch encoders (CNN, Transformer, MLP, etc.) for each modality
        :param input_adapters: List of Torch adapters for each modality (can be None if not required)
        :param embed_dim: Embedding size
        :param fusion: "concat" or "x-attn". For "x-attn", only "mean" pool is accepted.
        :param pool: "cls" or "mean", pooling strategy for the tokens
        :param n_heads: Number of heads in multi-heads attention blocks
        :param n_layers: Number of attention layers in latent fusion
        :param add_bias_kv: If `True`, add bias term in key/values mapping
        :param dropout: attention matrix dropout rate
        """
        super().__init__()
        assert len(encoders) == len(input_adapters), "Each encoder must have an adapter."
        assert pool in {'cls', 'mean'}, "pool type must be either cls (cls token) or mean (mean pooling)"
        self.input_adapters = nn.ModuleList(input_adapters)
        self.encoders = nn.ModuleList(encoders)
        self.pool = pool
        self.num_modalities = len(self.encoders)
        self.fusion_transformer = FusionTransformer(embed_dim, n_heads, n_layers,
                                                    fusion, pool, add_bias_kv, dropout,
                                                    batch_first=True)

    def forward(self, x: List[torch.Tensor],
                mask_modalities: Optional[Union[List[bool], List[List[bool]]]] = None):
        """
        :param x: List of tensors
        :param mask_modalities: Mask indicating which modalities are given.
            By default, `x` should have all modalities.
            If a list of lists is given, assume `x` has all modalities and computes
            a list of output by masking out modalitites according to `mask_modalities`.
        :return: a latent vector z or list of vector if `mask_modalities` is a list of list.
        """
        list_mask_mod = None
        if mask_modalities is None:
            mask_modalities = self.num_modalities * [True]
        elif isinstance(mask_modalities, list) and len(mask_modalities)>0 and isinstance(mask_modalities[0], list):
            list_mask_mod = mask_modalities
            mask_modalities = self.num_modalities * [True]

        assert len(mask_modalities) == self.num_modalities, (
            f"Mask size does not match `num_modalities`: {len(mask_modalities)} != {self.num_modalities}")

        num_modalities = sum(mask_modalities)
        assert len(x) == num_modalities, (
                f"Incorrect number of inputs: {len(x)} != {num_modalities}")

        encoders = [enc for (enc, m) in zip(self.encoders, mask_modalities) if m]
        input_adapters = [adapter for (adapter, m) in zip(self.input_adapters, mask_modalities) if m]
        attn_mask = []

        # 1. Encode input modalities
        z = []
        for (enc, xi) in zip(encoders, x):
            embedding = enc(xi)
            attn_mask_ = None
            if isinstance(embedding, dict):  # attention mask must be considered
                attn_mask_ = embedding["attention_mask"]
                embedding = embedding["token_embeddings"]
            z.append(embedding)
            attn_mask.append(attn_mask_)

        # 2. Tokenize each latent features
        latent_tokens = [adapter(zi) if adapter is not None else zi
                         for (adapter, zi) in zip(input_adapters, z)]
        attn_mask = [attn_mask_ if attn_mask_ is not None else torch.zeros_like(zi[:,:,0]).bool()
                     for (attn_mask_, zi) in zip(attn_mask, latent_tokens)]
        if list_mask_mod is None:
            # 3. FusionTransformer forward pass
            z = self.fusion_transformer(latent_tokens, key_padding_mask=attn_mask)
        else:
            # 3.bis Drop modalities according to `mask_modalities`
            z = []
            for mask_mod in list_mask_mod:
                latent_tokens_ = [z for (z, m) in zip(latent_tokens, mask_mod) if m]
                attn_mask_ = [attn for (attn, m) in zip(attn_mask, mask_mod) if m]
                # 3. FusionTransformer forward pass
                z.append(self.fusion_transformer(latent_tokens_))
        return z

    def encode_single_mod(self, x: torch.Tensor, mod: int):
        assert 0 <= mod < self.num_modalities, "Wrong input modality"
        return self.encoders[mod](x)


class LinearFusion(nn.Module):
    def __init__(self,
                 encoders: List[nn.Module],
                 mod_dims: List[int],
                 embed_dim: int = 512,
                 **kwargs):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.mod_dims = mod_dims
        assert len(self.mod_dims) == len(self.encoders)
        self.embed_dim = embed_dim
        self.num_modalities = len(self.encoders)
        # projector for each modality to common space
        self.projectors = nn.ModuleList([nn.Linear(mod_dim, embed_dim) for mod_dim in mod_dims])
        # projector for all modalities to common space
        self.head_projector = nn.Linear(int(sum(mod_dims)), embed_dim)

    def forward(self, x: List[torch.Tensor],
                mask_modalities: Optional[Union[List[bool], List[List[bool]]]] = None):

        list_mask_mod = None
        if mask_modalities is None:
            mask_modalities = self.num_modalities * [True]
        elif isinstance(mask_modalities, list) and len(mask_modalities)>0 and isinstance(mask_modalities[0], list):
            list_mask_mod = mask_modalities
            mask_modalities = self.num_modalities * [True]
        assert len(mask_modalities) == self.num_modalities, (
            f"Mask size does not match `num_modalities`: {len(mask_modalities)} != {self.num_modalities}")
        num_modalities = sum(mask_modalities)
        assert len(x) == num_modalities, (
                f"Incorrect number of inputs: {len(x)} != {num_modalities}")

        encoders = [enc for (enc, m) in zip(self.encoders, mask_modalities) if m]
        Z = [enc(xi) for enc, xi in zip(encoders, x)]
        if list_mask_mod is not None:
            Z_ = []
            for mask_mod in list_mask_mod:
                Z_.append(self.get_common_embedding(Z, mask_mod))
            return Z_
        return self.get_common_embedding(Z, mask_modalities)

    def get_common_embedding(self, z: List[torch.Tensor], mask_modalities: List[bool]):
        if np.sum(mask_modalities) == 1:
            idx = int(np.nonzero(mask_modalities)[0][0])
            return self.projectors[idx](z[idx])
        elif np.sum(mask_modalities) == 2:
            return self.head_projector(torch.cat(z, dim=-1))
        raise NotImplementedError()


class MLPFusion(nn.Module):
    def __init__(self,
                 encoders: List[nn.Module],
                 mod_dims: List[int],
                 embed_dim: int = 512,
                 **kwargs):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.mod_dims = mod_dims
        assert len(self.mod_dims) == len(self.encoders)
        self.embed_dim = embed_dim
        self.num_modalities = len(self.encoders)
        # non-linear projector for each modality to common space
        self.projectors = nn.ModuleList([MLP(mod_dim, embed_dim, embed_dim) for mod_dim in mod_dims])
        # non-linear projector for all modalities to common space
        self.head_projector = MLP(int(sum(mod_dims)), embed_dim, embed_dim)

    def forward(self, x: List[torch.Tensor],
                mask_modalities: Optional[Union[List[bool], List[List[bool]]]] = None):

        list_mask_mod = None
        if mask_modalities is None:
            mask_modalities = self.num_modalities * [True]
        elif isinstance(mask_modalities, list) and len(mask_modalities)>0 and isinstance(mask_modalities[0], list):
            list_mask_mod = mask_modalities
            mask_modalities = self.num_modalities * [True]
        assert len(mask_modalities) == self.num_modalities, (
            f"Mask size does not match `num_modalities`: {len(mask_modalities)} != {self.num_modalities}")
        num_modalities = sum(mask_modalities)
        assert len(x) == num_modalities, (
                f"Incorrect number of inputs: {len(x)} != {num_modalities}")

        encoders = [enc for (enc, m) in zip(self.encoders, mask_modalities) if m]
        Z = [enc(xi) for enc, xi in zip(encoders, x)]
        if list_mask_mod is not None:
            Z_ = []
            for mask_mod in list_mask_mod:
                Z_.append(self.get_common_embedding(Z, mask_mod))
            return Z_
        return self.get_common_embedding(Z, mask_modalities)

    def get_common_embedding(self, z: List[torch.Tensor], mask_modalities: List[bool]):
        if np.sum(mask_modalities) == 1:
            idx = int(np.nonzero(mask_modalities)[0][0])
            return self.projectors[idx](z[idx])
        elif np.sum(mask_modalities) == 2:
            return self.head_projector(torch.cat(z, dim=-1))
        raise NotImplementedError()


if __name__ == "__main__":
    width = 10
    batch = 3
    fusion = FusionTransformer(width, 2, 2)
    x = [torch.randn((batch, 2, width)), torch.randn((batch, 3, width))]
    # preserve modality 1
    mask = [torch.ones((batch, 2)).bool(), torch.ones((batch, 3)).bool()]
    print(fusion(x, mask))
    print(fusion([x[1]]))



