import math
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch
from einops import rearrange
from torch import nn as nn
from torch.nn import functional as F
from torchvision.models.vision_transformer import EncoderBlock
from torchvision.transforms.functional import to_pil_image


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        pos_embedding=None,
    ):
        super().__init__()
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default

        if pos_embedding is None:
            self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        else:
            self.pos_embedding = nn.Parameter(pos_embedding, requires_grad=False)
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(
            input.dim() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )

        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class AxialVisionTransformer(nn.Module):

    def __init__(
        self,
        conv_config: list[tuple],
        vit_config: dict,
        img_channels=3,
        img_size=448,
        hidden_dim: Optional[int] = None,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):

        super().__init__()

        self.img_size = img_size
        self.patch_size = vit_config["patch_size"]
        self.hidden_dim = hidden_dim
        self.attention_dropout = attention_dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        conv_proj_layers: OrderedDict[str, nn.Module] = OrderedDict()

        conv_config["in"][0] = img_channels
        for idx, (in_, out_, kernel_size, stride, padding) in enumerate(
            zip(
                conv_config["in"],
                conv_config["out"],
                conv_config["kernel_size"],
                conv_config["stride"],
                conv_config["padding"],
            )
        ):
            conv_proj_layers[f"projection_{idx}"] = nn.Conv2d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding)

        self.conv_proj = nn.Sequential(conv_proj_layers)

        if self.hidden_dim is None:
            with torch.no_grad():
                img_channels = img_channels
                x_ = self.conv_proj(torch.empty(img_channels, self.img_size, self.img_size))
                token_size = self.img_size // self.patch_size

                x_ = rearrange(x_, "c d (l k) -> l (c d k)", l=token_size)
                self.hidden_dim = x_.size(1)

        print(f"model hidden dim: {self.hidden_dim}")

        seq_length = self.img_size // self.patch_size

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        seq_length += 1

        pos_embedding = None
        if not vit_config["learn_pos_embedding"]:
            pe = torch.zeros(seq_length, self.hidden_dim)
            position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.hidden_dim, 2).float() * (-math.log(10000.0) / self.hidden_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pos_embedding = pe.unsqueeze(0)

        self.encoder = Encoder(
            seq_length,
            vit_config["num_layers"],
            vit_config["num_heads"],
            self.hidden_dim,
            vit_config["mlp_dim"],
            vit_config["dropout"],
            attention_dropout,
            norm_layer,
            pos_embedding=pos_embedding,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(self.hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(self.hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        for conv_proj in conv_proj_layers.values():
            fan_in = conv_proj.in_channels * conv_proj.kernel_size[0] * conv_proj.kernel_size[1]
            nn.init.trunc_normal_(conv_proj.weight, std=math.sqrt(1 / fan_in))
            if conv_proj.bias is not None:
                nn.init.zeros_(conv_proj.bias)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(
            h == self.img_size,
            f"Wrong image height! Expected {self.img_size} but got {h}!",
        )
        torch._assert(
            w == self.img_size,
            f"Wrong image width! Expected {self.img_size} but got {w}!",
        )
        token_size = h // p

        x = self.conv_proj(x)

        x = rearrange(x, "b c d (l k) -> b l (c d k)", l=token_size)

        # The self attention layer expects inputs in the format (B, S, E)
        # where S is the source sequence length, B is the batch size, E is the
        # embedding dimension

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x
