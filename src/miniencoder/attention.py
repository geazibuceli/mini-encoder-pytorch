"""Manual and optimized multi-head self-attention."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class _MultiHeadBase(nn.Module):
    def __init__(self, dimension, heads, dropout=0.0):
        super().__init__()
        if type(dimension) is not int or type(heads) is not int or dimension <= 0 or heads <= 0:
            raise ValueError("dimension and heads must be positive integers.")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if dimension % heads:
            raise ValueError("Embedding dimension must be divisible by heads.")
        self.dimension, self.heads, self.head_dimension, self.dropout = (
            dimension,
            heads,
            dimension // heads,
            dropout,
        )
        self.q_proj, self.k_proj, self.v_proj = (nn.Linear(dimension, dimension) for _ in range(3))
        self.out_proj = nn.Linear(dimension, dimension)

    def split(self, values):
        return values.view(
            values.size(0), values.size(1), self.heads, self.head_dimension
        ).transpose(1, 2)

    def project(self, inputs):
        return (
            self.split(self.q_proj(inputs)),
            self.split(self.k_proj(inputs)),
            self.split(self.v_proj(inputs)),
        )

    def merge(self, values):
        return (
            values.transpose(1, 2).contiguous().view(values.size(0), values.size(2), self.dimension)
        )


class ManualMultiHeadSelfAttention(_MultiHeadBase):
    def forward(self, inputs, attention_mask=None, return_attention=False):
        query, key, value = self.project(inputs)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dimension)
        if attention_mask is not None:
            scores = scores.masked_fill(
                ~attention_mask[:, None, None, :].bool(), torch.finfo(scores.dtype).min
            )
        weights = torch.softmax(scores, dim=-1)
        if attention_mask is not None:
            weights = weights.masked_fill(~attention_mask[:, None, None, :].bool(), 0.0)
        weights = F.dropout(weights, self.dropout, self.training)
        output = self.out_proj(self.merge(torch.matmul(weights, value)))
        return (output, weights) if return_attention else output


class SDPAMultiHeadSelfAttention(_MultiHeadBase):
    def forward(self, inputs, attention_mask=None, return_attention=False):
        if return_attention:
            raise ValueError("SDPA does not expose attention weights; use manual attention.")
        query, key, value = self.project(inputs)
        mask = attention_mask[:, None, None, :].bool() if attention_mask is not None else None
        output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0
        )
        return self.out_proj(self.merge(output))
