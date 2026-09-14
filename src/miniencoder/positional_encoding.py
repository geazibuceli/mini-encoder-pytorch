import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dimension: int, max_length: int):
        super().__init__()
        positions = torch.arange(max_length).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension))
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, inputs):
        if inputs.size(1) > self.encoding.size(1):
            raise ValueError("Sequence length exceeds positional-encoding capacity.")
        return inputs + self.encoding[:, :inputs.size(1)]
