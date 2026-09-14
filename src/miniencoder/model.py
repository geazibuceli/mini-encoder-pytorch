from torch import nn

from .attention import ManualMultiHeadSelfAttention, SDPAMultiHeadSelfAttention
from .positional_encoding import SinusoidalPositionalEncoding


class EncoderBlock(nn.Module):
    def __init__(
        self, dimension, heads, feedforward_dimension, dropout=0.1, attention_backend="manual"
    ):
        super().__init__()
        attention = (
            ManualMultiHeadSelfAttention
            if attention_backend == "manual"
            else SDPAMultiHeadSelfAttention
        )
        if attention_backend not in {"manual", "sdpa"}:
            raise ValueError("Unsupported attention backend.")
        self.norm1, self.norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
        self.attention = attention(dimension, heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(dimension, feedforward_dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dimension, dimension),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, attention_mask=None, return_attention=False):
        attended = self.attention(self.norm1(inputs), attention_mask, return_attention)
        weights = attended[1] if return_attention else None
        attended = attended[0] if return_attention else attended
        outputs = inputs + self.dropout(attended)
        outputs = outputs + self.dropout(self.feed_forward(self.norm2(outputs)))
        return (outputs, weights) if return_attention else outputs


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_dimension=64,
        heads=4,
        layers=2,
        feedforward_dimension=128,
        max_length=128,
        pooling="cls",
        positional_encoding=True,
        attention_backend="manual",
        padding_id=0,
        dropout=0.1,
    ):
        super().__init__()
        if layers <= 0 or max_length <= 0:
            raise ValueError("layers and max_length must be positive.")
        if pooling not in {"cls", "mean"}:
            raise ValueError("Pooling must be 'cls' or 'mean'.")
        self.model_config = dict(
            type="transformer",
            embedding_dimension=embedding_dimension,
            heads=heads,
            layers=layers,
            feedforward_dimension=feedforward_dimension,
            max_length=max_length,
            pooling=pooling,
            positional_encoding=positional_encoding,
            attention_backend=attention_backend,
            dropout=dropout,
        )
        self.pooling, self.attention_backend = pooling, attention_backend
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension, padding_idx=padding_id)
        self.position = (
            SinusoidalPositionalEncoding(embedding_dimension, max_length)
            if positional_encoding
            else None
        )
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                EncoderBlock(
                    embedding_dimension, heads, feedforward_dimension, dropout, attention_backend
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(embedding_dimension)
        self.classifier = nn.Linear(embedding_dimension, 2)

    def forward(self, input_ids, attention_mask, return_attention=False):
        outputs = self.embedding_dropout(
            self.position(self.embedding(input_ids)) if self.position else self.embedding(input_ids)
        )
        attentions = []
        for block in self.blocks:
            result = block(outputs, attention_mask, return_attention)
            outputs, weights = result if return_attention else (result, None)
            if return_attention:
                attentions.append(weights)
        outputs = self.norm(outputs)
        if self.pooling == "cls":
            pooled = outputs[:, 0]
        else:
            mask = attention_mask.unsqueeze(-1).to(outputs.dtype)
            pooled = (outputs * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        logits = self.classifier(pooled)
        return (logits, attentions) if return_attention else logits
