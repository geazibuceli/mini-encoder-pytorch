from torch import nn


class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, vocabulary_size, embedding_dimension=64, hidden_dimension=64, padding_id=0, dropout=0.1):
        super().__init__()
        self.model_config = dict(type="baseline", embedding_dimension=embedding_dimension,
                                 hidden_dimension=hidden_dimension, dropout=dropout)
        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension, padding_idx=padding_id)
        self.norm = nn.LayerNorm(embedding_dimension)
        self.classifier = nn.Sequential(nn.Linear(embedding_dimension, hidden_dimension), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(hidden_dimension, 2))

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(embeddings.dtype)
        pooled = (embeddings * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.classifier(self.norm(pooled))
