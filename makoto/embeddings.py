from typing import Sequence

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch import Tensor as T
from torch import nn


class ExpertEmbeddings(nn.Module):
    def __init__(self, model_names: Sequence[str], embedding_dim: int):
        super().__init__()

        self.model_names = model_names
        self.embeddings = nn.Embedding(num_embeddings=len(model_names), embedding_dim=embedding_dim)

    def __getitem__(self, expert_name: str) -> T:
        """
        Return the trainable embedding for a given expert by name
        """

        model_idx = T(self.model_names.index(expert_name)).to(int)
        return self.embeddings(model_idx)

    def to(self, device):
        self.embeddings.to(device)

    def forward(self, experts: Sequence[str]) -> T:

        expert_indices = T([self.model_names.index(expert_name) for expert_name in experts]).to(int)
        if torch.cuda.is_available():
            expert_indices = expert_indices.to("cuda")
        embeddings = self.embeddings(expert_indices)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class SentenceEmbedder(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: Don't hardcode.
        self.transformer = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
        sentence_dim = self.transformer.get_sentence_embedding_dimension()
        self.ff1 = nn.Linear(sentence_dim, sentence_dim)
        self.act1 = nn.ReLU()

    def forward(self, sequences: Sequence[str]) -> T:

        seq_embeddings = T(self.transformer.encode(sequences))
        if torch.cuda.is_available():
            seq_embeddings = seq_embeddings.to("cuda")

        seq_embeddings = self.ff1(seq_embeddings)
        seq_embeddings = self.act1(seq_embeddings)
        seq_embeddings = F.normalize(seq_embeddings, p=2, dim=1)

        return seq_embeddings

    def to(self, device):
        self.transformer.to(device)
        self.ff1.to(device)
        self.act1.to(device)

    @property
    def embedding_dimension(self) -> int:
        return self.transformer.get_sentence_embedding_dimension()