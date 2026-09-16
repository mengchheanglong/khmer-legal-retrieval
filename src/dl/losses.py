"""
Contrastive loss functions for dual-encoder retrieval training.

Implements InfoNCE (Information Noise-Contrastive Estimation) loss with
in-batch negatives and temperature scaling.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE loss with in-batch negatives for bi-encoder representations.

    Given a batch of query embeddings Q (B x D) and passage embeddings P (B x D),
    the cosine similarity matrix S (B x B) is scaled by temperature tau:
        S_{i,j} = (q_i . p_j) / tau

    The diagonal elements S_{i,i} represent positive pairs, while off-diagonal
    elements S_{i, j != i} represent in-batch negative pairs.

    Optionally computes symmetric loss:
        Loss = 0.5 * (Loss_{q -> p} + Loss_{p -> q})
    """

    def __init__(
        self,
        temperature: float = 0.05,
        symmetric: bool = True,
        normalize_embeddings: bool = True,
    ) -> None:
        """Initialize the InfoNCE loss module.

        Args:
            temperature: Softmax temperature parameter tau > 0 (default: 0.05).
            symmetric: Whether to average query->passage and passage->query losses (default: True).
            normalize_embeddings: Whether to L2-normalize embeddings before computing dot products.
        """
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"Temperature must be positive, got {temperature}")

        self.temperature = temperature
        self.symmetric = symmetric
        self.normalize_embeddings = normalize_embeddings

    def forward(
        self,
        query_embeddings: torch.Tensor,
        passage_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the InfoNCE loss for a batch of query and passage representations.

        Args:
            query_embeddings: Tensor of shape (batch_size, embed_dim).
            passage_embeddings: Tensor of shape (batch_size, embed_dim).

        Returns:
            Scalar tensor representing the mean batch InfoNCE loss.

        Raises:
            ValueError: If batch sizes or embedding dimensions do not match.
        """
        if query_embeddings.shape != passage_embeddings.shape:
            raise ValueError(
                f"Shape mismatch: queries {query_embeddings.shape} vs "
                f"passages {passage_embeddings.shape}"
            )

        batch_size = query_embeddings.size(0)
        if batch_size == 0:
            raise ValueError("Cannot compute InfoNCE loss on an empty batch")

        # L2-normalize embeddings if requested
        if self.normalize_embeddings:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
            passage_embeddings = F.normalize(passage_embeddings, p=2, dim=-1)

        # Compute cosine similarity matrix: (B, D) @ (D, B) -> (B, B)
        similarity_matrix = torch.matmul(query_embeddings, passage_embeddings.t())
        logits = similarity_matrix / self.temperature

        # Positive targets are on the main diagonal [0, 1, ..., B-1]
        targets = torch.arange(batch_size, device=query_embeddings.device, dtype=torch.long)

        # Query-to-passage loss
        loss_q2p = F.cross_entropy(logits, targets)

        if not self.symmetric:
            return loss_q2p

        # Passage-to-query loss (transposed logits)
        loss_p2q = F.cross_entropy(logits.t(), targets)

        return 0.5 * (loss_q2p + loss_p2q)
