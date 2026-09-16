"""
Approach A1: Bidirectional LSTM (BiLSTM) dual encoder for Khmer legal retrieval.

Implements Siamese dual-tower architecture with shared parameters:
- Word embedding layer (300-d, randomly initialized from scratch).
- 1-layer Bidirectional LSTM (hidden_size=256 per direction, output_dim=512).
- Dropout on embeddings and sequence outputs.
- Masked mean pooling over valid (non-padded) tokens.
- L2-normalization of final dense sentence representations.
"""

from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dl.models.vocab import KhmerVocab, PAD_ID


class BiLSTMEncoder(nn.Module):
    """Siamese BiLSTM sequence encoder for query and passage representations."""

    def __init__(
        self,
        vocab: KhmerVocab,
        embed_dim: int = 300,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_idx: int = PAD_ID,
    ) -> None:
        """Initialize the BiLSTM dual encoder.

        Args:
            vocab: KhmerVocab instance for text tokenization and integer encoding.
            embed_dim: Word embedding dimension (default: 300).
            hidden_dim: Hidden dimension per direction of BiLSTM (default: 256).
            dropout: Dropout probability (default: 0.1).
            max_seq_len: Maximum sequence length for input truncation (default: 256).
            pad_idx: Token ID corresponding to padding (default: 0).
        """
        super().__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = hidden_dim * 2  # Bidirectional doubles the dimension (512)
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx

        # 1. Randomly initialized embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx,
        )
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        with torch.no_grad():
            self.embedding.weight[pad_idx].fill_(0.0)

        # 2. Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # 3. Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode input token IDs into a single L2-normalized vector per sequence.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len) with token indices.
            attention_mask: Tensor of shape (batch_size, seq_len) with 1 for valid, 0 for pad.

        Returns:
            Tensor of shape (batch_size, 512) with unit-normalized sequence embeddings.
        """
        # (B, L) -> (B, L, embed_dim)
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        # Pack sequence or run directly through LSTM
        # (B, L, embed_dim) -> (B, L, 2 * hidden_dim)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)

        # Masked mean pooling over non-padded tokens
        mask_expanded = attention_mask.unsqueeze(-1)  # (B, L, 1)
        sum_embeddings = torch.sum(lstm_out * mask_expanded, dim=1)  # (B, 512)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # (B, 1)
        pooled = sum_embeddings / sum_mask  # (B, 512)

        # L2-normalize
        normalized = F.normalize(pooled, p=2, dim=-1)
        return normalized

    def encode_queries(
        self,
        queries: Sequence[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Tokenize, encode, and project a list of query strings into vector space.

        Args:
            queries: List of query strings.
            device: Target torch device.

        Returns:
            Tensor of shape (len(queries), 512).
        """
        target_device = device or next(self.parameters()).device
        input_ids, attention_mask = self.vocab.batch_encode(
            queries,
            max_seq_len=self.max_seq_len,
            device=target_device,
        )
        return self.forward(input_ids, attention_mask)

    def encode_passages(
        self,
        passages: Sequence[str],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Tokenize, encode, and project a list of passage strings into vector space.

        Shares the identical tower and parameters as encode_queries.

        Args:
            passages: List of passage strings.
            device: Target torch device.

        Returns:
            Tensor of shape (len(passages), 512).
        """
        return self.encode_queries(passages, device=device)
