"""
Approach A2: XLM-RoBERTa encoder (frozen) with linear projection probe.

Implements Siamese dual-tower architecture with:
- Pre-trained backbone: intfloat/multilingual-e5-base (XLM-RoBERTa-base).
- All backbone parameters are strictly frozen (requires_grad=False).
- Shared linear projection head: 768 -> 768 with bias (590,592 trainable parameters, ~0.59M).
- Identity initialization: W = Identity, b = 0 (exact zero-shot parity at epoch 0).
- Masked mean pooling over valid tokens.
- L2-normalization of final dense sentence vectors.
- Direct projection method for cached frozen embeddings.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.config.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class XLMRLinearProbeRetriever(nn.Module):
    """Dual-encoder retriever using frozen XLM-RoBERTa backbone and linear probe."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        embed_dim: int = 768,
        projection_dim: int = 768,
        max_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the linear probe retriever.

        Args:
            model_name: Pretrained Hugging Face model identifier.
            embed_dim: Backbone embedding dimension (default: 768).
            projection_dim: Output projection dimension (default: 768).
            max_length: Maximum token sequence length (default: 512).
            device: Computing device (CUDA or CPU).
        """
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.projection_dim = projection_dim
        self.max_length = max_length

        target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load pre-trained tokenizer and transformer backbone
        logger.info(f"Loading pre-trained backbone '{model_name}' for A2 linear probe...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)

        # 2. Freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # 3. Trainable linear projection head (shared across query and passage towers)
        self.proj = nn.Linear(embed_dim, projection_dim, bias=True)

        # 4. Identity initialization: W = Identity, b = 0
        nn.init.eye_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # Parameter count verification
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        logger.info(
            f"Initialized XLMRLinearProbeRetriever: "
            f"{trainable_params:,} trainable params (~0.59M), {frozen_params:,} frozen params"
        )
        assert trainable_params == (embed_dim * projection_dim + projection_dim), (
            f"Expected {embed_dim * projection_dim + projection_dim} trainable parameters, got {trainable_params}"
        )

        self.to(target_device)

    def project(self, pooled_embeddings: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize pooled representations.

        Allows near-instantaneous training on pre-cached frozen backbone embeddings.

        Args:
            pooled_embeddings: Tensor of shape (B, embed_dim) from frozen backbone.

        Returns:
            Tensor of shape (B, projection_dim), unit-normalized.
        """
        projected = self.proj(pooled_embeddings)
        return F.normalize(projected, p=2, dim=-1)

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply masked mean pooling over valid non-padded tokens."""
        masked_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        sum_hidden = masked_hidden.sum(dim=1)
        sum_mask = attention_mask.sum(dim=1)[..., None].clamp(min=1e-9)
        return sum_hidden / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode token IDs through frozen backbone and linear projection head.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len).
            attention_mask: Tensor of shape (batch_size, seq_len).

        Returns:
            Tensor of shape (batch_size, projection_dim), unit-normalized.
        """
        with torch.no_grad():
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self._pool(outputs.last_hidden_state, attention_mask)
        return self.project(pooled)

    @torch.no_grad()
    def extract_backbone_embeddings(
        self,
        texts: Sequence[str],
        prefix: str = "",
        batch_size: int = 32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Extract unprojected pooled representations from the frozen backbone.

        Args:
            texts: List of text strings.
            prefix: Prefix to prepend ('query: ' or 'passage: ').
            batch_size: Batch size for tokenization and model forward.
            device: Target torch device.

        Returns:
            Tensor of shape (len(texts), embed_dim) on CPU.
        """
        target_device = device or next(self.parameters()).device
        prefixed = [f"{prefix}{t}" for t in texts]
        all_pooled: list[torch.Tensor] = []

        for i in range(0, len(prefixed), batch_size):
            batch_texts = prefixed[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(target_device)

            outputs = self.backbone(**encoded)
            pooled = self._pool(outputs.last_hidden_state, encoded["attention_mask"])
            all_pooled.append(pooled.cpu())

        return torch.cat(all_pooled, dim=0)

    def encode_queries(
        self,
        queries: Sequence[str],
        device: Optional[torch.device] = None,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Tokenize, encode, and project queries with 'query: ' prefix."""
        target_device = device or next(self.parameters()).device
        pooled = self.extract_backbone_embeddings(
            queries, prefix="query: ", batch_size=batch_size, device=target_device
        ).to(target_device)
        return self.project(pooled)

    def encode_passages(
        self,
        passages: Sequence[str],
        device: Optional[torch.device] = None,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Tokenize, encode, and project passages with 'passage: ' prefix."""
        target_device = device or next(self.parameters()).device
        pooled = self.extract_backbone_embeddings(
            passages, prefix="passage: ", batch_size=batch_size, device=target_device
        ).to(target_device)
        return self.project(pooled)
