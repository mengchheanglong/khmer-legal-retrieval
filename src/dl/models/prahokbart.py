"""
Approach A4: PrahokBART Dual Encoder for Khmer Legal Retrieval.

Implements Siamese dual-tower architecture utilizing the Khmer-native
PrahokBART encoder (nict-astrec-att/prahokbart_base):
- Pretrained compact Khmer-centric model (~35.8M encoder parameters).
- Native SentencePiece tokenizer with 32,004 Khmer vocabulary items.
- 512-dimensional output embeddings.
- Masked mean pooling over valid attention tokens.
- L2 unit normalization of final dense representations.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, MBartForConditionalGeneration

from src.config.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class PrahokBARTDualEncoder(nn.Module):
    """Dual-encoder retriever utilizing the Khmer-native PrahokBART encoder."""

    def __init__(
        self,
        model_name: str = "nict-astrec-att/prahokbart_base",
        embed_dim: int = 512,
        max_length: int = 256,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the PrahokBART dual encoder.

        Args:
            model_name: HuggingFace model identifier.
            embed_dim: Transformer hidden representation dimension (512 for base).
            max_length: Maximum sequence truncation length (default: 256).
            device: Computing device (CUDA or CPU).
        """
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.max_length = max_length

        target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading Khmer-native backbone '{model_name}' for Approach A4...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        full_bart = MBartForConditionalGeneration.from_pretrained(model_name)

        # Extract only the encoder tower and embed tokens
        self.encoder = full_bart.model.encoder

        # Free decoder memory
        del full_bart.model.decoder
        del full_bart.lm_head

        # Ensure all encoder parameters are trainable
        for param in self.encoder.parameters():
            param.requires_grad = True

        total_params = sum(p.numel() for p in self.encoder.parameters())
        trainable_params = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)

        logger.info(
            f"Initialized PrahokBARTDualEncoder: "
            f"{trainable_params:,} trainable params ({trainable_params / 1e6:.2f}M, 100% trainable)"
        )

        self.to(target_device)

    def _pool(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply masked mean pooling over valid non-padded tokens."""
        mask = attention_mask.unsqueeze(-1).float()
        sum_hidden = (last_hidden_state * mask).sum(dim=1)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / sum_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through PrahokBART encoder with mean pooling and L2 normalization.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len).
            attention_mask: Tensor of shape (batch_size, seq_len).

        Returns:
            Unit-normalized Tensor of shape (batch_size, embed_dim).
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=-1)

    def _encode_batch(
        self,
        texts: Sequence[str],
        prefix: str = "",
        batch_size: int = 32,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Tokenize and encode text batch."""
        target_device = device or next(self.parameters()).device
        prefixed = [f"{prefix}{t}" for t in texts]
        all_embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, len(prefixed), batch_size):
                batch_texts = prefixed[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(target_device)

                emb = self.forward(encoded["input_ids"], encoded["attention_mask"])
                all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)

    def encode_queries(
        self,
        queries: Sequence[str],
        device: Optional[torch.device] = None,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode query strings."""
        return self._encode_batch(queries, prefix="", batch_size=batch_size, device=device)

    def encode_passages(
        self,
        passages: Sequence[str],
        device: Optional[torch.device] = None,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode passage strings."""
        return self._encode_batch(passages, prefix="", batch_size=batch_size, device=device)
