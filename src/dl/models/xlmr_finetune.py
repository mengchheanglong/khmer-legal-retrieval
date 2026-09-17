"""
Approach A3: XLM-RoBERTa full fine-tuning dual encoder for Khmer legal retrieval.

Implements Siamese dual-tower architecture with:
- Pre-trained backbone: intfloat/multilingual-e5-base (XLM-RoBERTa-base architecture).
- All 278M parameters are fully trainable (requires_grad = True across all layers).
- Shared query and passage tower with asymmetric 'query: ' and 'passage: ' prefixes.
- Masked mean pooling over valid tokens.
- L2-normalization of final 768-dimensional dense sentence embeddings.
- Truncation rate measurement helper.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from src.config.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


class XLMRFullFinetuneRetriever(nn.Module):
    """Dual-encoder retriever with end-to-end trainable XLM-RoBERTa backbone."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        embed_dim: int = 768,
        max_length: int = 256,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the fine-tuning retriever.

        Args:
            model_name: Pretrained Hugging Face model identifier.
            embed_dim: Transformer hidden dimension (default: 768).
            max_length: Maximum sequence length for truncation (default: 256).
            device: Computing device (CUDA or CPU).
        """
        super().__init__()
        self.model_name = model_name
        self.embed_dim = embed_dim
        self.max_length = max_length

        target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading pre-trained backbone '{model_name}' for A3 full fine-tuning...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

        # Ensure all parameters are trainable
        for param in self.encoder.parameters():
            param.requires_grad = True

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(
            f"Initialized XLMRFullFinetuneRetriever: "
            f"{trainable_params:,} trainable params ({trainable_params / 1e6:.2f}M, 100% trainable)"
        )
        assert trainable_params == total_params, "All parameters must be trainable for Approach A3!"

        self.to(target_device)

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
        """Forward pass through transformer encoder with mean pooling and L2 normalization.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len).
            attention_mask: Tensor of shape (batch_size, seq_len).

        Returns:
            Tensor of shape (batch_size, embed_dim), unit-normalized.
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
        """Encode query strings with 'query: ' prefix."""
        return self._encode_batch(queries, prefix="query: ", batch_size=batch_size, device=device)

    def encode_passages(
        self,
        passages: Sequence[str],
        device: Optional[torch.device] = None,
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode passage strings with 'passage: ' prefix."""
        return self._encode_batch(passages, prefix="passage: ", batch_size=batch_size, device=device)

    def compute_truncation_rate(self, texts: Sequence[str], max_length: Optional[int] = None) -> float:
        """Calculate the proportion of input texts truncated at max_length.

        Args:
            texts: List of raw text strings.
            max_length: Sequence length limit (default: self.max_length).

        Returns:
            Float representing truncation fraction in [0.0, 1.0].
        """
        limit = max_length or self.max_length
        truncated = 0
        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            # Add 2 for <s> and </s> special tokens
            if len(tokens) + 2 > limit:
                truncated += 1
        return truncated / max(1, len(texts))
