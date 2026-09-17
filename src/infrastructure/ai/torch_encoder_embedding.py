"""
PyTorch Encoder Embedding Adapter for Khmer Legal Retrieval.

Implements EmbeddingPort using our best fine-tuned Approach A3
(XLMRFullFinetuneRetriever) or Approach A2 checkpoint, providing
native Khmer dense vector embeddings for queries and statutory articles.
"""

from pathlib import Path
from typing import Optional, Sequence

import torch

from src.config.logging import get_logger
from src.domain.exceptions import EmbeddingError
from src.domain.ports.embedding_port import EmbeddingPort

logger = get_logger(__name__)


class TorchEncoderEmbedding(EmbeddingPort):
    """
    Adapter implementing EmbeddingPort using a locally trained PyTorch checkpoint.

    Features:
    - Native Khmer legal retrieval using our best fine-tuned model (Approach A3).
    - Asymmetric query and passage prefixing ('query: ' vs 'passage: ').
    - Batched encoding with L2 unit normalization.
    - Automatic CPU/CUDA device detection with graceful fallback.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        model_name: str = "intfloat/multilingual-e5-base",
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        """Initialize the TorchEncoderEmbedding adapter.

        Args:
            checkpoint_path: Path to .pt checkpoint. Defaults to results/checkpoints/a3_xlmr_finetune/best.pt.
            model_name: HuggingFace model backbone name.
            batch_size: Batch size for encoding passages.
            device: Torch device identifier ('cpu', 'cuda'). Auto-detected if None.
        """
        self._model_name = model_name
        self._batch_size = batch_size
        self._dim = 768

        if device:
            self._device = torch.device(device)
        else:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve checkpoint path
        if checkpoint_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            default_ckpt = base_dir / "results" / "checkpoints" / "a3_xlmr_finetune" / "best.pt"
            self._checkpoint_path = default_ckpt if default_ckpt.exists() else None
        else:
            self._checkpoint_path = Path(checkpoint_path)

        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """Load transformer backbone and checkpoint weights."""
        from src.dl.models.xlmr_finetune import XLMRFullFinetuneRetriever

        logger.info(
            "Initializing TorchEncoderEmbedding",
            model_name=self._model_name,
            device=str(self._device),
            checkpoint=str(self._checkpoint_path) if self._checkpoint_path else "None (zero-shot)",
        )

        try:
            model = XLMRFullFinetuneRetriever(
                model_name=self._model_name,
                embed_dim=self._dim,
                device=self._device,
            )

            if self._checkpoint_path and self._checkpoint_path.exists():
                logger.info(f"Loading checkpoint weights from {self._checkpoint_path}...")
                ckpt = torch.load(self._checkpoint_path, map_location=self._device, weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict, strict=False)
                logger.info("Successfully loaded fine-tuned model checkpoint.")
            else:
                logger.warning(
                    f"Checkpoint not found at {self._checkpoint_path}. Running with pre-trained backbone."
                )

            model.eval()
            self._model = model

        except Exception as e:
            logger.error("Failed to initialize TorchEncoderEmbedding", error=str(e))
            raise EmbeddingError(f"Could not load PyTorch retriever model: {e}")

    @property
    def dimensions(self) -> int:
        """Return dimensionality of the embedding vectors (768 for XLM-R base)."""
        return self._dim

    def embed_query(self, query: str) -> list[float]:
        """Generate a unit-normalized embedding for a single query string."""
        if not self._model:
            raise EmbeddingError("TorchEncoderEmbedding model is not initialized.")

        try:
            with torch.no_grad():
                tensor_emb = self._model.encode_queries([query], device=self._device, batch_size=1)
                return tensor_emb[0].tolist()
        except Exception as e:
            logger.error("Failed to embed query", error=str(e), query=query)
            raise EmbeddingError(f"Query embedding failed: {e}")

    def embed(self, texts: list[str], batch_size: Optional[int] = None) -> list[list[float]]:
        """Generate unit-normalized embeddings for a list of passage texts.

        Args:
            texts: List of passage strings to embed.
            batch_size: Optional batch size override.

        Returns:
            List of embedding vectors (each 768 floats).
        """
        if not texts:
            return []

        if not self._model:
            raise EmbeddingError("TorchEncoderEmbedding model is not initialized.")

        bs = batch_size or self._batch_size

        try:
            with torch.no_grad():
                tensor_emb = self._model.encode_passages(texts, device=self._device, batch_size=bs)
                return tensor_emb.tolist()
        except Exception as e:
            logger.error("Failed to embed passages batch", error=str(e), count=len(texts))
            raise EmbeddingError(f"Passage batch embedding failed: {e}")
