"""
Unit tests for TorchEncoderEmbedding adapter.

Verifies:
1. Dimension property (768).
2. Single query embedding returns unit-normalized 768-dim vector.
3. Batch passage embedding returns expected list of vectors.
4. Empty input handling.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
import torch


def test_torch_encoder_embedding_dimensions():
    """Verify adapter exposes correct 768 dimensions."""
    from src.infrastructure.ai.torch_encoder_embedding import TorchEncoderEmbedding

    mock_model = MagicMock()
    with patch("src.dl.models.xlmr_finetune.XLMRFullFinetuneRetriever", return_value=mock_model):
        with patch.object(Path, "exists", return_value=False):
            embedder = TorchEncoderEmbedding(checkpoint_path=Path("non_existent.pt"))
            assert embedder.dimensions == 768


def test_torch_encoder_embedding_empty_batch():
    """Verify empty batch returns empty list."""
    from src.infrastructure.ai.torch_encoder_embedding import TorchEncoderEmbedding

    mock_model = MagicMock()
    with patch("src.dl.models.xlmr_finetune.XLMRFullFinetuneRetriever", return_value=mock_model):
        with patch.object(Path, "exists", return_value=False):
            embedder = TorchEncoderEmbedding(checkpoint_path=Path("non_existent.pt"))
            assert embedder.embed([]) == []


def test_torch_encoder_embedding_query():
    """Verify embed_query calls encode_queries and returns float list."""
    from src.infrastructure.ai.torch_encoder_embedding import TorchEncoderEmbedding

    mock_model = MagicMock()
    fake_tensor = torch.randn(1, 768)
    fake_tensor = fake_tensor / fake_tensor.norm(p=2, dim=-1, keepdim=True)
    mock_model.encode_queries.return_value = fake_tensor

    with patch("src.dl.models.xlmr_finetune.XLMRFullFinetuneRetriever", return_value=mock_model):
        with patch.object(Path, "exists", return_value=False):
            embedder = TorchEncoderEmbedding(checkpoint_path=Path("non_existent.pt"))
            vec = embedder.embed_query("តើកិច្ចសន្យាគឺជាអ្វី?")

            assert len(vec) == 768
            assert isinstance(vec[0], float)
            mock_model.encode_queries.assert_called_once()


def test_torch_encoder_embedding_passages_batch():
    """Verify embed calls encode_passages and returns list of float lists."""
    from src.infrastructure.ai.torch_encoder_embedding import TorchEncoderEmbedding

    mock_model = MagicMock()
    fake_tensor = torch.randn(2, 768)
    fake_tensor = fake_tensor / fake_tensor.norm(p=2, dim=-1, keepdim=True)
    mock_model.encode_passages.return_value = fake_tensor

    with patch("src.dl.models.xlmr_finetune.XLMRFullFinetuneRetriever", return_value=mock_model):
        with patch.object(Path, "exists", return_value=False):
            embedder = TorchEncoderEmbedding(checkpoint_path=Path("non_existent.pt"))
            vecs = embedder.embed(["មាត្រា ១", "មាត្រា ២"])

            assert len(vecs) == 2
            assert len(vecs[0]) == 768
            assert len(vecs[1]) == 768
            mock_model.encode_passages.assert_called_once()
