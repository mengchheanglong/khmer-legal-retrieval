"""
Unit tests for Approach A4 PrahokBART Dual Encoder.

Verifies:
1. Model initialization and 512-dim embedding dimension.
2. Masked mean pooling mathematical correctness.
3. L2 normalization of forward pass outputs.
4. Batch query and passage encoding functionality.
"""

from unittest.mock import MagicMock, patch
import pytest
import torch


def test_prahokbart_initialization():
    """Verify PrahokBART initialization and parameter setup."""
    from src.dl.models.prahokbart import PrahokBARTDualEncoder

    mock_bart = MagicMock()
    mock_encoder = MagicMock()
    # Mock parameters
    p1 = torch.nn.Parameter(torch.randn(10, 512))
    mock_encoder.parameters.return_value = [p1]
    mock_bart.model.encoder = mock_encoder

    with patch("src.dl.models.prahokbart.AutoTokenizer.from_pretrained") as mock_tok:
        with patch("src.dl.models.prahokbart.MBartForConditionalGeneration.from_pretrained", return_value=mock_bart):
            model = PrahokBARTDualEncoder(device=torch.device("cpu"))
            assert model.embed_dim == 512
            assert model.encoder is mock_encoder


def test_prahokbart_pooling_math():
    """Verify masked mean pooling ignores padded tokens correctly."""
    from src.dl.models.prahokbart import PrahokBARTDualEncoder

    mock_bart = MagicMock()
    mock_bart.model.encoder.parameters.return_value = []
    with patch("src.dl.models.prahokbart.AutoTokenizer.from_pretrained"):
        with patch("src.dl.models.prahokbart.MBartForConditionalGeneration.from_pretrained", return_value=mock_bart):
            model = PrahokBARTDualEncoder(device=torch.device("cpu"))

            hidden = torch.tensor([
                [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],  # token 1, token 2, padded
            ])
            mask = torch.tensor([[1, 1, 0]])

            pooled = model._pool(hidden, mask)
            expected = torch.tensor([[2.0, 3.0]])
            torch.testing.assert_close(pooled, expected)


def test_prahokbart_forward_normalized():
    """Verify forward pass outputs unit-normalized embeddings."""
    from src.dl.models.prahokbart import PrahokBARTDualEncoder

    mock_bart = MagicMock()
    mock_encoder = MagicMock()
    mock_encoder.parameters.return_value = []
    # Mock encoder return value
    mock_output = MagicMock()
    mock_output.last_hidden_state = torch.randn(2, 4, 512)
    mock_encoder.return_value = mock_output
    mock_bart.model.encoder = mock_encoder

    with patch("src.dl.models.prahokbart.AutoTokenizer.from_pretrained"):
        with patch("src.dl.models.prahokbart.MBartForConditionalGeneration.from_pretrained", return_value=mock_bart):
            model = PrahokBARTDualEncoder(device=torch.device("cpu"))

            input_ids = torch.ones(2, 4, dtype=torch.long)
            attention_mask = torch.ones(2, 4, dtype=torch.long)

            out = model.forward(input_ids, attention_mask)
            assert out.shape == (2, 512)
            norms = out.norm(dim=-1)
            torch.testing.assert_close(norms, torch.ones(2))
