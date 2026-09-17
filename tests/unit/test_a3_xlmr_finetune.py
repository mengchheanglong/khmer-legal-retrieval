"""
Unit tests for Approach A3: XLM-R full fine-tuning dual encoder.

Verifies:
1. All parameters in the encoder backbone are trainable (requires_grad = True).
2. Trainable parameter count equals total parameter count (~278M).
3. Forward pass produces L2-normalized representations of shape (B, 768).
4. Gradient backpropagation flows through the entire 12-layer backbone to input embeddings.
5. Sequence truncation rate calculation functions properly.
"""

import pytest
import torch

from src.dl.models.xlmr_finetune import XLMRFullFinetuneRetriever


@pytest.fixture(scope="module")
def retriever() -> XLMRFullFinetuneRetriever:
    """Fixture providing an instance of XLMRFullFinetuneRetriever on CPU."""
    model = XLMRFullFinetuneRetriever(
        model_name="intfloat/multilingual-e5-base",
        embed_dim=768,
        max_length=256,
        device=torch.device("cpu"),
    )
    return model


def test_all_parameters_are_trainable(retriever: XLMRFullFinetuneRetriever) -> None:
    """Verify that 100% of parameters have requires_grad set to True."""
    for name, param in retriever.encoder.named_parameters():
        assert param.requires_grad, f"Parameter '{name}' is unexpectedly frozen!"


def test_trainable_parameter_count(retriever: XLMRFullFinetuneRetriever) -> None:
    """Verify that trainable parameters count equals total parameters (~278M)."""
    trainable_params = sum(p.numel() for p in retriever.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in retriever.parameters())
    assert trainable_params == total_params
    assert trainable_params > 270_000_000


def test_forward_shape_and_l2_norm(retriever: XLMRFullFinetuneRetriever) -> None:
    """Verify forward() produces unit-normalized vectors with correct dimensions."""
    texts = ["query: មាត្រា ១", "passage: បទប្បញ្ញត្តិទូទៅ"]
    encoded = retriever.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = retriever(encoded["input_ids"], encoded["attention_mask"])

    assert out.shape == (2, 768)
    norms = torch.norm(out, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_gradient_flow_to_embeddings(retriever: XLMRFullFinetuneRetriever) -> None:
    """Verify gradients backpropagate all the way to the lowest embedding layer."""
    retriever.zero_grad()
    input_ids = torch.tensor([[0, 100, 200, 2]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)

    out = retriever(input_ids, attention_mask)
    loss = out.sum()
    loss.backward()

    # Verify gradients at the word embeddings layer
    word_embeddings = retriever.encoder.embeddings.word_embeddings.weight
    assert word_embeddings.grad is not None
    assert torch.any(word_embeddings.grad != 0)


def test_truncation_rate_computation(retriever: XLMRFullFinetuneRetriever) -> None:
    """Verify calculation of sequence truncation rate."""
    short_texts = ["មាត្រា ១", "សិទ្ធិ"]
    rate_short = retriever.compute_truncation_rate(short_texts, max_length=128)
    assert rate_short == 0.0

    long_text = "មាត្រា " * 200  # Will exceed limit
    rate_long = retriever.compute_truncation_rate([long_text], max_length=64)
    assert rate_long == 1.0
