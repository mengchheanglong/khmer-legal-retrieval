"""
Unit tests for Approach A2: XLM-R frozen dual encoder with linear projection probe.

Verifies:
1. All transformer backbone parameters are strictly frozen (requires_grad = False).
2. Trainable parameters count is exactly 590,592 (~0.59M) for the 768->768 head.
3. Projection weight is initialized to identity (W = I) and bias to zero (b = 0).
4. Direct projection method output is L2-normalized.
5. Gradient flow updates linear head parameters only and leaves backbone unaffected.
6. Extracted backbone embeddings projected through head match end-to-end forward pass.
"""

import pytest
import torch
import torch.nn.functional as F

from src.dl.models.linear_probe import XLMRLinearProbeRetriever


@pytest.fixture(scope="module")
def retriever() -> XLMRLinearProbeRetriever:
    """Fixture providing an instance of XLMRLinearProbeRetriever on CPU."""
    model = XLMRLinearProbeRetriever(
        model_name="intfloat/multilingual-e5-base",
        embed_dim=768,
        projection_dim=768,
        device=torch.device("cpu"),
    )
    return model


def test_backbone_parameters_are_frozen(retriever: XLMRLinearProbeRetriever) -> None:
    """Verify all backbone parameters have requires_grad set to False."""
    for name, param in retriever.backbone.named_parameters():
        assert not param.requires_grad, f"Backbone parameter '{name}' is not frozen!"


def test_trainable_parameters_count(retriever: XLMRLinearProbeRetriever) -> None:
    """Verify exactly 590,592 trainable parameters (768 * 768 + 768)."""
    trainable_params = sum(p.numel() for p in retriever.parameters() if p.requires_grad)
    expected = 768 * 768 + 768  # 589,824 + 768 = 590,592
    assert trainable_params == expected, f"Expected {expected} trainable params, found {trainable_params}"


def test_identity_initialization(retriever: XLMRLinearProbeRetriever) -> None:
    """Verify projection layer is initialized to identity matrix with zero bias."""
    weight = retriever.proj.weight.data
    bias = retriever.proj.bias.data

    expected_weight = torch.eye(768)
    expected_bias = torch.zeros(768)

    assert torch.allclose(weight, expected_weight, atol=1e-6)
    assert torch.allclose(bias, expected_bias, atol=1e-6)

    # Test linear transformation on arbitrary random vector
    x = torch.randn(4, 768)
    out = retriever.proj(x)
    assert torch.allclose(out, x, atol=1e-5)


def test_project_l2_normalization(retriever: XLMRLinearProbeRetriever) -> None:
    """Verify project() produces unit-normalized vectors."""
    x = torch.randn(8, 768) * 10.0
    projected = retriever.project(x)

    norms = torch.norm(projected, p=2, dim=-1)
    expected_norms = torch.ones(8)
    assert torch.allclose(norms, expected_norms, atol=1e-5)


def test_gradient_flow_to_linear_head_only(retriever: XLMRLinearProbeRetriever) -> None:
    """Verify backpropagation computes gradients for linear probe and not for backbone."""
    retriever.zero_grad()
    x = torch.randn(2, 768, requires_grad=True)
    out = retriever.project(x)
    loss = out.sum()
    loss.backward()

    # Projection head must have valid non-zero gradients
    assert retriever.proj.weight.grad is not None
    assert retriever.proj.bias.grad is not None
    assert torch.any(retriever.proj.weight.grad != 0)

    # Backbone parameters must have no gradients
    for name, param in retriever.backbone.named_parameters():
        assert param.grad is None, f"Backbone parameter '{name}' unexpectedly received gradients!"


def test_forward_and_caching_equivalence(retriever: XLMRLinearProbeRetriever) -> None:
    """Verify extracting backbone representations and projecting yields identical results to forward()."""
    queries = ["មាត្រា ១.- បទប្បញ្ញត្តិទូទៅ", "កិច្ចសន្យាជួល"]
    encoded = retriever.tokenizer(
        [f"query: {q}" for q in queries],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        # End-to-end forward
        out_forward = retriever(encoded["input_ids"], encoded["attention_mask"])

        # Two-stage: extract backbone then project
        outputs = retriever.backbone(**encoded)
        pooled = retriever._pool(outputs.last_hidden_state, encoded["attention_mask"])
        out_cached = retriever.project(pooled)

    assert torch.allclose(out_forward, out_cached, atol=1e-5)
