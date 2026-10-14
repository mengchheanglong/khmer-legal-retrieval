"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.interfaces.api.main import app


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_health_endpoint(client: TestClient):
    """GET /health should return 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "rag_cambodia_law"


def test_retrieval_endpoint(client: TestClient):
    """POST /api/v1/retrieve should return ranked results."""
    response = client.post(
        "/api/v1/retrieve",
        json={"query": "កិច្ចសន្យា", "top_k": 3},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "កិច្ចសន្យា"
    assert "results" in data
    assert len(data["results"]) > 0
