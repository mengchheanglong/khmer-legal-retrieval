"""
Data Transfer Objects for use case inputs and outputs.

DTOs decouple the interface layer from domain entities,
providing clean request/response schemas for each use case.
"""

from typing import Optional

from pydantic import BaseModel, Field


# ── Ingestion ──────────────────────────────────────────────────────────

class IngestDocumentRequest(BaseModel):
    """Request to ingest a legal PDF document."""
    pdf_path: str = Field(..., description="Path to the PDF file")
    law_name: str = Field(..., description="Name of the law")
    language: str = Field(default="en", description="Language code (en or kh)")
    start_page: int = Field(default=1, description="First page to extract (1-indexed)")
    end_page: Optional[int] = Field(None, description="Last page to extract")


class IngestDocumentResponse(BaseModel):
    """Result of a document ingestion."""
    law_name: str
    chunks_created: int
    chunks_embedded: int
    status: str = "success"


# ── Retrieval ──────────────────────────────────────────────────────────

class RetrievalRequest(BaseModel):
    """Request for hybrid search retrieval."""
    query: str = Field(..., description="Search query text")
    top_k: int = Field(default=5, description="Number of results to return")
    law_filter: Optional[str] = Field(None, description="Filter by law name")
    chapter_filter: Optional[str] = Field(None, description="Filter by chapter")
    retriever_choice: Optional[str] = Field(None, description="Retriever choice (e.g. a3_xlmr, hybrid, bm25)")


# ── Q&A ────────────────────────────────────────────────────────────────

class LegalQARequest(BaseModel):
    """Request for a legal Q&A query."""
    question: str = Field(..., description="The legal question to answer")
    top_k: int = Field(default=5, description="Number of source articles to use")
    law_filter: Optional[str] = Field(None, description="Filter by law name")
    model: Optional[str] = Field(default=None, description="LLM model (e.g. deepseek-chat, deepseek-reasoner, gpt-4o)")
    api_key: Optional[str] = Field(default=None, description="Optional override API key")


class LegalQAResponse(BaseModel):
    """Response from the legal Q&A system."""
    question: str
    answer: str
    reasoning_content: Optional[str] = Field(default=None, description="Chain-of-thought from reasoning models")
    model_used: Optional[str] = Field(default=None, description="Model used for generation")
    citations: list[dict] = Field(default_factory=list)
    source_articles: list[dict] = Field(default_factory=list)
    disclaimer: str = (
        "This is AI-generated legal information, not legal advice. "
        "Consult a qualified Cambodian lawyer for professional guidance."
    )
