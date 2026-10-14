"""
Domain entities for Cambodian legal documents.

These are pure data models with no external dependencies.
They represent the core business concepts of the RAG system.
"""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Language(str, Enum):
    """Supported document languages."""
    ENGLISH = "en"
    KHMER = "kh"


class LegalMetadata(BaseModel):
    """
    Hierarchical metadata for a Cambodian legal provision.

    Cambodian laws follow: Law → Book (គន្ថី) → Title → Chapter (ជំពូក)
    → Section (ផ្នែក) → Article (មាត្រា)
    """
    law_name: str = Field(..., description="Name of the law (e.g., 'Civil Code 2007')")
    law_name_kh: Optional[str] = Field(None, description="Khmer name of the law")
    book: Optional[str] = Field(None, description="Book number and title")
    title: Optional[str] = Field(None, description="Title within the book")
    chapter: Optional[str] = Field(None, description="Chapter number and title")
    section: Optional[str] = Field(None, description="Section within the chapter")
    article_number: int = Field(..., description="Article number (មាត្រា)")
    article_title: Optional[str] = Field(None, description="Article title if present")
    promulgation_date: Optional[date] = Field(None, description="Date the law was enacted")
    language: Language = Field(default=Language.ENGLISH)
    page_number: Optional[int] = Field(None, description="Source PDF page number")


class LegalChunk(BaseModel):
    """
    A single chunk of legal text with its hierarchical metadata.

    Each chunk typically corresponds to one Article (មាត្រា),
    which is the atomic unit of Cambodian legal text.
    """
    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    content: str = Field(..., description="The raw text content of the chunk")
    content_with_context: str = Field(
        ...,
        description="Content prefixed with hierarchical context for embedding"
    )
    metadata: LegalMetadata
    token_count: Optional[int] = Field(None, description="Number of tokens in content")


class RetrievedDocument(BaseModel):
    """
    A chunk returned from retrieval with relevance scores.

    Contains scores from each retrieval stage for debugging and evaluation.
    """
    chunk: LegalChunk
    dense_score: Optional[float] = Field(None, description="Vector similarity score")
    sparse_score: Optional[float] = Field(None, description="BM25 relevance score")
    rrf_score: Optional[float] = Field(None, description="Reciprocal Rank Fusion score")
    rerank_score: Optional[float] = Field(None, description="Cross-encoder reranker score")


class Citation(BaseModel):
    """
    A verified legal citation extracted from the LLM response.

    Links a claim in the generated answer to a specific article.
    Distinguishes citation presence in retrieved context from factual support.
    """
    law_name: str
    article_number: int
    excerpt: str = Field(..., description="Verbatim text from the cited article")
    is_verified: bool = Field(
        default=False,
        description="Whether this citation was found in the retrieved context (citation presence)"
    )
    has_factual_support: Optional[bool] = Field(
        default=None,
        description="Whether the cited retrieved article provides factual support for the proposition"
    )
    support_notes: Optional[str] = Field(
        default=None,
        description="Explanation of factual grounding vs non-support/irrelevant citation"
    )



class QAResponse(BaseModel):
    """
    The final response from the legal Q&A system.

    Contains the generated answer, citations, source documents,
    and metadata for evaluation and auditing.
    """
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    source_documents: list[RetrievedDocument] = Field(default_factory=list)
    confidence: Optional[float] = Field(
        None,
        description="Model confidence score (0-1)"
    )
    disclaimer: str = Field(
        default="This is AI-generated legal information, not legal advice. "
                "Consult a qualified Cambodian lawyer for professional guidance.",
    )
