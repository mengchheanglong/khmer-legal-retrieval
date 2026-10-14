"""
Dependency injection wiring for FastAPI.

Resolves Domain Ports to concrete Infrastructure Adapters
and provides initialized use case instances to route handlers.
"""

from functools import lru_cache
from pathlib import Path

from src.config.settings import Settings, get_settings
from src.application.use_cases.answer_legal_qa import AnswerLegalQAUseCase
from src.application.use_cases.hybrid_retrieve import HybridRetrieveUseCase
from src.domain.ports.embedding_port import EmbeddingPort
from src.infrastructure.ai.openai_embedding import OpenAIEmbedding
from src.infrastructure.ai.openai_llm import OpenAILLM
from src.infrastructure.ai.torch_encoder_embedding import TorchEncoderEmbedding
from src.infrastructure.retrieval.bm25_retriever import BM25Retriever
from src.infrastructure.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.infrastructure.storage.db_session import create_db_engine, get_session_factory
from src.infrastructure.storage.pgvector_repository import PgVectorRepository


@lru_cache()
def get_bm25_retriever() -> BM25Retriever:
    """Singleton BM25 retriever instance."""
    return BM25Retriever()


@lru_cache()
def get_pgvector_repository() -> PgVectorRepository:
    """Singleton pgvector repository with local fallback."""
    settings = get_settings()
    session_factory = None
    try:
        engine = create_db_engine(settings)
        session_factory = get_session_factory(engine)
    except Exception:
        pass
    return PgVectorRepository(session_factory=session_factory, settings=settings)


@lru_cache()
def get_openai_embedding() -> OpenAIEmbedding:
    """Singleton OpenAI embedding adapter."""
    return OpenAIEmbedding()


@lru_cache()
def get_embedder() -> EmbeddingPort:
    """
    Singleton dense embedder for FastAPI.

    Prioritizes the local fine-tuned PyTorch checkpoint (Approach A3: XLM-R)
    if available in results/checkpoints, otherwise falls back to OpenAI embeddings.
    """
    ckpt = Path("results/checkpoints/a3_xlmr_finetune/best.pt")
    if ckpt.exists():
        try:
            return TorchEncoderEmbedding(checkpoint_path=ckpt)
        except Exception:
            pass
    return get_openai_embedding()


@lru_cache()
def get_openai_llm() -> OpenAILLM:
    """Singleton OpenAI LLM generation adapter."""
    return OpenAILLM()


@lru_cache()
def get_reranker() -> CrossEncoderReranker:
    """Singleton Cross-Encoder reranker adapter."""
    return CrossEncoderReranker()


def get_hybrid_retriever() -> HybridRetrieveUseCase:
    """Factory for HybridRetrieveUseCase."""
    settings = get_settings()
    return HybridRetrieveUseCase(
        embedder=get_embedder(),
        vector_store=get_pgvector_repository(),
        sparse_retriever=get_bm25_retriever(),
        reranker=get_reranker(),
        rrf_k=settings.rrf_k,
    )


def get_qa_use_case() -> AnswerLegalQAUseCase:
    """Factory for AnswerLegalQAUseCase."""
    return AnswerLegalQAUseCase(
        retriever=get_hybrid_retriever(),
        llm=get_openai_llm(),
    )
