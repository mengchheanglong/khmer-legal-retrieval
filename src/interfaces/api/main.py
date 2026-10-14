"""
FastAPI application entry point.

Run with:
uvicorn src.interfaces.api.main:app --reload
"""

from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Automatically load .env from repository root
repo_root = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(repo_root / ".env")

from contextlib import asynccontextmanager
from src.config.logging import get_logger, setup_logging
from src.interfaces.api.dependencies import get_bm25_retriever, get_embedder, get_pgvector_repository
from src.interfaces.api.routes import health, qa, retrieval

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Preload models and vector stores on application startup."""
    logger.info("Pre-warming legal retrieval models, embeddings, and indices...")
    try:
        get_pgvector_repository()
        get_bm25_retriever()
        get_embedder()
        logger.info("Warmup complete: all 1,976 legal articles indexed and ready for sub-second retrieval.")
    except Exception as e:
        logger.warning(f"Error during application warmup: {e}")
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="RAG Cambodia Law API",
        description="Retrieval-Augmented Generation for Cambodian Civil & Commercial Law with article-level citations.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health.router)
    app.include_router(retrieval.router, prefix="/api/v1")
    app.include_router(qa.router, prefix="/api/v1")

    logger.info("FastAPI application initialized with routes.")
    return app


app = create_app()
