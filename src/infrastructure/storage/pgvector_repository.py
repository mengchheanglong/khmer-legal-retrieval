"""
PgVector and Local Vector Repository.

Implements VectorStorePort. Connects to PostgreSQL with pgvector
for production, and provides high-performance local vector similarity
fallback (via numpy cosine similarity) when running without a database.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from src.config.logging import get_logger
from src.config.settings import Settings, get_settings
from src.domain.entities import Language, LegalChunk, LegalMetadata, RetrievedDocument
from src.domain.ports.vector_store_port import VectorStorePort
from src.infrastructure.storage.models import LegalChunkModel

logger = get_logger(__name__)


class PgVectorRepository(VectorStorePort):
    """
    Vector store adapter implementing VectorStorePort.

    Supports:
    - PostgreSQL + pgvector with HNSW cosine distance querying
    - Seamless local memory/file fallback for offline/local development
    """

    def __init__(
        self,
        session_factory=None,
        local_store_path: Optional[Path] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self._session_factory = session_factory
        self._settings = settings or get_settings()

        # Local vector store structures (used as local cache or offline store)
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        self._local_store_path = local_store_path or (base_dir / "data" / "indices" / "local_vectors.pkl")
        self._local_store_path.parent.mkdir(parents=True, exist_ok=True)

        self._local_chunks: list[LegalChunk] = []
        self._local_embeddings: Optional[np.ndarray] = None

        if self._local_store_path.exists():
            self._load_local_store()

        if len(self._local_chunks) == 0:
            self._load_precomputed_embeddings(base_dir)

    def store(
        self,
        chunks: list[LegalChunk],
        embeddings: list[list[float]],
    ) -> None:
        """Store chunks and their embedding vectors in both DB and local store."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match")

        # 1. Update local vector storage
        self._local_chunks = list(chunks)
        self._local_embeddings = np.array(embeddings, dtype=np.float32)
        self._save_local_store()
        logger.info(f"Stored {len(chunks)} embeddings in local vector store.")

        # 2. If Postgres session factory is available, upsert to pgvector
        if self._session_factory:
            try:
                with self._session_factory() as session:
                    for chunk, emb in zip(chunks, embeddings):
                        db_model = LegalChunkModel(
                            id=chunk.chunk_id,
                            law_name=chunk.metadata.law_name,
                            law_name_kh=chunk.metadata.law_name_kh,
                            book=chunk.metadata.book,
                            title=chunk.metadata.title,
                            chapter=chunk.metadata.chapter,
                            section=chunk.metadata.section,
                            article_number=chunk.metadata.article_number,
                            article_title=chunk.metadata.article_title,
                            language=chunk.metadata.language.value,
                            promulgation_date=chunk.metadata.promulgation_date,
                            page_number=chunk.metadata.page_number,
                            content=chunk.content,
                            content_with_context=chunk.content_with_context,
                            embedding=emb,
                        )
                        session.merge(db_model)
                    session.commit()
                logger.info(f"Persisted {len(chunks)} chunks to PostgreSQL/pgvector.")
            except Exception as e:
                logger.warning(f"Failed to persist to PostgreSQL (using local store): {e}")

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 50,
        filters: Optional[dict] = None,
    ) -> list[RetrievedDocument]:
        """
        Perform vector similarity search (cosine distance).

        Uses pgvector if DB is connected, otherwise uses fast numpy cosine similarity.
        """
        # Try database first if session factory is configured
        if self._session_factory:
            try:
                return self._search_pgvector(query_embedding, top_k, filters)
            except Exception as e:
                logger.debug(f"pgvector query fallback to local: {e}")

        # Fallback to local numpy similarity
        return self._search_local(query_embedding, top_k, filters)

    def _search_local(
        self,
        query_embedding: list[float],
        top_k: int = 50,
        filters: Optional[dict] = None,
    ) -> list[RetrievedDocument]:
        """Search using local in-memory numpy cosine similarity."""
        if self._local_embeddings is None or len(self._local_chunks) == 0:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        norms = np.linalg.norm(self._local_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_embs = self._local_embeddings / norms

        similarities = np.dot(normalized_embs, q_vec)

        # Apply metadata filters if present
        valid_indices = []
        for idx, chunk in enumerate(self._local_chunks):
            if filters:
                if "law_name" in filters and chunk.metadata.law_name != filters["law_name"]:
                    continue
                if "chapter" in filters and chunk.metadata.chapter != filters["chapter"]:
                    continue
            valid_indices.append(idx)

        if not valid_indices:
            return []

        # Sort filtered matches by similarity descending
        filtered_scores = [(idx, float(similarities[idx])) for idx in valid_indices]
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        top_results = filtered_scores[:top_k]

        return [
            RetrievedDocument(
                chunk=self._local_chunks[idx],
                dense_score=score,
            )
            for idx, score in top_results
        ]

    def _search_pgvector(
        self,
        query_embedding: list[float],
        top_k: int = 50,
        filters: Optional[dict] = None,
    ) -> list[RetrievedDocument]:
        """Search using pgvector cosine distance operator in PostgreSQL."""
        with self._session_factory() as session:
            stmt = select(
                LegalChunkModel,
                LegalChunkModel.embedding.cosine_distance(query_embedding).label("distance"),
            )

            if filters:
                if "law_name" in filters:
                    stmt = stmt.where(LegalChunkModel.law_name == filters["law_name"])
                if "chapter" in filters:
                    stmt = stmt.where(LegalChunkModel.chapter == filters["chapter"])

            stmt = stmt.order_by("distance").limit(top_k)
            rows = session.execute(stmt).all()

            results: list[RetrievedDocument] = []
            for row in rows:
                m: LegalChunkModel = row[0]
                distance: float = row[1]
                similarity = 1.0 - distance  # Convert distance to similarity score

                metadata = LegalMetadata(
                    law_name=m.law_name,
                    law_name_kh=m.law_name_kh,
                    book=m.book,
                    title=m.title,
                    chapter=m.chapter,
                    section=m.section,
                    article_number=m.article_number,
                    article_title=m.article_title,
                    promulgation_date=m.promulgation_date,
                    language=Language(m.language),
                    page_number=m.page_number,
                )
                chunk = LegalChunk(
                    chunk_id=m.id,
                    content=m.content,
                    content_with_context=m.content_with_context,
                    metadata=metadata,
                )
                results.append(RetrievedDocument(chunk=chunk, dense_score=similarity))

            return results

    def delete_by_law(self, law_name: str) -> int:
        """Delete chunks for a specific law."""
        deleted_count = 0
        if self._session_factory:
            try:
                with self._session_factory() as session:
                    stmt = text("DELETE FROM legal_chunks WHERE law_name = :law")
                    res = session.execute(stmt, {"law": law_name})
                    session.commit()
                    deleted_count = res.rowcount
            except Exception as e:
                logger.warning(f"Error deleting from DB: {e}")

        # Update local store
        keep_indices = [
            i for i, c in enumerate(self._local_chunks) if c.metadata.law_name != law_name
        ]
        self._local_chunks = [self._local_chunks[i] for i in keep_indices]
        if self._local_embeddings is not None and len(keep_indices) > 0:
            self._local_embeddings = self._local_embeddings[keep_indices]
        else:
            self._local_embeddings = None
        self._save_local_store()

        return deleted_count

    def _save_local_store(self) -> None:
        """Serialize local vector store to disk."""
        try:
            with open(self._local_store_path, "wb") as f:
                pickle.dump(
                    {
                        "chunks": self._local_chunks,
                        "embeddings": self._local_embeddings,
                    },
                    f,
                )
        except Exception as e:
            logger.error("Failed to save local vector store", error=str(e))

    def _load_local_store(self) -> None:
        """Load local vector store from disk."""
        try:
            with open(self._local_store_path, "rb") as f:
                data = pickle.load(f)
                self._local_chunks = data.get("chunks", [])
                self._local_embeddings = data.get("embeddings")
            logger.info(f"Loaded {len(self._local_chunks)} vectors from local store.")
        except Exception as e:
            logger.warning(f"Could not load local vector store: {e}")

    def _load_precomputed_embeddings(self, base_dir: Path) -> None:
        """Load precomputed 1,976 embeddings from data/indices/khmer_article_embeddings.pt."""
        emb_path = base_dir / "data" / "indices" / "khmer_article_embeddings.pt"
        chunks_dir = base_dir / "data" / "04_chunks"
        if not emb_path.exists() or not chunks_dir.exists():
            return

        try:
            import torch
            emb_data = torch.load(emb_path, map_location="cpu")
            chunk_ids = emb_data.get("chunk_ids", [])
            raw_embs = emb_data.get("embeddings")

            if raw_embs is None or not chunk_ids:
                return

            if hasattr(raw_embs, "numpy"):
                embeddings_np = raw_embs.numpy().astype(np.float32)
            else:
                embeddings_np = np.array(raw_embs, dtype=np.float32)

            # Load chunks from json files
            chunks_map: dict[str, LegalChunk] = {}
            for json_file in sorted(chunks_dir.glob("*_chunks.json")):
                try:
                    with open(json_file, encoding="utf-8") as f:
                        for item in json.load(f):
                            chunk = LegalChunk.model_validate(item)
                            chunks_map[chunk.chunk_id] = chunk
                except Exception as e:
                    logger.warning(f"Error reading {json_file}: {e}")

            # Align chunks in the exact same order as chunk_ids
            aligned_chunks = []
            valid_indices = []
            for i, cid in enumerate(chunk_ids):
                if cid in chunks_map:
                    aligned_chunks.append(chunks_map[cid])
                    valid_indices.append(i)

            if aligned_chunks:
                self._local_chunks = aligned_chunks
                self._local_embeddings = embeddings_np[valid_indices]
                logger.info(
                    "Loaded precomputed article embeddings into local vector store",
                    count=len(self._local_chunks),
                    shape=list(self._local_embeddings.shape),
                )
                self._save_local_store()
        except Exception as e:
            logger.warning(f"Could not load precomputed embeddings: {e}")
