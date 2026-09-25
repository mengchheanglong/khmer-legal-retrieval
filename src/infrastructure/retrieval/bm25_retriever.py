"""
BM25 Sparse Retriever Adapter.

Implements SparseRetrieverPort using rank-bm25 (BM25Okapi).
Supports indexing, querying, serialization to disk, legal-specific tokenization,
and auto-building from data/04_chunks/*.json on first run.
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import regex as re
from rank_bm25 import BM25Okapi

from src.config.logging import get_logger
from src.domain.entities import LegalChunk, RetrievedDocument
from src.domain.ports.sparse_retriever_port import SparseRetrieverPort

logger = get_logger(__name__)


def tokenize_legal_text(text: str) -> list[str]:
    """
    Tokenize legal text for BM25 keyword matching.

    Supports both Khmer (via khmer-nltk word segmentation) and English/alphanumeric text.
    Preserves numbers (like article numbers "315") and legal terms.
    """
    if not text:
        return []

    # If text contains Khmer characters, use khmer-nltk word segmentation
    if re.search(r"[\u1780-\u17ff]", text):
        try:
            from src.dl.text import tokenize_khmer
            tokens = tokenize_khmer(text)
            if tokens:
                return tokens
        except Exception:
            pass

    cleaned = text.lower()
    # Extract alphanumeric tokens, keeping legal identifiers intact
    tokens = re.findall(r"\b[a-z0-9_\u1780-\u17ff]+\b", cleaned)
    return tokens


class BM25Retriever(SparseRetrieverPort):
    """
    Sparse keyword retriever implementing BM25Okapi algorithm.

    High recall for exact article citations (e.g. 'Article 315')
    and statutory terminology.
    """

    def __init__(self, index_path: Optional[Path] = None) -> None:
        self._chunks: list[LegalChunk] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None

        base_dir = Path(__file__).resolve().parent.parent.parent.parent
        if index_path:
            self._index_path = Path(index_path)
        else:
            khmer_idx = base_dir / "data" / "indices" / "bm25_khmer_index.pkl"
            self._index_path = khmer_idx if khmer_idx.exists() else (base_dir / "data" / "indices" / "bm25_index.pkl")
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        if self._index_path.exists():
            self.load()
        else:
            # Auto-build from data/04_chunks/*.json if index.pkl is not found
            self._auto_index_from_chunks(base_dir / "data" / "04_chunks")

    def _auto_index_from_chunks(self, chunks_dir: Path) -> None:
        """Automatically load and build index from data/04_chunks/*.json if index.pkl is missing."""
        if not chunks_dir.exists():
            return
        all_chunks: list[LegalChunk] = []
        for json_file in sorted(chunks_dir.glob("*_chunks.json")):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data:
                        all_chunks.append(LegalChunk.model_validate(item))
            except Exception as e:
                logger.warning(f"Failed to auto-load chunks from {json_file}: {e}")

        if all_chunks:
            logger.info("Auto-building BM25 index from chunks directory", count=len(all_chunks))
            self.index(all_chunks)
            try:
                self.save()
            except Exception:
                pass

    def index(self, chunks: list[LegalChunk]) -> None:
        """
        Build or update the BM25 index from a list of LegalChunk objects.

        Args:
            chunks: Legal chunks to index.
        """
        if not chunks:
            logger.warning("Empty chunk list passed to BM25 index.")
            return

        self._chunks = list(chunks)
        self._corpus_tokens = [
            tokenize_legal_text(chunk.content_with_context) for chunk in self._chunks
        ]
        self._bm25 = BM25Okapi(self._corpus_tokens)

        logger.info(
            "BM25 index built",
            total_documents=len(self._chunks),
            vocab_size=len(self._bm25.idf) if hasattr(self._bm25, "idf") else None,
        )

    def search(
        self,
        query: str,
        top_k: int = 50,
        law_filter: Optional[str] = None,
    ) -> list[RetrievedDocument]:
        """
        Search for relevant legal chunks using BM25.

        Args:
            query: User's query string.
            top_k: Maximum number of chunks to return.
            law_filter: Optional statute filter (e.g. 'Civil Code 2007', 'Criminal Code 2009').

        Returns:
            List of RetrievedDocument with sparse_score populated.
        """
        if not self._bm25 or not self._chunks:
            logger.warning("BM25 search called on uninitialized index.")
            return []

        query_tokens = tokenize_legal_text(query)
        if not query_tokens:
            return []

        doc_scores = self._bm25.get_scores(query_tokens)

        # Get sorted indices by score descending
        scored_indices = sorted(
            range(len(doc_scores)),
            key=lambda i: doc_scores[i],
            reverse=True,
        )

        results: list[RetrievedDocument] = []
        clean_filter = law_filter.lower() if law_filter else None
        for idx in scored_indices:
            score = float(doc_scores[idx])
            if score <= 0.0:
                break  # Scores are sorted descending
            chunk = self._chunks[idx]
            if clean_filter:
                chunk_law = chunk.metadata.law_name.lower()
                if "civil" in clean_filter and "civil" not in chunk_law:
                    continue
                if "crim" in clean_filter and "crim" not in chunk_law:
                    continue
            results.append(
                RetrievedDocument(
                    chunk=chunk,
                    sparse_score=score,
                )
            )
            if len(results) >= top_k:
                break

        return results

    def save(self, path: Optional[Path] = None) -> None:
        """Serialize the BM25 index and chunk metadata to disk."""
        target_path = path or self._index_path
        data = {
            "chunks": [chunk.model_dump() for chunk in self._chunks],
            "corpus_tokens": self._corpus_tokens,
        }
        with open(target_path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Saved BM25 index to disk", path=str(target_path), total_documents=len(self._chunks))

    def load(self, path: Optional[Path] = None) -> None:
        """Load a serialized BM25 index from disk."""
        target_path = path or self._index_path
        if not target_path.exists():
            logger.warning("BM25 index is not found on disk", path=str(target_path))
            return

        with open(target_path, "rb") as f:
            data = pickle.load(f)

        self._chunks = [LegalChunk.model_validate(chunk_data) for chunk_data in data["chunks"]]
        self._corpus_tokens = data["corpus_tokens"]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        logger.info(
            "Loaded BM25 index from disk",
            path=str(target_path),
            total_documents=len(self._chunks),
        )
