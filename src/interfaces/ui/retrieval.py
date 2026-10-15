"""Cached, benchmark-aligned retrieval for the local Streamlit demonstration."""
import hashlib
from pathlib import Path
from typing import Optional, TYPE_CHECKING
import numpy as np
import streamlit as st
import torch
from src.domain.entities import LegalChunk, RetrievedDocument
from src.infrastructure.retrieval.bm25_retriever import BM25Retriever

if TYPE_CHECKING:
    from src.infrastructure.ai.torch_encoder_embedding import TorchEncoderEmbedding

repo_root = Path(__file__).resolve().parents[3]

@st.cache_resource(show_spinner="Loading Khmer BM25 Sparse Index (1,976 articles)...")
def load_cached_bm25_retriever() -> BM25Retriever:
    """Load singleton BM25 retriever with 1,976 official Khmer legal chunks."""
    idx_path = repo_root / "data" / "indices" / "bm25_khmer_index.pkl"
    return BM25Retriever(index_path=idx_path, benchmark_mode=True)


@st.cache_resource(show_spinner="Loading A3 XLM-R Fine-Tuned Encoder...")
def load_cached_torch_embedder() -> "TorchEncoderEmbedding | None":
    """Load fine-tuned Approach A3 (XLM-R) embedding adapter."""
    try:
        from src.infrastructure.ai.torch_encoder_embedding import TorchEncoderEmbedding
        ckpt = repo_root / "results" / "checkpoints" / "a3_xlmr_finetune" / "best.pt"
        if not ckpt.exists():
            return None
        embedder = TorchEncoderEmbedding(
            checkpoint_path=ckpt,
            device="cpu",
        )
        return embedder
    except Exception as e:
        st.error(f"A3 could not load: {e}. Select BM25 explicitly to continue.")
        return None


@st.cache_resource(show_spinner="Loading Precomputed Corpus Embeddings...")
def load_cached_corpus_embeddings() -> tuple[np.ndarray | None, list[str] | None]:
    """Load pre-encoded matrix for the 1,976 Khmer statutory articles if present."""
    emb_file = repo_root / "data" / "indices" / "khmer_benchmark_embeddings.pt"
    if emb_file.exists():
        try:
            data = torch.load(emb_file, map_location="cpu", weights_only=False)
            inputs = {
                "corpus_sha256": repo_root / "data/05_splits/corpus.jsonl",
                "checkpoint_sha256": repo_root / "results/checkpoints/a3_xlmr_finetune/best.pt",
            }
            for key, path in inputs.items():
                with path.open("rb") as handle:
                    if data.get(key) != hashlib.file_digest(handle, "sha256").hexdigest():
                        raise ValueError("Index inputs changed; run python scripts/build_demo_index.py")
            matrix = data["embeddings"].numpy()
            chunk_ids = data["chunk_ids"]
            if matrix.shape != (1976, 768) or len(set(chunk_ids)) != 1976 or not np.isfinite(matrix).all():
                raise ValueError("Invalid corpus embedding dimensions, IDs, or values")
            # Unit normalize
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            matrix = matrix / norms
            return matrix, chunk_ids
        except Exception as exc:
            st.error(f"A3 corpus index could not load: {exc}")
            return None, None
    return None, None


def execute_unified_retrieval(
    query: str,
    retriever_choice: str,
    top_k: int = 5,
    law_filter: Optional[str] = None,
) -> list[RetrievedDocument]:
    """Execute article retrieval according to the user's selected architecture."""
    bm25 = load_cached_bm25_retriever()

    # 1. Pure BM25 Sparse Search
    if retriever_choice.startswith("BM25"):
        return bm25.search(query=query, top_k=top_k, law_filter=law_filter)

    # 2. A3 Full Fine-Tune Dense Search
    if "A3" in retriever_choice or "Hybrid" in retriever_choice:
        corpus_matrix, chunk_ids = load_cached_corpus_embeddings()
        embedder = load_cached_torch_embedder()

        dense_docs: list[RetrievedDocument] = []
        if corpus_matrix is not None and embedder is not None:
            try:
                q_emb = np.array(embedder.embed_query(query), dtype=np.float32)
                norm = np.linalg.norm(q_emb)
                if norm > 0:
                    q_emb = q_emb / norm
                sims = np.dot(corpus_matrix, q_emb)

                # Map chunks by chunk_id
                chunks_map = {c.chunk_id: c for c in bm25._chunks}
                sorted_idx = np.argsort(-sims)

                clean_law = law_filter.lower() if law_filter else None
                for idx in sorted_idx:
                    cid = chunk_ids[idx]
                    chunk = chunks_map.get(cid)
                    if not chunk:
                        continue
                    if clean_law:
                        c_law = chunk.metadata.law_name.lower()
                        if "civil" in clean_law and "civil" not in c_law:
                            continue
                        if "crim" in clean_law and "crim" not in c_law:
                            continue
                    dense_docs.append(
                        RetrievedDocument(
                            chunk=chunk,
                            dense_score=float(sims[idx]),
                        )
                    )
                    if len(dense_docs) >= (top_k * 2 if "Hybrid" in retriever_choice else top_k):
                        break
            except Exception as e:
                st.info(f"Dense vector search note: {e}")

        # If pure A3 requested and dense results found
        if "A3" in retriever_choice and "Hybrid" not in retriever_choice:
            if dense_docs:
                return dense_docs[:top_k]
            st.error("A3 retrieval is unavailable. Check the checkpoint and corpus index, or select BM25 explicitly.")
            return []

        # 3. Hybrid Search (RRF of Dense + BM25)
        bm25_docs = bm25.search(query=query, top_k=top_k * 2, law_filter=law_filter)
        if not dense_docs:
            st.error("Hybrid retrieval needs A3 and BM25. Select BM25 explicitly while A3 is unavailable.")
            return []

        # Reciprocal Rank Fusion
        rrf_k = 60
        scores: dict[str, float] = {}
        doc_store: dict[str, LegalChunk] = {}

        for rank, d in enumerate(dense_docs):
            cid = d.chunk.chunk_id
            doc_store[cid] = d.chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        for rank, d in enumerate(bm25_docs):
            cid = d.chunk.chunk_id
            doc_store[cid] = d.chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)

        sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:top_k]
        return [
            RetrievedDocument(chunk=doc_store[cid], rrf_score=scores[cid])
            for cid in sorted_cids
        ]

    st.error("This retriever is not implemented in the live demo.")
    return []


