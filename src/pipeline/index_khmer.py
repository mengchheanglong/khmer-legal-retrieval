"""
Khmer Legal Corpus Indexing Pipeline.

Indexes the official 1,976 Khmer legal chunks (Civil Code 2007 + Criminal Code 2009)
from data/04_chunks/*_kh_chunks.json into:
1. Native Khmer BM25 Sparse Index (using khmer-nltk word segmentation).
2. Native Khmer Dense Vector Store using our fine-tuned Approach A3 checkpoint.
"""

import json
from pathlib import Path
from typing import Optional

import torch

from src.config.logging import get_logger, setup_logging
from src.domain.entities import LegalChunk
from src.infrastructure.ai.torch_encoder_embedding import TorchEncoderEmbedding
from src.infrastructure.retrieval.bm25_retriever import BM25Retriever

setup_logging()
logger = get_logger(__name__)


def build_khmer_indices(
    chunks_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    build_dense: bool = True,
) -> tuple[int, Path, Optional[Path]]:
    """Build sparse and dense indices for the official Khmer legal corpus.

    Args:
        chunks_dir: Directory containing *_kh_chunks.json. Defaults to data/04_chunks.
        output_dir: Directory to save serialized indices. Defaults to data/indices.
        checkpoint_path: Path to fine-tuned A3 checkpoint. Defaults to best.pt.
        build_dense: Whether to compute and save dense embeddings with TorchEncoderEmbedding.

    Returns:
        Tuple of (num_chunks, bm25_index_path, dense_embeddings_path).
    """
    base_dir = Path(__file__).resolve().parent.parent.parent
    c_dir = chunks_dir or (base_dir / "data" / "04_chunks")
    out_dir = output_dir or (base_dir / "data" / "indices")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Khmer chunks
    chunk_files = sorted(c_dir.glob("*_kh_chunks.json"))
    if not chunk_files:
        # Fallback to any chunk file if kh specific not found
        chunk_files = sorted(c_dir.glob("*_chunks.json"))

    if not chunk_files:
        raise FileNotFoundError(f"No chunk files found in {c_dir}")

    all_chunks: list[LegalChunk] = []
    for cf in chunk_files:
        logger.info(f"Loading chunks from {cf.name}...")
        with open(cf, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                all_chunks.append(LegalChunk.model_validate(item))

    logger.info(f"Loaded {len(all_chunks):,} official Khmer statutory chunks.")

    # 2. Build Khmer BM25 Index with word segmentation
    bm25_path = out_dir / "bm25_khmer_index.pkl"
    logger.info("Building Khmer BM25 sparse index with word segmentation...")
    bm25 = BM25Retriever(index_path=bm25_path)
    bm25.index(all_chunks)
    bm25.save(bm25_path)
    logger.info(f"Saved Khmer BM25 index to {bm25_path} ({len(all_chunks)} articles).")

    dense_path: Optional[Path] = None
    if build_dense:
        dense_path = out_dir / "khmer_article_embeddings.pt"
        logger.info("Generating dense embeddings with Approach A3 TorchEncoderEmbedding...")
        embedder = TorchEncoderEmbedding(
            checkpoint_path=checkpoint_path,
            batch_size=32,
        )
        passages = [chunk.content_with_context for chunk in all_chunks]
        embeddings = embedder.embed(passages)
        emb_tensor = torch.tensor(embeddings, dtype=torch.float32)

        torch.save(
            {
                "chunk_ids": [c.chunk_id for c in all_chunks],
                "embeddings": emb_tensor,
                "model_name": embedder._model_name,
                "dimensions": embedder.dimensions,
            },
            dense_path,
        )
        logger.info(f"Saved {len(embeddings)} dense embeddings to {dense_path}.")

    return len(all_chunks), bm25_path, dense_path


if __name__ == "__main__":
    count, bm25_p, dense_p = build_khmer_indices(build_dense=False)
    print(f"Successfully indexed {count} Khmer legal articles.")
    print(f"BM25 Index: {bm25_p}")
