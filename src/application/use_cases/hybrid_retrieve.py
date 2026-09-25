"""
Use Case: Hybrid retrieval combining dense and sparse search.

Pipeline: Query → Embed → Dense Search + BM25 Search → RRF Fusion → Rerank
"""

from src.config.logging import get_logger
from src.application.dtos import RetrievalRequest
from src.domain.entities import RetrievedDocument
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.ports.vector_store_port import VectorStorePort
from src.domain.ports.sparse_retriever_port import SparseRetrieverPort
from src.domain.ports.reranker_port import RerankerPort

logger = get_logger(__name__)


class HybridRetrieveUseCase:
    """
    Combines dense (vector) and sparse (BM25) retrieval with reranking.

    Uses Reciprocal Rank Fusion (RRF) to merge results from both
    retrieval methods, then applies a cross-encoder reranker.
    """

    def __init__(
        self,
        embedder: EmbeddingPort,
        vector_store: VectorStorePort,
        sparse_retriever: SparseRetrieverPort,
        reranker: RerankerPort,
        rrf_k: int = 60,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._sparse_retriever = sparse_retriever
        self._reranker = reranker
        self._rrf_k = rrf_k

    def execute(
        self,
        request: RetrievalRequest,
        dense_top_k: int = 50,
        sparse_top_k: int = 50,
    ) -> list[RetrievedDocument]:
        """
        Run hybrid retrieval for a query.

        Args:
            request: Query parameters (text, filters, top_k).
            dense_top_k: Number of candidates from dense retrieval.
            sparse_top_k: Number of candidates from sparse retrieval.

        Returns:
            Reranked list of RetrievedDocument (length = request.top_k).
        """
        logger.info("Starting hybrid retrieval", query=request.query)

        # Step 1 & 2: Dense retrieval (vector similarity) with fallback
        dense_results: list[RetrievedDocument] = []
        try:
            query_embedding = self._embedder.embed_query(request.query)
            filters = {}
            if request.law_filter:
                filters["law_name"] = request.law_filter
            if request.chapter_filter:
                filters["chapter"] = request.chapter_filter

            dense_results = self._vector_store.search(
                query_embedding=query_embedding,
                top_k=dense_top_k,
                filters=filters or None,
            )
            logger.info("Dense retrieval complete", count=len(dense_results))
        except Exception as e:
            logger.warning("Dense vector retrieval unavailable, falling back to BM25 sparse search", error=str(e))

        # Step 3: Sparse retrieval (BM25)
        try:
            sparse_results = self._sparse_retriever.search(
                query=request.query,
                top_k=sparse_top_k,
                law_filter=request.law_filter,
            )
        except TypeError:
            sparse_results = self._sparse_retriever.search(
                query=request.query,
                top_k=sparse_top_k,
            )
        logger.info("Sparse retrieval complete", count=len(sparse_results))

        # Step 4: Reciprocal Rank Fusion
        if dense_results and sparse_results:
            fused_results = self._reciprocal_rank_fusion(dense_results, sparse_results)
        elif dense_results:
            fused_results = dense_results
        else:
            fused_results = sparse_results

        logger.info("Fusion complete", count=len(fused_results))

        # Step 5: Rerank top candidates
        reranked = self._reranker.rerank(
            query=request.query,
            documents=fused_results[:dense_top_k],
            top_k=request.top_k,
        )
        logger.info("Reranking complete", count=len(reranked))

        return reranked

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievedDocument],
        sparse_results: list[RetrievedDocument],
    ) -> list[RetrievedDocument]:
        """
        Merge dense and sparse results using Reciprocal Rank Fusion.

        RRF(d) = Σ 1 / (k + rank(d)) for each ranking method.
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, RetrievedDocument] = {}

        # Score from dense retrieval
        for rank, doc in enumerate(dense_results):
            chunk_id = doc.chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0)
            rrf_scores[chunk_id] += 1.0 / (self._rrf_k + rank + 1)
            doc_map[chunk_id] = doc

        # Score from sparse retrieval
        for rank, doc in enumerate(sparse_results):
            chunk_id = doc.chunk.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0)
            rrf_scores[chunk_id] += 1.0 / (self._rrf_k + rank + 1)
            if chunk_id not in doc_map:
                doc_map[chunk_id] = doc

        # Build fused results
        fused: list[RetrievedDocument] = []
        for chunk_id, score in sorted(
            rrf_scores.items(), key=lambda x: x[1], reverse=True
        ):
            doc = doc_map[chunk_id]
            fused.append(
                RetrievedDocument(
                    chunk=doc.chunk,
                    dense_score=doc.dense_score,
                    sparse_score=doc.sparse_score,
                    rrf_score=score,
                )
            )

        return fused
