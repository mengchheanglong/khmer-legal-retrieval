"""Regression tests for truthful live retriever selection and scope."""
from unittest.mock import Mock
import numpy as np
import pytest
from src.domain.entities import LegalChunk, LegalMetadata, RetrievedDocument
from src.interfaces.ui import retrieval as demo


@pytest.fixture
def retrieval_inputs(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    """Provide disjoint dense and sparse candidates without loading model weights."""
    chunks = [LegalChunk(chunk_id=str(i), content='article', content_with_context='article',
        metadata=LegalMetadata(law_name='Civil Code 2007' if i==0 else 'Criminal Code 2009', article_number=i+1)) for i in range(2)]
    sparse = Mock(_chunks=chunks)
    sparse.search.return_value = [RetrievedDocument(chunk=chunks[1], sparse_score=1.0)]
    embedder = Mock()
    embedder.embed_query.return_value = [1.0,0.0]
    monkeypatch.setattr(demo,'load_cached_bm25_retriever',lambda:sparse)
    monkeypatch.setattr(demo,'load_cached_torch_embedder',lambda:embedder)
    monkeypatch.setattr(demo,'load_cached_corpus_embeddings',lambda:(np.eye(2),['0','1']))
    monkeypatch.setattr(demo.st,'error',Mock())
    return sparse,embedder


def test_hybrid_executes_dense_branch(retrieval_inputs: tuple[Mock, Mock]) -> None:
    """A Hybrid label containing BM25 must still invoke the neural encoder and RRF."""
    sparse,embedder=retrieval_inputs
    result=demo.execute_unified_retrieval('query','Hybrid: A3 Dense + BM25 Sparse',top_k=2)
    embedder.embed_query.assert_called_once()
    assert len(result)==2
    assert all(d.rrf_score is not None for d in result)


def test_a3_missing_index_does_not_impersonate_bm25(retrieval_inputs: tuple[Mock, Mock], monkeypatch: pytest.MonkeyPatch) -> None:
    """Unavailable A3 returns no results instead of mislabeled sparse output."""
    sparse,_=retrieval_inputs
    monkeypatch.setattr(demo,'load_cached_corpus_embeddings',lambda:(None,None))
    assert demo.execute_unified_retrieval('query','A3: XLM-R Fine-Tuned (Best)')==[]
    sparse.search.assert_not_called()


def test_dense_filter_never_broadens_scope(retrieval_inputs: tuple[Mock, Mock]) -> None:
    """Dense candidates respect the selected statute even when it scores lower."""
    result=demo.execute_unified_retrieval('query','A3',law_filter='Criminal Code 2009')
    assert [d.chunk.metadata.law_name for d in result]==['Criminal Code 2009']


def test_unsupported_model_does_not_return_bm25(retrieval_inputs: tuple[Mock, Mock]) -> None:
    """Historical model labels cannot silently use the sparse retriever."""
    sparse,_=retrieval_inputs
    assert demo.execute_unified_retrieval('query','A1: BiLSTM')==[]
    sparse.search.assert_not_called()
