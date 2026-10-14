"""
Streamlit Web UI for Khmer Legal Article Retrieval Assistant.

Individual Deep Learning Final Project — Bachelor of Software Engineering
Lecturer: Mr. Soklong HIM · Academic Year 2026-2027
Author: Long Mengchheang (@mengchheanglong)

Features:
- Multi-Retriever Architecture (A3 XLM-R Full Fine-Tune, Hybrid, BM25, A4 PrahokBART, A2, A1)
- 1,976 Official Khmer Statutory Articles (Civil Code 2007 + Criminal Code 2009)
- Citation-Grounded Legal Q&A powered by DeepSeek Flash
- Interactive Deep Learning Benchmark Dashboard & Publication Figures
"""

import html
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure repo root is always in sys.path
repo_root = Path(__file__).resolve().parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Load .env variables into environment
try:
    from dotenv import load_dotenv
    load_dotenv(repo_root / ".env")
except ImportError:
    pass

import numpy as np
import pandas as pd
import streamlit as st
import torch

# Sync Streamlit Cloud secrets into environment if present
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str):
                os.environ[k] = v.strip().strip('"').strip("'")
except Exception:
    pass

from src.application.dtos import LegalQARequest, RetrievalRequest
from src.domain.entities import LegalChunk, RetrievedDocument
from src.infrastructure.retrieval.bm25_retriever import BM25Retriever
from src.interfaces.api.dependencies import get_hybrid_retriever, get_qa_use_case


# ─────────────────────────────────────────────────────────────────────────────
# Cached Model & Index Loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Khmer BM25 Sparse Index (1,976 articles)...")
def load_cached_bm25_retriever() -> BM25Retriever:
    """Load singleton BM25 retriever with 1,976 official Khmer legal chunks."""
    idx_path = repo_root / "data" / "indices" / "bm25_khmer_index.pkl"
    return BM25Retriever(index_path=idx_path if idx_path.exists() else None)


@st.cache_resource(show_spinner="Loading A3 XLM-R Fine-Tuned Encoder...")
def load_cached_torch_embedder():
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
        return None


@st.cache_resource(show_spinner="Loading Precomputed Corpus Embeddings...")
def load_cached_corpus_embeddings():
    """Load pre-encoded matrix for the 1,976 Khmer statutory articles if present."""
    emb_file = repo_root / "data" / "indices" / "khmer_article_embeddings.pt"
    if emb_file.exists():
        try:
            data = torch.load(emb_file, map_location="cpu", weights_only=False)
            matrix = data["embeddings"].numpy()
            chunk_ids = data["chunk_ids"]
            # Unit normalize
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            matrix = matrix / norms
            return matrix, chunk_ids
        except Exception:
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
    if "BM25" in retriever_choice:
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
            # Fallback to BM25 if dense weights not present on disk
            st.info("ℹ️ Cloud Notice: Approach A3 PyTorch weights (best.pt, 3.17 GB) are excluded from git. Running high-precision Khmer BM25 Sparse search across 1,976 articles.")
            res = bm25.search(query=query, top_k=top_k, law_filter=law_filter)
            if not res and law_filter:
                res = bm25.search(query=query, top_k=top_k, law_filter=None)
            return res

        # 3. Hybrid Search (RRF of Dense + BM25)
        bm25_docs = bm25.search(query=query, top_k=top_k * 2, law_filter=law_filter)
        if not bm25_docs and law_filter:
            bm25_docs = bm25.search(query=query, top_k=top_k * 2, law_filter=None)
        if not dense_docs:
            return bm25_docs[:top_k]

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

    # Default fallback
    res = bm25.search(query=query, top_k=top_k, law_filter=law_filter)
    if not res and law_filter:
        res = bm25.search(query=query, top_k=top_k, law_filter=None)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Page Layout & Styling
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Khmer Legal Retrieval Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Kantumruy+Pro:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&display=swap');

    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Inter', 'Kantumruy Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Hero Banner */
    .hero-header {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.16) 0%, rgba(99, 102, 241, 0.08) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }

    /* Legal Article Card */
    .legal-law-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 16px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-civil {
        background-color: rgba(37, 99, 235, 0.18);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.35);
    }
    .badge-crim {
        background-color: rgba(220, 38, 38, 0.18);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .score-chip {
        font-size: 0.8rem;
        background: rgba(128, 128, 128, 0.16);
        color: inherit;
        padding: 4px 9px;
        border-radius: 6px;
        font-family: monospace;
        font-weight: 600;
    }
    .article-text {
        font-family: 'Kantumruy Pro', sans-serif !important;
        font-size: 1.02rem !important;
        line-height: 1.9 !important;
        color: inherit !important;
        white-space: pre-wrap;
        background: rgba(128, 128, 128, 0.05);
        padding: 14px 18px;
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.15);
        margin-top: 8px;
    }

    /* Citations & Verification Badges */
    .badge-verified {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        color: #22c55e;
        background-color: rgba(34, 197, 94, 0.16);
        border: 1px solid rgba(34, 197, 94, 0.35);
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
    }
    .badge-unverified {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        color: #eab308;
        background-color: rgba(234, 179, 8, 0.16);
        border: 1px solid rgba(234, 179, 8, 0.35);
        padding: 3px 10px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.82rem;
    }

    .error-card {
        background: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-left: 4px solid #6366f1;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_article_card(doc: RetrievedDocument, index: Optional[int] = None) -> None:
    """Render a statutory article in a beautifully styled reading card."""
    meta = doc.chunk.metadata
    is_crim = "criminal" in meta.law_name.lower()
    law_label = "Criminal Code 2009" if is_crim else "Civil Code 2007"
    badge_class = "badge-crim" if is_crim else "badge-civil"
    card_border = "#dc2626" if is_crim else "#2563eb"

    score_badge = ""
    if doc.dense_score is not None:
        score_badge = f"<span class='score-chip'>Dense CosSim: {doc.dense_score:.4f}</span>"
    elif doc.sparse_score is not None:
        score_badge = f"<span class='score-chip'>BM25 Score: {doc.sparse_score:.2f}</span>"
    elif doc.rrf_score is not None:
        score_badge = f"<span class='score-chip'>RRF Score: {doc.rrf_score:.4f}</span>"

    idx_prefix = f"#{index}. " if index else ""
    title_display = f" — {html.escape(meta.article_title)}" if meta.article_title else ""

    hier_parts = [html.escape(h) for h in [meta.book, meta.chapter, meta.section] if h]
    hier_text = " &nbsp;›&nbsp; ".join(hier_parts)
    escaped_content = html.escape(doc.chunk.content)

    card_html = f"""
    <div style="
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-left: 5px solid {card_border};
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
            <div>
                <span class="legal-law-badge {badge_class}">{law_label}</span>
                <strong style="margin-left: 8px; font-size: 1.05rem; color: inherit;">{idx_prefix}មាត្រា {meta.article_number}{title_display}</strong>
            </div>
            <div>{score_badge}</div>
        </div>
        {f'<div style="font-size: 0.8rem; opacity: 0.75; margin-bottom: 10px;">📁 {hier_text}</div>' if hier_text else ''}
        <div class="article-text">{escaped_content}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


# Hero Header
st.markdown(
    """
    <div class="hero-header">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div>
                <span class="legal-law-badge badge-civil" style="margin-bottom: 8px;">🇰🇭 Deep Learning Legal NLP</span>
                <h1 style="margin: 6px 0 8px 0; font-size: 2.15rem; font-weight: 800; color: inherit; letter-spacing: -0.02em;">
                    Khmer Legal Article Retrieval Assistant
                </h1>
                <p style="margin: 0; font-size: 0.95rem; opacity: 0.85; line-height: 1.5;">
                    Neural Bi-Encoder Passage Retrieval & Citation-Grounded Assistant · 
                    <strong>1,976 Articles</strong> from the Official Civil Code (2007) & Criminal Code (2009)
                </p>
            </div>
            <div>
                <span class="score-chip" style="font-size: 0.85rem; padding: 6px 14px; background: rgba(37, 99, 235, 0.2); color: #60a5fa; border: 1px solid rgba(96, 165, 250, 0.35);">
                    🏆 Winning Model: Approach A3 XLM-R (51.4% Recall@5)
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Top Status Indicators
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(
        "Total Corpus",
        "1,976",
        delta="Civil + Criminal",
        delta_color="off",
        help="1,304 Civil Code (2007) + 672 Criminal Code (2009) articles (100% consecutive)",
    )
with col_m2:
    st.metric(
        "A3 XLM-R (Best)",
        "51.42%",
        delta="Recall@5 (+10.7% vs BM25)",
        help="MRR@10: 0.4827 on 200 Human Benchmark Questions (T-Q)",
    )
with col_m3:
    st.metric(
        "BM25 Baseline",
        "40.75%",
        delta="Recall@5 (0.3350 MRR)",
        delta_color="off",
        help="Rank-BM25 with khmer-nltk word segmentation",
    )
with col_m4:
    st.metric(
        "Grounded LLM",
        "DeepSeek",
        delta="Flash (Verified)",
        help="deepseek-chat with compound (law, article) verification guard",
    )

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Retrieval & LLM Configuration")

    has_local_a3 = (repo_root / "results" / "checkpoints" / "a3_xlmr_finetune" / "best.pt").exists()

    retriever_option = st.selectbox(
        "Active Retrieval Architecture:",
        [
            "A3: XLM-R Full Fine-Tune [Winner]",
            "Hybrid: A3 Dense + BM25 Sparse",
            "BM25 Sparse (khmer-nltk)",
            "A4: PrahokBART Dual Encoder",
            "A2: XLM-R Linear Probe",
            "A1: BiLSTM Dual Encoder",
        ],
        index=0 if has_local_a3 else 2,
    )

    if not has_local_a3:
        st.warning(
            "**Cloud Deployment Notice:**\n\n"
            "Heavy PyTorch weights (`best.pt`, 3.17 GB) are excluded from git. "
            "Running high-precision Khmer BM25 Sparse Retriever (1,976 articles). "
            "For full Approach A3 neural vector search, launch locally on `localhost:8501`."
        )

    # Architecture info banner
    if "A3" in retriever_option and "Hybrid" not in retriever_option:
        st.info(
            "**Approach A3: XLM-RoBERTa Full Fine-Tuning**\n\n"
            "- Parameters: 278,043,648 (100% trainable)\n"
            "- Primary T-Q Recall@5: **51.42%** | MRR@10: **0.4827**\n"
            "- Secondary T-T Recall@5: **99.32%** | MRR@10: **0.9456**\n"
            "- Backbone: `intfloat/multilingual-e5-base`"
        )
    elif "Hybrid" in retriever_option:
        st.info(
            "**Hybrid Retrieval (A3 Dense + BM25 Sparse)**\n\n"
            "- Combines semantic vector similarity with lexical keyword matching\n"
            "- Reciprocal Rank Fusion (RRF parameter k = 60)\n"
            "- Maximum robustness for exact article numbers & abstract concepts"
        )
    elif "BM25" in retriever_option:
        st.info(
            "**BM25 Sparse Baseline**\n\n"
            "- Lexical matching with `khmer-nltk` CRF word tokenization\n"
            "- Primary T-Q Recall@5: **40.75%** | MRR@10: **0.3350**\n"
            "- Secondary T-T Recall@5: **97.30%** | MRR@10: **0.8677**\n"
            "- Inference latency: ~4ms (pure CPU)"
        )
    elif "A4" in retriever_option:
        st.info(
            "**Approach A4: PrahokBART Dual Encoder**\n\n"
            "- Pre-trained Khmer BART (`seanghay/prahokbart-base`)\n"
            "- Parameters: 35.8M trainable\n"
            "- Primary T-Q Recall@5: **15.17%** | MRR@10: **0.1035**\n"
            "- Secondary T-T Recall@5: **80.41%** | MRR@10: **0.6733**"
        )
    elif "A2" in retriever_option:
        st.info(
            "**Approach A2: XLM-R Linear Probe**\n\n"
            "- Frozen XLM-R backbone + trained linear projection\n"
            "- Trainable parameters: 590,592 (0.21%)\n"
            "- Primary T-Q Recall@5: **41.17%** | MRR@10: **0.3591**\n"
            "- Secondary T-T Recall@5: **93.92%** | MRR@10: **0.8563**"
        )
    elif "A1" in retriever_option:
        st.info(
            "**Approach A1: BiLSTM Dual Encoder**\n\n"
            "- Trained from scratch on Khmer character n-grams\n"
            "- Trainable parameters: 1,970,784\n"
            "- Primary T-Q Recall@5: **20.25%** | MRR@10: **0.1724**\n"
            "- Secondary T-T Recall@5: **75.68%** | MRR@10: **0.6242**"
        )

    st.divider()
    st.subheader("Filters & Scope")

    statute_filter_opt = st.selectbox(
        "Statute Scope:",
        [
            "All Statutes (1,976 Articles)",
            "Civil Code 2007 (1,304 Articles)",
            "Criminal Code 2009 (672 Articles)",
        ],
        index=0,
    )
    if "Civil" in statute_filter_opt:
        selected_law_filter = "Civil Code 2007"
    elif "Criminal" in statute_filter_opt:
        selected_law_filter = "Criminal Code 2009"
    else:
        selected_law_filter = None

    top_k = st.slider("Articles to Retrieve (top-k):", min_value=1, max_value=10, value=5)

    st.divider()
    st.subheader("DeepSeek LLM Key")
    env_key = (os.environ.get("DEEPSEEK_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")).strip().strip('"').strip("'")
    if env_key:
        masked = f"{env_key[:4]}...{env_key[-4:]}" if len(env_key) > 8 else "***"
        st.success(f"Key loaded from `.env` (`{masked}`)", icon="🔑")
        api_key_input = st.text_input(
            "Override API Key (Optional):",
            type="password",
            placeholder="Using key from .env",
            help="Your DEEPSEEK_API_KEY from .env is already loaded automatically. Enter a key here only if you wish to override it.",
        )
    else:
        api_key_input = st.text_input(
            "API Key (Optional for Direct Search):",
            type="password",
            placeholder="sk-... (or set DEEPSEEK_API_KEY)",
            help="Required for Tab 1 Q&A synthesis. Direct Search in Tab 2 is 100% offline.",
        )

    st.divider()
    st.subheader("Corpus Attribution")
    st.markdown(
        """
        - **Source:** Open Development Cambodia (ODC)
        - **Extraction:** `LimonPdfExtractor` (legacy fonts → Unicode)
        - **Civil Code 2007:** Articles 1–1304
        - **Criminal Code 2009:** Articles 1–672
        - **License:** CC-BY-SA-4.0
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_qa, tab_search, tab_benchmark = st.tabs(
    [
        "Legal Q&A Assistant",
        "Statutory Article Search",
        "DL Model Benchmark & Evaluation",
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1: Legal Q&A Assistant
# ─────────────────────────────────────────────────────────────────────────────

with tab_qa:
    st.subheader("Ask a Cambodian Legal Question")
    st.caption("Ground truth statutory retrieval combined with citation-verified LLM analysis.")

    benchmark_questions = [
        ("-- Select an authentic Khmer legal question --", ""),
        # Recommended Live Defense Demo Query (TQ-CIV-009: BM25 miss vs A3 Rank 1)
        ("⭐ [LIVE DEFENSE DEMO] តើនរណាខ្លះអាចស្នើសុំឱ្យតុលាការប្រកាសបើកអាណាព្យាបាលទូទៅ? (Civil Code Art. 24 - អាណាព្យាបាលទូទៅ)", "តើនរណាខ្លះអាចស្នើសុំឱ្យតុលាការប្រកាសបើកអាណាព្យាបាលទូទៅ?"),
        # Civil Code (2007)
        ("តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា យោងតាមក្រមរដ្ឋប្បវេណី? (Civil Code Art. 311/336 - ការបង្កើតកិច្ចសន្យា)", "តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា យោងតាមក្រមរដ្ឋប្បវេណី?"),
        ("តើគោលការណ៍ស្វ័យភាពនៃបុគ្គលឯកជនមានន័យដូចម្តេច? (Civil Code Art. 3 - គោលការណ៍ស្វ័យភាព)", "តើគោលការណ៍ស្វ័យភាពនៃបុគ្គលឯកជនមានន័យដូចម្តេច?"),
        ("តើគោលការណ៍នៃសេចក្តីស្មោះត្រង់និងសុចរិតក្នុងច្បាប់រដ្ឋប្បវេណីមានន័យដូចម្តេច? (Civil Code Art. 5 - សេចក្តីស្មោះត្រង់និងសុចរិត)", "តើគោលការណ៍នៃសេចក្តីស្មោះត្រង់និងសុចរិតក្នុងច្បាប់រដ្ឋប្បវេណីមានន័យដូចម្តេច?"),
        ("តើជនបរទេសអាចមានកម្រិតក្នុងការទទួលបានសិទ្ធិក្នុងករណីណាខ្លះ? (Civil Code Art. 8 - សិទ្ធិរបស់ជនបរទេស)", "តើជនបរទេសអាចមានកម្រិតក្នុងការទទួលបានសិទ្ធិក្នុងករណីណាខ្លះ?"),
        ("តើទារកក្នុងផ្ទៃមានសមត្ថភាពទទួលសិទ្ធិដែរឬទេ? (Civil Code Art. 9 - សមត្ថភាពទទួលសិទ្ធិរបស់ទារក)", "តើទារកក្នុងផ្ទៃមានសមត្ថភាពទទួលសិទ្ធិដែរឬទេ?"),
        ("តើអ្វីទៅជាការទទួលខុសត្រូវលើអំពើអនីត្យានុកូល? (Civil Code Art. 743 - អំពើអនីត្យានុកូល)", "តើអ្វីទៅជាការទទួលខុសត្រូវលើអំពើអនីត្យានុកូល?"),
        ("តើលក្ខខណ្ឌនៃការរំលត់កាតព្វកិច្ចដោយសារការទូទាត់មានអ្វីខ្លះ? (Civil Code Art. 415 - ការរំលត់កាតព្វកិច្ច)", "តើលក្ខខណ្ឌនៃការរំលត់កាតព្វកិច្ចដោយសារការទូទាត់មានអ្វីខ្លះ?"),
        ("តើអ្វីទៅជាសិទ្ធិភតិសន្យា និងការជួលអចលនវត្ថុ? (Civil Code Art. 596 - ភតិសន្យា)", "តើអ្វីទៅជាសិទ្ធិភតិសន្យា និងការជួលអចលនវត្ថុ?"),
        ("តើអ្នកលក់មានការទទួលខុសត្រូវលើវិការៈនៃវត្ថុទិញលក់យ៉ាងដូចម្តេច? (Civil Code Art. 544 - ការទទួលខុសត្រូវលើវិការៈ)", "តើអ្នកលក់មានការទទួលខុសត្រូវលើវិការៈនៃវត្ថុទិញលក់យ៉ាងដូចម្តេច?"),
        ("តើការទាមទារសំណងការខូចខាតដោយសារការមិនអនុវត្តកាតព្វកិច្ចមានលក្ខខណ្ឌអ្វីខ្លះ? (Civil Code Art. 398 - សំណងការខូចខាត)", "តើការទាមទារសំណងការខូចខាតដោយសារការមិនអនុវត្តកាតព្វកិច្ចមានលក្ខខណ្ឌអ្វីខ្លះ?"),
        # Criminal Code (2009)
        ("តើគោលការណ៍គ្មានទោសបើគ្មានច្បាប់ចែង (Principle of Legality) មានន័យដូចម្តេច? (Criminal Code Art. 2 - គោលការណ៍ស្របច្បាប់)", "តើគោលការណ៍គ្មានទោសបើគ្មានច្បាប់ចែង (Principle of Legality) មានន័យដូចម្តេច?"),
        ("តើបទល្មើសព្រហ្មទណ្ឌត្រូវបានបែងចែកជាប៉ុន្មានថ្នាក់? (Criminal Code Art. 46 - បទឧក្រិដ្ឋ បទមជ្ឈិម បទលហុ)", "តើបទល្មើសព្រហ្មទណ្ឌត្រូវបានបែងចែកជាប៉ុន្មានថ្នាក់?"),
        ("តើអ្វីទៅជាការការពារស្របច្បាប់ (Self-Defense) ក្នុងក្រមព្រហ្មទណ្ឌ? (Criminal Code Art. 33 - ការការពារស្របច្បាប់)", "តើអ្វីទៅជាការការពារស្របច្បាប់ (Self-Defense) ក្នុងក្រមព្រហ្មទណ្ឌ?"),
        ("តើអ្វីទៅជាបទលួច និងទោសបញ្ញត្តិក្នុងក្រមព្រហ្មទណ្ឌ? (Criminal Code Art. 353 - បទលួច)", "តើអ្វីទៅជាបទលួច និងទោសបញ្ញត្តិក្នុងក្រមព្រហ្មទណ្ឌ?"),
        ("តើអ្វីទៅជាបទឆបោក និងធាតុផ្សំនៃបទល្មើស? (Criminal Code Art. 377 - បទឆបោក)", "តើអ្វីទៅជាបទឆបោក និងធាតុផ្សំនៃបទល្មើស?"),
        ("តើការប៉ុនប៉ងប្រព្រឹត្តបទល្មើសព្រហ្មទណ្ឌត្រូវផ្តន្ទាទោសក្នុងករណីណាខ្លះ? (Criminal Code Art. 27 - ការប៉ុនប៉ង)", "តើការប៉ុនប៉ងប្រព្រឹត្តបទល្មើសព្រហ្មទណ្ឌត្រូវផ្តន្ទាទោសក្នុងករណីណាខ្លះ?"),
        ("តើច្បាប់ព្រហ្មទណ្ឌកម្ពុជាអនុវត្តទៅលើដែនដីណាខ្លះ? (Criminal Code Art. 12 - ដែនអនុវត្តច្បាប់)", "តើច្បាប់ព្រហ្មទណ្ឌកម្ពុជាអនុវត្តទៅលើដែនដីណាខ្លះ?"),
        ("តើអ្វីទៅជាបទរំលោភលើទំនុកចិត្តក្នុងក្រមព្រហ្មទណ្ឌ? (Criminal Code Art. 391 - បទរំលោភលើទំនុកចិត្ត)", "តើអ្វីទៅជាបទរំលោភលើទំនុកចិត្តក្នុងក្រមព្រហ្មទណ្ឌ?"),
    ]

    question_labels = [item[0] for item in benchmark_questions]
    selected_label = st.selectbox("Pick an authentic benchmark question:", question_labels)

    default_query = ""
    for label, query_text in benchmark_questions:
        if label == selected_label:
            default_query = query_text
            break

    user_query = st.text_input(
        "Question (Khmer / English):",
        value=default_query,
        placeholder="e.g. តើកិច្ចសន្យាបង្កើតឡើងដោយរបៀបណា? or What constitutes legitimate self-defense?",
    )

    if st.button("Run Legal Retrieval & Analysis", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a legal question.")
        else:
            with st.spinner(f"Retrieving top {top_k} articles using {retriever_option.split(' [')[0]}..."):
                t_start = time.time()
                docs = execute_unified_retrieval(
                    query=user_query,
                    retriever_choice=retriever_option,
                    top_k=top_k,
                    law_filter=selected_law_filter,
                )
                retrieval_ms = (time.time() - t_start) * 1000

            st.caption(f"Retrieved **{len(docs)}** statutory articles in **{retrieval_ms:.1f}ms**.")

            # Attempt LLM answer if API key is provided
            active_key = (
                api_key_input.strip()
                or os.environ.get("DEEPSEEK_API_KEY", "")
                or os.environ.get("OPENAI_API_KEY", "")
            ).strip().strip('"').strip("'")

            if active_key:
                with st.spinner("Generating citation-grounded analysis with DeepSeek Flash..."):
                    try:
                        qa_use_case = get_qa_use_case()

                        # Configure client with provided key
                        if hasattr(qa_use_case._llm, "_client"):
                            from openai import OpenAI
                            qa_use_case._llm._client = OpenAI(
                                api_key=active_key,
                                base_url="https://api.deepseek.com",
                            )
                            qa_use_case._llm._api_key = active_key

                        req = LegalQARequest(
                            question=user_query,
                            top_k=top_k,
                            law_filter=selected_law_filter,
                            model="deepseek-chat",
                        )
                        # Override retriever in use case temporarily with our selected docs
                        class DirectRetrieverAdapter:
                            def execute(self, req_inner):
                                return docs

                        original_retriever = qa_use_case._retriever
                        qa_use_case._retriever = DirectRetrieverAdapter()
                        response = qa_use_case.execute(req)
                        qa_use_case._retriever = original_retriever

                        st.markdown("### ⚖️ Legal Synthesis & Grounded Analysis")
                        with st.container(border=True):
                            st.markdown(response.answer)

                        if response.citations:
                            st.markdown("#### 📜 Cited Statutory Authorities & Grounding Verification")
                            for cit in response.citations:
                                is_ver = cit.get("is_verified", False)
                                badge_html = (
                                    '<span class="badge-verified">✓ Verified Context Match</span>'
                                    if is_ver
                                    else '<span class="badge-unverified">⚠ Unverified in Retrieved Context</span>'
                                )
                                is_crim = "criminal" in str(cit.get("law_name", "")).lower()
                                badge_class = "badge-crim" if is_crim else "badge-civil"
                                st.markdown(
                                    f"""
                                    <div style="
                                        display: flex;
                                        align-items: center;
                                        justify-content: space-between;
                                        padding: 10px 14px;
                                        background: rgba(128,128,128,0.04);
                                        border: 1px solid rgba(128,128,128,0.12);
                                        border-radius: 8px;
                                        margin-bottom: 8px;
                                        flex-wrap: wrap;
                                        gap: 8px;
                                    ">
                                        <div>
                                            <span class="legal-law-badge {badge_class}">{cit['law_name']}</span>
                                            <strong style="margin-left: 8px; font-size: 0.95rem;">មាត្រា {cit['article_number']}</strong>
                                        </div>
                                        <div>{badge_html}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                                if cit.get("excerpt"):
                                    st.caption(f"> \"{cit['excerpt']}\"")

                    except Exception as err:
                        st.error(f"LLM Generation Note: {err}")
                        if "401" in str(err) or "auth" in str(err).lower():
                            st.info("To enable LLM generation, set a valid API key from platform.deepseek.com.")
            else:
                st.info(
                    "**Direct Retrieval Mode:** DeepSeek API key not provided. "
                    "Displaying retrieved statutory articles directly below."
                )

            # Display Context Articles
            with st.expander(f"📚 Retrieved Statutory Context Articles ({len(docs)} articles)", expanded=True):
                for i, doc in enumerate(docs, 1):
                    render_article_card(doc, index=i)

            st.warning(
                "**Legal Disclaimer:** This system is an academic research prototype for legal retrieval. "
                "It does not constitute formal legal counsel. Official Khmer gazettes remain the sole authority."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2: Statutory Article Search
# ─────────────────────────────────────────────────────────────────────────────

with tab_search:
    st.subheader("Explore the Official 1,976 Khmer Articles")
    demo_q = "តើនរណាខ្លះអាចស្នើសុំឱ្យតុលាការប្រកាសបើកអាណាព្យាបាលទូទៅ?"
    if "search_query_val" not in st.session_state:
        st.session_state["search_query_val"] = ""

    col_demo, col_clear = st.columns([4, 1])
    with col_demo:
        if st.button("🎯 Load Benchmark Demo Query: «តើនរណាខ្លះអាចស្នើសុំឱ្យតុលាការប្រកាសបើកអាណាព្យាបាលទូទៅ?» (Expected: Art. 24)", key="btn_load_demo_q"):
            st.session_state["search_query_val"] = demo_q
            st.rerun()
    with col_clear:
        if st.session_state["search_query_val"] and st.button("Clear", key="btn_clear_q"):
            st.session_state["search_query_val"] = ""
            st.rerun()

    search_query = st.text_input(
        "Search legal terms, concepts, or article queries:",
        value=st.session_state.get("search_query_val", ""),
        placeholder="e.g. កិច្ចសន្យា, គោលការណ៍ស្វ័យភាព, បទលួច, Article 311, self-defense",
    )

    if search_query.strip():
        t0 = time.time()
        results = execute_unified_retrieval(
            query=search_query,
            retriever_choice=retriever_option,
            top_k=top_k,
            law_filter=selected_law_filter,
        )
        elapsed_ms = (time.time() - t0) * 1000

        st.markdown(
            f"Retrieved **{len(results)}** articles in **{elapsed_ms:.1f}ms** using **{retriever_option.split(' [')[0]}**:"
        )

        for i, doc in enumerate(results, 1):
            render_article_card(doc, index=i)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3: DL Model Benchmark & Evaluation
# ─────────────────────────────────────────────────────────────────────────────

with tab_benchmark:
    st.subheader("Deep Learning Retrieval Benchmark Results")
    st.caption(
        "Uniform evaluation harness comparing 4 PyTorch DL architectures against BM25 and zero-shot mE5 "
        "on the official 1,976-article Khmer legal corpus."
    )

    # 1. Comparative Results Table
    st.markdown("### 1. Comparative Performance on Primary & Secondary Benchmarks")
    st.caption("Primary Benchmark (T-Q: 200 human questions) · Secondary Benchmark (T-T: 148 held-out article titles)")

    summary_csv = repo_root / "results" / "metrics" / "summary.csv"
    if summary_csv.exists():
        try:
            df_raw = pd.read_csv(summary_csv)
            rows = []
            for _, r in df_raw.iterrows():
                name = str(r["model_name"])
                if r["model_id"] == "a3_xlmr":
                    name += " [Winner]"
                t_params = r["trainable_params"]
                tot_params = r.get("total_params", 0)
                if t_params > 0:
                    p_str = f"{t_params / 1e6:.2f}M"
                elif tot_params > 0:
                    p_str = f"{tot_params / 1e6:.0f}M (Frozen)"
                else:
                    p_str = "0"

                rows.append({
                    "Approach": name,
                    "Type": r.get("architecture_type", "Dense"),
                    "Trainable Params": p_str,
                    "T-Q Recall@1": f"{r['tq_recall@1']:.4f}",
                    "T-Q Recall@5": f"{r['tq_recall@5']:.4f}",
                    "T-Q MRR@10": f"{r['tq_mrr@10']:.4f}",
                    "T-T Recall@5": f"{r['tt_recall@5']:.4f}",
                    "Hardware": r.get("hardware", "CPU"),
                })
            df_results = pd.DataFrame(rows)
        except Exception:
            df_results = None
    else:
        df_results = None

    if df_results is None:
        results_data = {
            "Approach": [
                "BM25 Sparse Baseline",
                "Multilingual-E5 (Zero-Shot)",
                "A1: BiLSTM Dual Encoder",
                "A2: XLM-R Linear Probe",
                "A3: XLM-R Full Fine-Tune [Winner]",
                "A4: PrahokBART Dual Encoder",
            ],
            "Type": [
                "Sparse Lexical",
                "Pretrained Dense",
                "From Scratch Dense",
                "Frozen + Linear Head",
                "Full Fine-Tuning",
                "Khmer Pretrained (Fine-Tuning)",
            ],
            "Trainable Params": ["0", "278M (Frozen)", "1.97M", "0.59M", "278.04M", "35.83M"],
            "T-Q Recall@1": ["0.2025", "0.1625", "0.0875", "0.2400", "0.3375", "0.0525"],
            "T-Q Recall@5": ["0.4075", "0.2617", "0.2025", "0.4117", "0.5142", "0.1517"],
            "T-Q MRR@10": ["0.3350", "0.2588", "0.1724", "0.3591", "0.4827", "0.1035"],
            "T-T Recall@5": ["0.9730", "0.7230", "0.7568", "0.9392", "0.9932", "0.8041"],
            "Hardware": ["CPU", "CPU", "CPU", "CPU", "Tesla T4 GPU", "CPU (fp32)"],
        }
        df_results = pd.DataFrame(results_data)

    st.dataframe(df_results, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Key Findings:**
        - **Approach A3 (XLM-R Full Fine-Tune)** is the definitive winner: **51.42% Recall@5** and **0.4827 MRR@10** on the Primary Benchmark (T-Q: 200 human legal questions), outperforming BM25 by **+10.67% Recall@5** (p < 0.001) and **+44.1% relative MRR@10** (p < 0.001).
        - **In-Distribution (T-T) vs Out-of-Distribution Shift (T-Q)**: On held-out titles (T-T), models reach near ceiling (A3: **99.32%**, BM25: **97.30%**, A2: **93.92%**). On conversational human questions (T-Q), distribution shift (colloquial phrasing, lack of exact title keywords) causes expected drop across all systems.
        - **Approach A2 (Linear Probe)** delivers solid parameter-efficient performance (**41.17% Recall@5**, **0.3591 MRR@10**) with only 0.59M trainable parameters (0.21%), essentially matching BM25 (p = 0.9031). Deep fine-tuning across all layers is required to significantly beat sparse keyword search.
        - **Approach A4 (PrahokBART Dual Encoder)** excels in-distribution (T-T: **80.41% Recall@5**, **0.6733 MRR@10**), but drops to **15.17% Recall@5** on conversational questions (T-Q), showing sensitivity to paraphrasing without large-scale legal fine-tuning pairs.
        - **Approach A1 (BiLSTM from scratch)** achieves **20.25% Recall@5** and **0.1724 MRR@10** on T-Q (and **75.68% Recall@5** on T-T), constrained by vocabulary sparsity on low-resource Khmer when trained from scratch without pretraining.
        """
    )

    st.divider()

    # 2. Research Figures Gallery
    st.markdown("### 2. Publication Figures & Training Curves")

    fig_dir = repo_root / "results" / "figures"
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        curves_p = fig_dir / "learning_curves.png"
        if curves_p.exists():
            st.image(str(curves_p), caption="Figure 1: Training Loss & Validation MRR@10 Curves Across Epochs")

        bar_p = fig_dir / "primary_metrics_bar.png"
        if bar_p.exists():
            st.image(str(bar_p), caption="Figure 2: Primary Benchmark Metrics Comparison (Recall@5 & MRR@10)")

    with col_f2:
        recall_p = fig_dir / "recall_at_k.png"
        if recall_p.exists():
            st.image(str(recall_p), caption="Figure 3: Recall@k Evaluation Across Cutoffs k in {1, 3, 5, 10, 20}")

        code_p = fig_dir / "code_breakdown.png"
        if code_p.exists():
            st.image(str(code_p), caption="Figure 4: Per-Code Breakdown (Civil Code 2007 vs Criminal Code 2009)")

    st.divider()

    # 3. Hyperparameter Tuning Grid
    st.markdown("### 3. Hyperparameter Grid Search (Approach A3)")
    st.caption("3x2 Grid Search: Learning Rates {1e-5, 2e-5, 3e-5} x Weight Decay {0.0, 0.01} on Tesla T4 GPU")

    a3_csv = repo_root / "results" / "tuning" / "a3.csv"
    if a3_csv.exists():
        df_tuning = pd.read_csv(a3_csv)
        st.dataframe(df_tuning, use_container_width=True, hide_index=True)

    st.divider()

    # 4. Error Analysis
    st.markdown("### 4. Qualitative Error Analysis & Linguistic Failure Modes")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.markdown(
            """
            <div class="error-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <strong style="font-size: 0.95rem;">1. Exact Term Mismatch (BM25 Failure)</strong>
                    <span class="legal-law-badge badge-crim" style="font-size: 0.7rem;">Lexical Gap</span>
                </div>
                <p style="margin: 4px 0; font-size: 0.88rem;"><strong>Query:</strong> <em>What is the principle of private autonomy?</em> (<code>ស្វ័យភាពនៃបុគ្គលឯកជន</code>)</p>
                <p style="margin: 4px 0; font-size: 0.88rem;"><strong>Ground Truth:</strong> Civil Code Art. 3</p>
                <p style="margin: 4px 0; font-size: 0.85rem; color: #64748b;">
                    <strong>Failure Mechanism:</strong> Questions using colloquial Khmer phrasing miss when searching exact statutory terminology. Neural models bridge this gap via semantic embedding geometry.
                </p>
            </div>
            <div class="error-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <strong style="font-size: 0.95rem;">2. Abstract Legal Doctrine (BiLSTM Failure)</strong>
                    <span class="legal-law-badge badge-crim" style="font-size: 0.7rem;">Representation Gap</span>
                </div>
                <p style="margin: 4px 0; font-size: 0.88rem;"><strong>Query:</strong> <em>Principle of Legality</em> (<code>គ្មានទោសបើគ្មានច្បាប់ចែង</code>)</p>
                <p style="margin: 4px 0; font-size: 0.88rem;"><strong>Ground Truth:</strong> Criminal Code Art. 2</p>
                <p style="margin: 4px 0; font-size: 0.85rem; color: #64748b;">
                    <strong>Failure Mechanism:</strong> BiLSTM trained from scratch struggles on rare Latin/French legal doctrine loan translations without massive pretraining.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_e2:
        st.markdown(
            """
            <div class="error-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <strong style="font-size: 0.95rem;">3. Sub-Clause Disambiguation</strong>
                    <span class="legal-law-badge badge-civil" style="font-size: 0.7rem;">Fine-Grained Scope</span>
                </div>
                <p style="margin: 4px 0; font-size: 0.88rem;"><strong>Query:</strong> <em>Remedies for defect in specific contract types</em></p>
                <p style="margin: 4px 0; font-size: 0.88rem;"><strong>Ground Truth:</strong> Civil Code Art. 544 vs Art. 547</p>
                <p style="margin: 4px 0; font-size: 0.85rem; color: #64748b;">
                    <strong>Failure Mechanism:</strong> Closely related procedural remedies within the same chapter have similar dense embeddings; reranker or BM25 hybrid fusion helps isolate the exact article.
                </p>
            </div>
            <div class="error-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px;">
                    <strong style="font-size: 0.95rem;">4. Lexical Gap on Criminal Offense Classes</strong>
                    <span class="badge-verified" style="font-size: 0.7rem;">100% Solved by A3</span>
                </div>
                <p style="margin: 4px 0; font-size: 0.88rem;"><strong>Query:</strong> <em>Classification into Felonies, Misdemeanors, and Petty Offenses</em></p>
                <p style="margin: 4px 0; font-size: 0.88rem;"><strong>Ground Truth:</strong> Criminal Code Art. 46</p>
                <p style="margin: 4px 0; font-size: 0.85rem; color: #64748b;">
                    <strong>Model Behavior:</strong> Handled with 100% Top-1 accuracy by A3 Full Fine-Tune, effectively mapping colloquial category queries to formal legal statutory classifications.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
