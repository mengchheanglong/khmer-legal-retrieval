"""
Streamlit Web UI for Khmer Legal Article Retrieval Assistant.

Individual Deep Learning Final Project — Bachelor of Software Engineering
Lecturer: Mr. Soklong HIM · Academic Year 2026-2027
Author: Long Mengchheang (@mengchheanglong)

Features:
- Multi-Retriever Architecture (A3 XLM-R Full Fine-Tune, Hybrid, BM25)
- 1,976 Official Khmer Statutory Articles (Civil Code 2007 + Criminal Code 2009)
- Citation-Grounded Legal Q&A powered by DeepSeek Flash
- Interactive Deep Learning Benchmark Dashboard & Publication Figures
"""

import html
import json
import os
import sys
import time
from pathlib import Path

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

import pandas as pd
import streamlit as st

# Sync Streamlit Cloud secrets into environment if present
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str):
                os.environ[k] = v.strip().strip('"').strip("'")
except Exception:
    pass

from src.application.dtos import LegalQARequest, RetrievalRequest
from src.domain.entities import RetrievedDocument
from src.application.use_cases.answer_legal_qa import AnswerLegalQAUseCase
from src.infrastructure.ai.openai_llm import OpenAILLM


# ─────────────────────────────────────────────────────────────────────────────
# Cached Model & Index Loaders
# ─────────────────────────────────────────────────────────────────────────────

from src.interfaces.ui.retrieval import execute_unified_retrieval


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
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

    /* Base Typography — Target application containers; NEVER use [class*="st-"] which breaks icon ligatures */
    .stApp,
    .stApp p,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp h5,
    .stApp h6,
    .stApp label,
    .stApp input,
    .stApp textarea,
    .stApp select {
        font-family: 'Inter', 'Kantumruy Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* PRESERVE STREAMLIT NATIVE MATERIAL SYMBOLS & ICONS (fixes arrow_down, keyboard_double, visibility text) */
    [data-testid="stIconMaterial"],
    .material-symbols-rounded,
    .material-symbols-outlined,
    .material-icons,
    [data-testid="stExpanderToggleIcon"],
    [data-testid="stExpanderToggleIcon"] *,
    button[aria-label*="password"] *,
    button[data-testid="baseButton-headerNoPadding"] *,
    [data-testid="stSidebarCollapseButton"] *,
    span[class*="material-symbols"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        font-weight: normal !important;
        font-style: normal !important;
        font-size: 20px !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        display: inline-block !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        -webkit-font-feature-settings: 'liga' !important;
        -webkit-font-smoothing: antialiased !important;
        font-feature-settings: 'liga' 1 !important;
    }

    /* Expander visual refinement */
    [data-testid="stExpander"] {
        border-radius: 8px !important;
        border: 1px solid rgba(128, 128, 128, 0.22) !important;
        overflow: hidden;
        margin-bottom: 12px;
    }
    [data-testid="stExpander"] summary {
        font-weight: 600;
        padding: 10px 14px !important;
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


def render_article_card(doc: RetrievedDocument, index: int | None = None) -> None:
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
        delta="Recall@5 (+10.7 pp vs BM25)",
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
        delta="deepseek-chat",
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
            "A3: XLM-R Fine-Tuned (Best)",
            "Hybrid: A3 Dense + BM25 Sparse",
            "BM25 Sparse (khmer-nltk)",
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
    api_key_input = ""
    if env_key:
        st.success("API key loaded from environment", icon="🔑")
        with st.expander("⚙️ Change / Override Key"):
            override_val = st.text_input(
                "Custom DeepSeek Key:",
                type="password",
                placeholder="sk-...",
                help="Leave empty to use the active key from .env",
                key="override_deepseek_key",
            )
            if override_val.strip():
                api_key_input = override_val.strip()
    else:
        api_key_input = st.text_input(
            "DeepSeek API Key:",
            type="password",
            placeholder="sk-... (or add DEEPSEEK_API_KEY to .env)",
            help="Required for Tab 1 Q&A synthesis. Direct Search in Tab 2 is 100% offline.",
            key="direct_deepseek_key",
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

    benchmark_path = repo_root / "tests" / "evaluation" / "ground_truth_qa_kh.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    benchmark.sort(key=lambda item: item["id"] != "tq-civ-009")
    benchmark_questions = [("-- Select a benchmark question --", "")]
    benchmark_questions.extend(
        (f"{item['id']} | {item['question_kh']} "
         f"({item['expected_law']}, Art. {', '.join(map(str, item['expected_articles']))})",
         item["question_kh"])
        for item in benchmark
    )

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

            if active_key and docs:
                with st.spinner("Generating citation-grounded analysis with DeepSeek Flash..."):
                    try:
                        class DirectRetrieverAdapter:
                            """Reuse the selected results without loading another model."""

                            def execute(self, request: RetrievalRequest) -> list[RetrievedDocument]:
                                """Return the already retrieved context."""
                                return docs

                        qa_use_case = AnswerLegalQAUseCase(
                            retriever=DirectRetrieverAdapter(),
                            llm=OpenAILLM(api_key=active_key, model="deepseek-chat"),
                        )

                        req = LegalQARequest(
                            question=user_query,
                            top_k=top_k,
                            law_filter=selected_law_filter,
                            model="deepseek-chat",
                        )
                        response = qa_use_case.execute(req)
                        st.caption(f"Answer source: {response.model_used}")
                        st.caption("Citation checks confirm article identity in the context; they do not establish that every claim is correct.")

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

    st.dataframe(df_results, width="stretch", hide_index=True)

    st.markdown(
        """
        **Key Findings:**
        - **Approach A3 (XLM-R Full Fine-Tune)** is the definitive winner: **51.42% Recall@5** and **0.4827 MRR@10** on the Primary Benchmark (T-Q: 200 human legal questions), outperforming BM25 by **+10.67 percentage points Recall@5** (p < 0.001) and **+44.1% relative MRR@10** (p < 0.001).
        - **In-Distribution (T-T) vs Out-of-Distribution Shift (T-Q)**: On held-out titles (T-T), models reach near ceiling (A3: **99.32%**, BM25: **97.30%**, A2: **93.92%**). On conversational human questions (T-Q), distribution shift (colloquial phrasing, lack of exact title keywords) causes expected drop across all systems.
        - **Approach A2 (Linear Probe)** delivers solid parameter-efficient performance (**41.17% Recall@5**, **0.3591 MRR@10**) with only 0.59M trainable parameters (0.21%), essentially matching BM25 (p = 0.9031). A3 significantly outperformed sparse search in this experiment.
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
        st.dataframe(df_tuning, width="stretch", hide_index=True)

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
