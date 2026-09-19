"""
Detailed error analysis and diagnostic categorization for Khmer legal retrieval models.

Investigates failure cases on the Primary Benchmark (T-Q: 200 questions) at k=5:
1. Categorizes misses:
   - Vocabulary Mismatch: colloquial Khmer synonyms absent from formal legal statutes.
   - Code Confusion: retrieved Criminal articles for Civil questions (or vice versa).
   - Near-Miss: relevant article ranked at positions 6-10.
   - Shared Title Ambiguity: articles sharing generic headings (e.g. 'ការប៉ុនប៉ង').
   - Multi-Article Complexity: questions citing multiple interrelated articles.
2. Extracts concrete worked examples for presentation slides.
3. Exports results to results/error_analysis/.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch

from src.config.logging import get_logger, setup_logging
from src.dl.checkpoint import load_checkpoint
from src.dl.data import CorpusDataset, QABenchmarkDataset
from src.dl.evaluate import BM25KhmerRetriever, DenseRetriever
from src.dl.models.bilstm import BiLSTMEncoder
from src.dl.models.linear_probe import XLMRLinearProbeRetriever
from src.dl.models.vocab import KhmerVocab
from src.dl.models.xlmr_finetune import XLMRFullFinetuneRetriever
from src.dl.seed import get_device, set_seed

setup_logging()
logger = get_logger(__name__)


def categorize_error(
    question: dict[str, Any],
    retrieved_doc_ids: list[str],
    corpus: CorpusDataset,
) -> str:
    """Classify a retrieval failure into an analytical error mode."""
    expected_doc_ids = set(question["expected_doc_ids"])
    top_5 = retrieved_doc_ids[:5]
    top_10 = retrieved_doc_ids[:10]

    # Check near miss (rank 6 to 10)
    if any(doc in top_10 for doc in expected_doc_ids):
        return "Near-Miss (Rank 6-10)"

    # Check code confusion (Civil vs Criminal)
    expected_law = question["expected_law"]
    top_laws = []
    for doc in top_5[:3]:
        try:
            top_laws.append(corpus.get_by_doc_id(doc).get("law_name", ""))
        except KeyError:
            pass

    wrong_code_count = sum(1 for law in top_laws if law != expected_law)
    if wrong_code_count >= 2:
        return "Civil/Criminal Code Confusion"

    # Check multi-article complexity
    if len(expected_doc_ids) > 1:
        return "Multi-Article Complexity"

    # Check shared title ambiguity
    top_doc = corpus.get_by_doc_id(top_5[0]) if top_5 else None
    if top_doc and top_doc.get("title") in ["ការប៉ុនប៉ង", "និយមន័យ", "ទោសបន្ថែម", "អត្ថន័យ"]:
        return "Shared Title Ambiguity"

    return "Vocabulary Mismatch / Paraphrasing"


def get_rankings_with_cache(
    model_id: str,
    retriever: Any,
    queries: list[str],
    cache_dir: Path,
) -> list[list[str]]:
    """Retrieve rankings or load from disk cache or metrics JSON."""
    cache_file = cache_dir / f"{model_id}_rankings.json"
    if cache_file.exists():
        logger.info(f"Loading cached rankings for '{model_id}' from {cache_file}...")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # Check if pre-computed rankings exist in results/metrics/<model_id>.json
    metric_candidates = [
        cache_dir.parent / "metrics" / f"{model_id}.json",
        cache_dir.parent / "metrics" / f"{model_id}_baseline.json",
    ]
    for m_cand in metric_candidates:
        if m_cand.exists():
            try:
                with open(m_cand, "r", encoding="utf-8") as f:
                    m_data = json.load(f)
                rankings = m_data.get("tq_benchmark", {}).get("rankings")
                if rankings and len(rankings) == len(queries):
                    logger.info(f"Loading pre-computed rankings for '{model_id}' from {m_cand}...")
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(rankings, f, ensure_ascii=False)
                    return rankings
            except Exception as e:
                logger.warning(f"Could not load rankings from {m_cand}: {e}")

    if retriever is None:
        raise ValueError(f"No cached rankings found for '{model_id}' and retriever is None")

    logger.info(f"Computing rankings for '{model_id}'...")
    rankings = retriever.search(queries, top_k=10)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(rankings, f, ensure_ascii=False)
    logger.info(f"Saved rankings cache -> {cache_file}")
    return rankings


def analyze_model_rankings(
    model_name: str,
    all_rankings: list[list[str]],
    qa_ds: QABenchmarkDataset,
    corpus: CorpusDataset,
) -> dict[str, Any]:
    """Analyze misses at k=5 for pre-computed model rankings."""
    misses: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {
        "Near-Miss (Rank 6-10)": 0,
        "Civil/Criminal Code Confusion": 0,
        "Multi-Article Complexity": 0,
        "Shared Title Ambiguity": 0,
        "Vocabulary Mismatch / Paraphrasing": 0,
    }

    for i, item in enumerate(qa_ds.records):
        retrieved = all_rankings[i]
        expected = set(item["expected_doc_ids"])
        hit_at_5 = any(doc in retrieved[:5] for doc in expected)

        if not hit_at_5:
            cat = categorize_error(item, retrieved, corpus)
            category_counts[cat] += 1

            misses.append({
                "question_id": item["id"],
                "question_kh": item["question"],
                "expected_law": item["expected_law"],
                "expected_articles": item["expected_articles"],
                "expected_doc_ids": list(expected),
                "retrieved_top5": retrieved[:5],
                "error_category": cat,
                "first_relevant_rank": next((idx + 1 for idx, doc in enumerate(retrieved) if doc in expected), None),
            })

    total_questions = len(qa_ds)
    total_misses = len(misses)
    hit_5_rate = (total_questions - total_misses) / total_questions

    return {
        "model_name": model_name,
        "total_questions": total_questions,
        "total_misses_at_5": total_misses,
        "hit_at_5_rate": hit_5_rate,
        "category_counts": category_counts,
        "misses": misses,
    }


def _art_label(corpus: "CorpusDataset", doc_id: str) -> str:
    """'Civil Code Art. 350 (title)' style label, looked up from the real corpus."""
    try:
        rec = corpus.get_by_doc_id(doc_id)
    except KeyError:
        return doc_id
    law = rec.get("law_name", "")
    num = rec.get("article_number", "")
    title = rec.get("title") or rec.get("article_title") or ""
    return f"{law} Art. {num} ({title})" if title else f"{law} Art. {num}"


def generate_worked_examples(
    output_path: Path,
    qa_ds: "QABenchmarkDataset",
    corpus: "CorpusDataset",
    rankings: dict[str, list[list[str]]],
) -> None:
    """
    Reconstruct concrete worked examples directly from ground truth + real per-query rankings.

    ``rankings`` maps a model key ("bm25", "a1", "a2", "a3", "a4") to its list of per-query top-10
    doc_id rankings, in the same order as ``qa_ds.records``. Only keys that are present are used, so
    this degrades gracefully when only a subset of models was analyzed (the CLI default used to be
    ``--models bm25,a1``, which produced no A3-based examples at all). Every article cited below is
    looked up from ``corpus`` at generation time, so a wrong statute number cannot silently persist
    across a regeneration.
    """
    records = qa_ds.records
    md: list[str] = [
        "# Qualitative Error Analysis & Worked Examples\n",
        "This document details representative retrieval cases across distinct error categories on "
        "the Primary Benchmark (T-Q: 200 human-verified questions).\n",
        "**Provenance:** generated by `generate_worked_examples()` in `src/dl/error_analysis.py` "
        "directly from `tests/evaluation/ground_truth_qa_kh.json` and the cached per-query rankings "
        "in `results/error_analysis/*_rankings.json`. Every article number and title is looked up "
        "from the corpus at generation time, not hand-typed.\n",
    ]
    example_no = 0

    # 1. Vocabulary mismatch: a lexical baseline (bm25) misses a single-article question that a
    #    trained dense model (prefer a3, else a2, else a1) hits at rank 1.
    if "bm25" in rankings:
        for dense_key in ("a3", "a2", "a1"):
            if dense_key not in rankings:
                continue
            found = False
            for i, item in enumerate(records):
                expected = set(item["expected_doc_ids"])
                if len(expected) != 1:
                    continue
                bm25_top10 = rankings["bm25"][i]
                dense_top10 = rankings[dense_key][i]
                if not any(d in bm25_top10 for d in expected) and dense_top10 and dense_top10[0] in expected:
                    example_no += 1
                    md += [
                        f"### Example {example_no} -- Vocabulary Mismatch (`{item['id']}`)",
                        f"- **Question**: `{item['question']}`",
                        f"- **Expected**: {_art_label(corpus, next(iter(expected)))}",
                        "- **BM25 (lexical)**: **Miss** -- not in its top 10 at all.",
                        f"- **{dense_key.upper()} (dense)**: **Rank 1 (hit)**.",
                        "- **Why**: the question does not reuse the statute's exact wording, so lexical "
                        "term-matching cannot lock onto it; the dense encoder still places the question "
                        "near the article's meaning.\n",
                    ]
                    found = True
                    break
            if found:
                break

    # 2. Exact statutory wording: both bm25 and the best dense model available hit rank 1 -- shows
    #    BM25 is not a weak baseline when the question quotes the statute almost verbatim.
    if "bm25" in rankings:
        for dense_key in ("a3", "a2", "a1"):
            if dense_key not in rankings:
                continue
            found = False
            for i, item in enumerate(records):
                expected = set(item["expected_doc_ids"])
                if len(expected) != 1:
                    continue
                bm25_top10 = rankings["bm25"][i]
                dense_top10 = rankings[dense_key][i]
                if bm25_top10 and bm25_top10[0] in expected and dense_top10 and dense_top10[0] in expected:
                    example_no += 1
                    md += [
                        f"### Example {example_no} -- Exact Term Match Favours BM25 Too (`{item['id']}`)",
                        f"- **Question**: `{item['question']}`",
                        f"- **Expected**: {_art_label(corpus, next(iter(expected)))}",
                        "- **BM25**: **Rank 1**.",
                        f"- **{dense_key.upper()}**: **Rank 1** as well.",
                        "- **Why it matters**: when a question already uses the statute's own wording, "
                        "lexical search is not a weak baseline -- the gap between BM25 and a fine-tuned "
                        "model shows up on paraphrased, conversational questions, not on quoted ones.\n",
                    ]
                    found = True
                    break
            if found:
                break

    # 3. Multi-article question: a dense model catches >=2 of the expected articles in its top 5,
    #    while bm25 catches fewer.
    for dense_key in ("a3", "a2", "a1"):
        if dense_key not in rankings:
            continue
        found = False
        for i, item in enumerate(records):
            expected = item["expected_doc_ids"]
            if len(expected) < 2:
                continue
            dense_top5 = rankings[dense_key][i][:5]
            dense_hits = [d for d in expected if d in dense_top5]
            if len(dense_hits) < 2:
                continue
            bm25_hits = [d for d in expected if "bm25" in rankings and d in rankings["bm25"][i][:5]]
            if len(bm25_hits) >= len(dense_hits):
                continue
            example_no += 1
            expected_labels = ", ".join(_art_label(corpus, d) for d in expected)
            ranks = ", ".join(str(dense_top5.index(d) + 1) for d in dense_hits)
            md += [
                f"### Example {example_no} -- Multi-Article Question (`{item['id']}`)",
                f"- **Question**: `{item['question']}`",
                f"- **Expected**: {expected_labels}",
                f"- **BM25 top 5**: catches {len(bm25_hits)} of the {len(expected)} expected articles.",
                f"- **{dense_key.upper()} top 5**: catches {len(dense_hits)} of {len(expected)}, at rank(s) {ranks}.",
                "- **Why**: the question genuinely asks about more than one article; a single top-1 "
                "metric understates cases like this, which is why Recall@5 rather than Recall@1 is "
                "the primary metric.\n",
            ]
            found = True
            break
        if found:
            break

    # 4. Civil/Criminal Code confusion: a weaker model's top-1 guess is from the wrong statute while
    #    a stronger dense model gets the right one.
    if "a1" in rankings:
        for dense_key in ("a3", "a2"):
            if dense_key not in rankings:
                continue
            found = False
            for i, item in enumerate(records):
                expected = set(item["expected_doc_ids"])
                if len(expected) != 1:
                    continue
                expected_law = item["expected_law"]
                weak_top1 = rankings["a1"][i][0] if rankings["a1"][i] else None
                dense_top1 = rankings[dense_key][i][0] if rankings[dense_key][i] else None
                if weak_top1 is None or dense_top1 is None:
                    continue
                try:
                    weak_law = corpus.get_by_doc_id(weak_top1).get("law_name", "")
                except KeyError:
                    continue
                if weak_law == expected_law or dense_top1 not in expected:
                    continue
                example_no += 1
                md += [
                    f"### Example {example_no} -- Civil/Criminal Code Confusion (`{item['id']}`)",
                    f"- **Question**: `{item['question']}`",
                    f"- **Expected**: {_art_label(corpus, next(iter(expected)))}",
                    f"- **A1 (BiLSTM, from scratch)**: top-1 guess is {_art_label(corpus, weak_top1)} -- "
                    "**wrong statute entirely**.",
                    f"- **{dense_key.upper()}**: top-1 is {_art_label(corpus, dense_top1)} -- **correct**.",
                    "- **Why**: a from-scratch encoder trained on ~1,176 pairs has not learned a "
                    "reliable boundary between the two codes' vocabularies; the pre-trained backbone "
                    "has (see `results/error_analysis/error_breakdown.csv` for the full 200-question "
                    "count of this error mode per model).\n",
                ]
                found = True
                break
            if found:
                break

    if example_no == 0:
        md.append(
            "_No worked examples could be reconstructed: the rankings needed (at minimum BM25 plus "
            "one dense model) were not available. Re-run `python -m src.dl.error_analysis --models all` "
            "to compute all five rankings first._\n"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info(f"Saved worked examples -> {output_path}")


def run_error_analysis(
    results_dir: Path = Path("results"),
    splits_dir: Path = Path("data/05_splits"),
    models_to_run: str = "bm25,a1",
    seed: int = 42,
) -> Path:
    """Execute error analysis across selected approaches."""
    set_seed(seed)
    device = get_device()

    out_dir = results_dir / "error_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_path = splits_dir.parent.parent / "tests" / "evaluation" / "ground_truth_qa_kh.json"
    corpus_path = splits_dir / "corpus.jsonl"

    qa_ds = QABenchmarkDataset(qa_path)
    corpus = CorpusDataset(corpus_path)
    queries = qa_ds.questions

    selected_models = [m.strip().lower() for m in models_to_run.split(",")]
    analyzed_results = []
    bm25_rankings: Optional[list] = None
    a1_rankings: Optional[list] = None
    a2_rankings: Optional[list] = None
    a3_rankings: Optional[list] = None
    a4_rankings: Optional[list] = None

    # 1. BM25
    if "bm25" in selected_models or "all" in selected_models:
        bm25 = BM25KhmerRetriever(corpus)
        bm25_rankings = get_rankings_with_cache("bm25", bm25, queries, out_dir)
        bm25_res = analyze_model_rankings("BM25 Baseline", bm25_rankings, qa_ds, corpus)
        analyzed_results.append(bm25_res)

    # 2. A1 BiLSTM
    if "a1" in selected_models or "all" in selected_models:
        vocab_path = splits_dir / "vocab_kh.json"
        vocab = KhmerVocab.load(vocab_path) if vocab_path.exists() else KhmerVocab.build_from_train_file(splits_dir / "train.jsonl")
        a1_model = BiLSTMEncoder(vocab=vocab)
        a1_ckpt = results_dir / "checkpoints" / "a1_bilstm" / "best.pt"
        if a1_ckpt.exists():
            load_checkpoint(a1_ckpt, model=a1_model, device=device)
            a1_retriever = DenseRetriever(a1_model, corpus, device=device, batch_size=32)
            a1_rankings = get_rankings_with_cache("a1_bilstm", a1_retriever, queries, out_dir)
            a1_res = analyze_model_rankings("A1: BiLSTM", a1_rankings, qa_ds, corpus)
            analyzed_results.append(a1_res)

    # 3. A2 Linear Probe
    if "a2" in selected_models or "all" in selected_models:
        a2_ckpt = results_dir / "checkpoints" / "a2_linear_probe" / "best.pt"
        if a2_ckpt.exists():
            a2_model = XLMRLinearProbeRetriever(device=device)
            load_checkpoint(a2_ckpt, model=a2_model, device=device)
            a2_retriever = DenseRetriever(a2_model, corpus, device=device, batch_size=32)
            a2_rankings = get_rankings_with_cache("a2_linear_probe", a2_retriever, queries, out_dir)
            a2_res = analyze_model_rankings("A2: Linear Probe", a2_rankings, qa_ds, corpus)
            analyzed_results.append(a2_res)

    # 4. A3 Full Fine-Tuning
    if "a3" in selected_models or "all" in selected_models:
        a3_ckpt = results_dir / "checkpoints" / "a3_xlmr_finetune" / "best.pt"
        if a3_ckpt.exists():
            a3_model = XLMRFullFinetuneRetriever(device=device)
            load_checkpoint(a3_ckpt, model=a3_model, device=device)
            a3_retriever = DenseRetriever(a3_model, corpus, device=device, batch_size=32)
            a3_rankings = get_rankings_with_cache("a3_xlmr", a3_retriever, queries, out_dir)
            a3_res = analyze_model_rankings("A3: Full Fine-Tuning", a3_rankings, qa_ds, corpus)
            analyzed_results.append(a3_res)

    # 5. A4 PrahokBART Khmer Native
    if "a4" in selected_models or "all" in selected_models:
        a4_ckpt = results_dir / "checkpoints" / "a4_prahokbart" / "best.pt"
        if a4_ckpt.exists():
            from src.dl.models.prahokbart import PrahokBARTDualEncoder

            a4_model = PrahokBARTDualEncoder(device=device)
            load_checkpoint(a4_ckpt, model=a4_model, device=device)
            a4_retriever = DenseRetriever(a4_model, corpus, device=device, batch_size=32)
            a4_rankings = get_rankings_with_cache("a4_prahokbart", a4_retriever, queries, out_dir)
            a4_res = analyze_model_rankings("A4: PrahokBART", a4_rankings, qa_ds, corpus)
            analyzed_results.append(a4_res)

    # Save summary table
    rows = []
    for res in analyzed_results:
        row = {
            "model_name": res["model_name"],
            "total_misses_at_5": res["total_misses_at_5"],
            "hit_at_5_rate": f"{res['hit_at_5_rate'] * 100:.1f}%",
        }
        row.update(res["category_counts"])
        rows.append(row)

    df_errors = pd.DataFrame(rows)
    err_csv = out_dir / "error_breakdown.csv"
    df_errors.to_csv(err_csv, index=False)
    logger.info(f"Saved error breakdown table -> {err_csv}")

    # Generate worked examples markdown
    available_rankings = {
        key: val for key, val in {
            "bm25": bm25_rankings, "a1": a1_rankings, "a2": a2_rankings,
            "a3": a3_rankings, "a4": a4_rankings,
        }.items() if val is not None
    }
    generate_worked_examples(out_dir / "worked_examples.md", qa_ds, corpus, available_rankings)

    print("\n" + "=" * 80)
    print("PRIMARY BENCHMARK (T-Q) ERROR CATEGORIZATION AT k=5")
    print("=" * 80)
    print(df_errors.to_string(index=False))
    print("=" * 80 + "\n")

    return err_csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze retrieval errors on primary benchmark.")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated models to analyze (bm25, a1, a2, a3, a4, all). Default: all (needed for the A3-based worked examples in worked_examples.md).",
    )
    args = parser.parse_args()
    run_error_analysis(models_to_run=args.models)
