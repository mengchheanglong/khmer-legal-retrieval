"""
Dataset split preparation for Khmer legal retrieval experiments.

Prepares reproducible dataset splits from chunked Khmer legal articles:
1. Builds a unified 1,976-article corpus (data/05_splits/corpus.jsonl).
2. Implements a strict leakage guard against the ~200 question benchmark (T-Q).
3. Selects the 1,706 unique-title articles and strips the title line from passages.
4. Splits eligible (query=title, passage=body) pairs 80/10/10 with fixed seed 42 into:
   - train.jsonl (training set)
   - val.jsonl (validation set for hyperparameter tuning & model selection)
   - test_titles.jsonl (T-T test set of held-out article titles)
5. Generates manifest.json with counts, distributions, and SHA-256 hashes.
"""

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

# Constants
TOTAL_CORPUS_COUNT = 1976
CIVIL_CODE_COUNT = 1304
CRIMINAL_CODE_COUNT = 672
EXPECTED_UNIQUE_TITLES = 1706


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file.

    Args:
        file_path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def generate_doc_id(law_name: str, article_number: int) -> str:
    """Generate a clean, standardized document identifier.

    Args:
        law_name: Name of the law (e.g. 'Civil Code 2007').
        article_number: Sequential article number (e.g. 1).

    Returns:
        Standardized identifier string like 'civil_code_kh_art_1'.
    """
    if "Civil" in law_name:
        prefix = "civil_code_kh"
    elif "Criminal" in law_name:
        prefix = "criminal_code_kh"
    else:
        prefix = law_name.lower().replace(" ", "_")
    return f"{prefix}_art_{article_number}"


def strip_title_line(content: str) -> str:
    """Strip the leading article title line from chunk content.

    In Cambodian legal chunks, the first line is formatted as:
        មាត្រា N.- <article_title>
    Following lines contain the substantive legal body. Stripping the title
    line prevents models from taking a trivial shortcut during retrieval training
    by matching identical title strings.

    Args:
        content: Raw chunk content containing the header line and body.

    Returns:
        The article body with the title line removed.

    Raises:
        ValueError: If the header line does not start with 'មាត្រា' or the
            remaining body text is empty.
    """
    parts = content.split("\n", 1)
    if not parts[0].strip().startswith("មាត្រា"):
        raise ValueError(
            f"Chunk does not begin with an article header: {parts[0][:50]!r}"
        )
    if len(parts) < 2 or not parts[1].strip():
        raise ValueError(
            f"Chunk body is empty after stripping title line: {parts[0][:50]!r}"
        )
    return parts[1].strip()


def load_corpus_chunks(chunks_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load Civil and Criminal Code Khmer chunks from disk.

    Args:
        chunks_dir: Directory containing chunk JSON files.

    Returns:
        Tuple of (civil_chunks, criminal_chunks).

    Raises:
        FileNotFoundError: If expected chunk files are missing.
        ValueError: If article counts do not match expected legal counts.
    """
    civil_file = chunks_dir / "civil_code_2007_kh_chunks.json"
    crim_file = chunks_dir / "criminal_code_2009_kh_chunks.json"

    if not civil_file.exists():
        raise FileNotFoundError(f"Civil Code chunks not found: {civil_file}")
    if not crim_file.exists():
        raise FileNotFoundError(f"Criminal Code chunks not found: {crim_file}")

    with open(civil_file, "r", encoding="utf-8") as f:
        civil_chunks = json.load(f)
    with open(crim_file, "r", encoding="utf-8") as f:
        crim_chunks = json.load(f)

    if len(civil_chunks) != CIVIL_CODE_COUNT:
        raise ValueError(
            f"Expected {CIVIL_CODE_COUNT} Civil Code chunks, found {len(civil_chunks)}"
        )
    if len(crim_chunks) != CRIMINAL_CODE_COUNT:
        raise ValueError(
            f"Expected {CRIMINAL_CODE_COUNT} Criminal Code chunks, found {len(crim_chunks)}"
        )

    return civil_chunks, crim_chunks


def load_protected_articles(qa_path: Path) -> set[tuple[str, int]]:
    """Extract all (law_name, article_number) pairs referenced in the QA benchmark.

    These articles are strictly protected: they must never appear in training
    or validation splits to eliminate data leakage.

    Args:
        qa_path: Path to the ground_truth_qa_kh.json file.

    Returns:
        Set of protected tuples: (expected_law, article_number).
    """
    if not qa_path.exists():
        raise FileNotFoundError(f"Benchmark QA file not found: {qa_path}")

    with open(qa_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    protected: set[tuple[str, int]] = set()
    for item in qa_data:
        law_name = item.get("expected_law")
        articles = item.get("expected_articles", [])
        if not law_name or not articles:
            continue
        for art in articles:
            protected.add((law_name, int(art)))

    logger.info(
        f"Loaded {len(qa_data)} benchmark questions citing "
        f"{len(protected)} unique protected articles."
    )
    return protected


def build_corpus_records(
    civil_chunks: list[dict[str, Any]],
    crim_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build standardized corpus records for all 1,976 legal articles.

    Args:
        civil_chunks: Chunk dictionaries for Civil Code 2007.
        crim_chunks: Chunk dictionaries for Criminal Code 2009.

    Returns:
        List of 1,976 unified corpus records.
    """
    records: list[dict[str, Any]] = []
    for chunk in civil_chunks + crim_chunks:
        meta = chunk.get("metadata", {})
        law_name = meta.get("law_name", "")
        art_num = int(meta.get("article_number", 0))
        doc_id = generate_doc_id(law_name, art_num)
        body = strip_title_line(chunk["content"])

        records.append({
            "doc_id": doc_id,
            "chunk_id": chunk.get("chunk_id", ""),
            "law_name": law_name,
            "law_name_kh": meta.get("law_name_kh"),
            "article_number": art_num,
            "title": meta.get("article_title", ""),
            "text": chunk["content"],
            "text_without_title": body,
            "content_with_context": chunk.get("content_with_context", ""),
            "book": meta.get("book"),
            "chapter": meta.get("chapter"),
            "section": meta.get("section"),
        })

    return records


def filter_candidate_pairs(
    civil_chunks: list[dict[str, Any]],
    crim_chunks: list[dict[str, Any]],
    protected_articles: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    """Filter candidate (query=title, passage=body) pairs.

    Enforces two strict constraints:
    1. Unique titles only: Articles sharing generic titles (e.g. 'បទប្បញ្ញត្តិទូទៅ')
       are excluded from query training.
    2. Leakage guard: Articles cited by any question in the T-Q benchmark are
       excluded from training and validation.

    Args:
        civil_chunks: Civil Code chunks.
        crim_chunks: Criminal Code chunks.
        protected_articles: Set of protected (law_name, article_number) pairs.

    Returns:
        List of eligible candidate pair dictionaries sorted deterministically.
    """
    # Count title occurrences within each code
    civ_counts = Counter(c["metadata"]["article_title"] for c in civil_chunks)
    crim_counts = Counter(c["metadata"]["article_title"] for c in crim_chunks)

    unique_civ = [c for c in civil_chunks if civ_counts[c["metadata"]["article_title"]] == 1]
    unique_crim = [c for c in crim_chunks if crim_counts[c["metadata"]["article_title"]] == 1]

    total_unique = len(unique_civ) + len(unique_crim)
    logger.info(
        f"Unique-title articles: {total_unique} "
        f"({len(unique_civ)} Civil + {len(unique_crim)} Criminal)."
    )

    if total_unique != EXPECTED_UNIQUE_TITLES:
        logger.warning(
            f"Expected {EXPECTED_UNIQUE_TITLES} unique titles, but found {total_unique}."
        )

    candidates: list[dict[str, Any]] = []
    dropped_leakage = 0

    for chunk in unique_civ + unique_crim:
        meta = chunk["metadata"]
        law_name = meta["law_name"]
        art_num = int(meta["article_number"])

        if (law_name, art_num) in protected_articles:
            dropped_leakage += 1
            continue

        doc_id = generate_doc_id(law_name, art_num)
        body = strip_title_line(chunk["content"])
        title = meta["article_title"]

        candidates.append({
            "query": title,
            "passage": body,
            "doc_id": doc_id,
            "law_name": law_name,
            "article_number": art_num,
            "title": title,
        })

    logger.info(
        f"Leakage guard dropped {dropped_leakage} articles cited in benchmark. "
        f"Remaining candidate training pairs: {len(candidates)}."
    )

    # Sort deterministically by (law_name, article_number) before shuffle
    candidates.sort(key=lambda x: (x["law_name"], x["article_number"]))
    return candidates


def create_splits(
    candidates: list[dict[str, Any]],
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, list[dict[str, Any]]]:
    """Split candidate pairs deterministically into train, val, and test_titles.

    Args:
        candidates: Deterministically ordered list of candidate pairs.
        seed: Random seed for reproducibility.
        train_ratio: Proportion for the training split (default: 0.8).
        val_ratio: Proportion for the validation split (default: 0.1).

    Returns:
        Dictionary mapping split name ('train', 'val', 'test_titles') to records.
    """
    shuffled = list(candidates)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_data = shuffled[:n_train]
    val_data = shuffled[n_train:n_train + n_val]
    test_data = shuffled[n_train + n_val:]

    # Assign structured sequence IDs
    for i, item in enumerate(train_data):
        item["id"] = f"train_{i + 1:04d}"
    for i, item in enumerate(val_data):
        item["id"] = f"val_{i + 1:04d}"
    for i, item in enumerate(test_data):
        item["id"] = f"tt_{i + 1:04d}"

    return {
        "train": train_data,
        "val": val_data,
        "test_titles": test_data,
    }


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write records to a JSON Lines file in UTF-8.

    Args:
        records: List of JSON-serializable dictionaries.
        output_path: Target file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_splits(
    chunks_dir: Path,
    qa_path: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, Any]:
    """Execute the complete dataset split preparation pipeline.

    Args:
        chunks_dir: Directory containing chunk JSON files.
        qa_path: Path to the benchmark QA file.
        output_dir: Target directory for generated split files.
        seed: Random seed for reproducible splitting.
        train_ratio: Training split ratio (default: 0.8).
        val_ratio: Validation split ratio (default: 0.1).

    Returns:
        Manifest dictionary describing dataset statistics and SHA-256 hashes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    civil_chunks, crim_chunks = load_corpus_chunks(chunks_dir)
    protected_articles = load_protected_articles(qa_path)

    # 2. Build and save full corpus.jsonl (1,976 articles)
    corpus_records = build_corpus_records(civil_chunks, crim_chunks)
    corpus_file = output_dir / "corpus.jsonl"
    write_jsonl(corpus_records, corpus_file)
    logger.info(f"Saved unified corpus ({len(corpus_records)} articles) -> {corpus_file}")

    # 3. Filter candidate pairs with leakage guard
    candidates = filter_candidate_pairs(civil_chunks, crim_chunks, protected_articles)

    # 4. Create deterministic splits
    splits = create_splits(
        candidates,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    # 5. Save split files
    split_files: dict[str, Path] = {
        "train": output_dir / "train.jsonl",
        "val": output_dir / "val.jsonl",
        "test_titles": output_dir / "test_titles.jsonl",
    }

    for name, path in split_files.items():
        write_jsonl(splits[name], path)
        logger.info(f"Saved {name} split ({len(splits[name])} pairs) -> {path}")

    # 6. Compute distribution statistics
    law_distribution: dict[str, dict[str, int]] = {}
    for name, split_records in splits.items():
        law_distribution[name] = dict(Counter(r["law_name"] for r in split_records))

    # 7. Compute SHA-256 hashes
    all_files = {"corpus.jsonl": corpus_file}
    for name, path in split_files.items():
        all_files[f"{name}.jsonl"] = path

    file_manifest: dict[str, dict[str, Any]] = {}
    for filename, path in all_files.items():
        count = len(corpus_records) if filename == "corpus.jsonl" else len(splits[filename.replace(".jsonl", "")])
        file_manifest[filename] = {
            "records_count": count,
            "sha256": compute_sha256(path),
        }

    # 8. Write manifest.json
    manifest: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "parameters": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": round(1.0 - train_ratio - val_ratio, 2),
        },
        "sources": {
            "civil_chunks": str(chunks_dir / "civil_code_2007_kh_chunks.json"),
            "criminal_chunks": str(chunks_dir / "criminal_code_2009_kh_chunks.json"),
            "qa_benchmark": str(qa_path),
        },
        "statistics": {
            "total_corpus_articles": len(corpus_records),
            "civil_code_articles": len(civil_chunks),
            "criminal_code_articles": len(crim_chunks),
            "unique_title_articles": EXPECTED_UNIQUE_TITLES,
            "qa_protected_articles_count": len(protected_articles),
            "candidates_after_leakage_guard": len(candidates),
            "split_counts": {k: len(v) for k, v in splits.items()},
            "law_distribution": law_distribution,
        },
        "files": file_manifest,
    }

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved dataset manifest -> {manifest_file}")

    return manifest


def main() -> None:
    """Command-line interface for split preparation."""
    parser = argparse.ArgumentParser(
        description="Prepare deterministic train/val/test splits for Khmer legal retrieval."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling (default: 42).",
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=Path("data/04_chunks"),
        help="Directory containing article chunk files (default: data/04_chunks).",
    )
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=Path("tests/evaluation/ground_truth_qa_kh.json"),
        help="Path to the benchmark QA file (default: tests/evaluation/ground_truth_qa_kh.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/05_splits"),
        help="Directory to save generated split files (default: data/05_splits).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of candidate pairs for training (default: 0.8).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of candidate pairs for validation (default: 0.1).",
    )

    args = parser.parse_args()

    logger.info(
        f"Starting dataset split preparation with seed={args.seed}, "
        f"chunks_dir={args.chunks_dir}, output_dir={args.output_dir}"
    )

    manifest = prepare_splits(
        chunks_dir=args.chunks_dir,
        qa_path=args.qa_path,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    stats = manifest["statistics"]
    print("\n" + "=" * 60)
    print("Khmer Legal Retrieval Dataset Split Summary")
    print("=" * 60)
    print(f"Total Corpus Articles:       {stats['total_corpus_articles']} (Civil: {stats['civil_code_articles']}, Criminal: {stats['criminal_code_articles']})")
    print(f"Unique-Title Articles:       {stats['unique_title_articles']}")
    print(f"QA Benchmark Protected Arts: {stats['qa_protected_articles_count']}")
    print(f"Eligible Candidate Pairs:    {stats['candidates_after_leakage_guard']}")
    print("-" * 60)
    print(f"Train Pairs:                 {stats['split_counts']['train']} (Civil: {stats['law_distribution']['train'].get('Civil Code 2007', 0)}, Criminal: {stats['law_distribution']['train'].get('Criminal Code 2009', 0)})")
    print(f"Val Pairs:                   {stats['split_counts']['val']} (Civil: {stats['law_distribution']['val'].get('Civil Code 2007', 0)}, Criminal: {stats['law_distribution']['val'].get('Criminal Code 2009', 0)})")
    print(f"Test (T-T) Pairs:            {stats['split_counts']['test_titles']} (Civil: {stats['law_distribution']['test_titles'].get('Civil Code 2007', 0)}, Criminal: {stats['law_distribution']['test_titles'].get('Criminal Code 2009', 0)})")
    print("=" * 60)
    print(f"Files saved to:              {args.output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
