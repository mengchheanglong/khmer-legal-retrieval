"""
Chunking pipeline script.

Processes extracted plain text files from data/02_extracted/,
applies hierarchical legal chunking (Article-level with Book/Chapter metadata),
and writes structured chunks to data/04_chunks/ as JSON and JSONL.
"""

import json
from pathlib import Path

from src.config.logging import get_logger, setup_logging
from src.infrastructure.chunking.legal_hierarchical_chunker import LegalHierarchicalChunker

setup_logging()
logger = get_logger(__name__)

DOC_CHUNK_CONFIGS = [
    {
        "law_name": "Civil Code 2007",
        "law_key": "civil_code_2007_en",
        "extracted_file": "data/02_extracted/civil_code_2007_en.txt",
        "language": "en",
    },
    {
        "law_name": "Law on Commercial Arbitration 2006",
        "law_key": "commercial_arbitration_2006_en",
        "extracted_file": "data/02_extracted/commercial_arbitration_2006_en.txt",
        "language": "en",
    },
    {
        "law_name": "Civil Code 2007",
        "law_name_kh": "ក្រមរដ្ឋប្បវេណី",
        "law_key": "civil_code_2007_kh",
        "extracted_file": "data/02_extracted/civil_code_2007_kh.txt",
        "language": "kh",
    },
    {
        "law_name": "Criminal Code 2009",
        "law_name_kh": "ក្រមព្រហ្មទណ្ឌ",
        "law_key": "criminal_code_2009_kh",
        "extracted_file": "data/02_extracted/criminal_code_2009_kh.txt",
        "language": "kh",
    },
]


def run_chunking() -> None:
    """Execute article-level chunking for all extracted documents."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    chunks_dir = base_dir / "data" / "04_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunker = LegalHierarchicalChunker()
    all_chunks_summary = []

    for config in DOC_CHUNK_CONFIGS:
        txt_path = base_dir / config["extracted_file"]
        law_name = config["law_name"]
        law_key = config["law_key"]
        language = config["language"]

        if not txt_path.exists():
            logger.warning(f"Extracted file not found: {txt_path}. Skipping.")
            continue

        logger.info(f"Chunking '{law_name}' from {txt_path.name}...")

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunker.chunk(
            text=text,
            law_name=law_name,
            language=language,
            law_name_kh=config.get("law_name_kh"),
        )

        # Write as JSON (array of chunks)
        output_json = chunks_dir / f"{law_key}_chunks.json"
        chunks_data = [chunk.model_dump(mode="json") for chunk in chunks]
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        # Write as JSONL (one chunk per line, great for streaming & embedding)
        output_jsonl = chunks_dir / f"{law_key}_chunks.jsonl"
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for item in chunks_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        summary = {
            "law_name": law_name,
            "law_key": law_key,
            "chunks_count": len(chunks),
            "first_article": chunks[0].metadata.article_number if chunks else None,
            "last_article": chunks[-1].metadata.article_number if chunks else None,
            "json_path": str(output_json.relative_to(base_dir)),
            "jsonl_path": str(output_jsonl.relative_to(base_dir)),
        }
        all_chunks_summary.append(summary)

        logger.info(
            f"Successfully chunked '{law_name}': {len(chunks)} article chunks -> {output_json.name}"
        )

    # Write overall index summary
    summary_file = chunks_dir / "chunks_manifest.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks_summary, f, indent=2, ensure_ascii=False)

    logger.info("Chunking pipeline finished.", total_documents=len(all_chunks_summary))


if __name__ == "__main__":
    run_chunking()
