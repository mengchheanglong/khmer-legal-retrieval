"""Unit tests for dataset split preparation and leakage prevention."""

import json
from pathlib import Path
import pytest

from src.dl.prepare_splits import (
    compute_sha256,
    generate_doc_id,
    strip_title_line,
    load_corpus_chunks,
    load_protected_articles,
    filter_candidate_pairs,
    create_splits,
    prepare_splits,
    TOTAL_CORPUS_COUNT,
    CIVIL_CODE_COUNT,
    CRIMINAL_CODE_COUNT,
    EXPECTED_UNIQUE_TITLES,
)


class TestTitleStripping:
    """Tests for title line stripping from chunk passages."""

    def test_strip_title_line_standard(self) -> None:
        """Should strip leading article header and preserve body."""
        content = "មាត្រា ១.- ច្បាប់ទូទៅនៃនីតិឯកជន\nក្រមនេះ បញ្ញត្តអំពីគោលការណ៍ទូទៅ..."
        body = strip_title_line(content)
        assert body == "ក្រមនេះ បញ្ញត្តអំពីគោលការណ៍ទូទៅ..."
        assert not body.startswith("មាត្រា")

    def test_strip_title_line_multiline_body(self) -> None:
        """Should preserve multiline body formatting."""
        content = "មាត្រា ២.- ការអនុវត្ត\nកថាខណ្ឌ ១\nកថាខណ្ឌ ២"
        body = strip_title_line(content)
        assert body == "កថាខណ្ឌ ១\nកថាខណ្ឌ ២"

    def test_strip_title_line_missing_header_raises(self) -> None:
        """Should raise ValueError if content does not start with មាត្រា."""
        with pytest.raises(ValueError, match="does not begin with an article header"):
            strip_title_line("បទបញ្ញត្តិទូទៅ\nខ្លឹមសារ...")

    def test_strip_title_line_empty_body_raises(self) -> None:
        """Should raise ValueError if article body is empty."""
        with pytest.raises(ValueError, match="body is empty"):
            strip_title_line("មាត្រា ១.- ចំណងជើង\n   \n")


class TestDocIdGeneration:
    """Tests for standardized document identifier generation."""

    def test_civil_code_doc_id(self) -> None:
        assert generate_doc_id("Civil Code 2007", 1) == "civil_code_kh_art_1"
        assert generate_doc_id("Civil Code 2007", 1304) == "civil_code_kh_art_1304"

    def test_criminal_code_doc_id(self) -> None:
        assert generate_doc_id("Criminal Code 2009", 1) == "criminal_code_kh_art_1"
        assert generate_doc_id("Criminal Code 2009", 672) == "criminal_code_kh_art_672"


class TestCorpusAndLeakageGuard:
    """Tests for corpus integrity and test-set leakage prevention."""

    @pytest.fixture
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent.parent

    @pytest.fixture
    def chunks_dir(self, base_dir: Path) -> Path:
        return base_dir / "data" / "04_chunks"

    @pytest.fixture
    def qa_path(self, base_dir: Path) -> Path:
        return base_dir / "tests" / "evaluation" / "ground_truth_qa_kh.json"

    @pytest.fixture
    def splits_dir(self, base_dir: Path) -> Path:
        return base_dir / "data" / "05_splits"

    def test_corpus_chunk_counts(self, chunks_dir: Path) -> None:
        """Verify Civil and Criminal Code chunk counts."""
        civ, crim = load_corpus_chunks(chunks_dir)
        assert len(civ) == CIVIL_CODE_COUNT
        assert len(crim) == CRIMINAL_CODE_COUNT
        assert len(civ) + len(crim) == TOTAL_CORPUS_COUNT

    def test_qa_benchmark_coverage(self, qa_path: Path) -> None:
        """Benchmark should contain exactly 200 questions across both laws."""
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        assert len(qa_data) == 200

        civ_qa = [q for q in qa_data if q.get("expected_law") == "Civil Code 2007"]
        crim_qa = [q for q in qa_data if q.get("expected_law") == "Criminal Code 2009"]
        assert len(civ_qa) == 100
        assert len(crim_qa) == 100

        protected = load_protected_articles(qa_path)
        assert len(protected) > 200  # Multi-article questions cite > 200 unique articles

    def test_leakage_guard_removes_all_test_articles(
        self, chunks_dir: Path, qa_path: Path
    ) -> None:
        """Candidate pairs must not contain any protected articles."""
        civ, crim = load_corpus_chunks(chunks_dir)
        protected = load_protected_articles(qa_path)
        candidates = filter_candidate_pairs(civ, crim, protected)

        candidate_articles = {(c["law_name"], c["article_number"]) for c in candidates}
        overlap = candidate_articles.intersection(protected)
        assert len(overlap) == 0, f"Found {len(overlap)} leaked articles in candidates: {overlap}"

    def test_splits_files_exist_and_match_manifest(self, splits_dir: Path) -> None:
        """Generated split files must exist and match checksums in manifest.json."""
        manifest_path = splits_dir / "manifest.json"
        assert manifest_path.exists(), "manifest.json does not exist"

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["statistics"]["total_corpus_articles"] == TOTAL_CORPUS_COUNT
        assert manifest["statistics"]["unique_title_articles"] == EXPECTED_UNIQUE_TITLES

        for filename, info in manifest["files"].items():
            file_path = splits_dir / filename
            assert file_path.exists(), f"File {filename} is missing"

            # Check line count
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
            assert len(lines) == info["records_count"]

            # Check SHA-256 hash
            actual_sha = compute_sha256(file_path)
            assert actual_sha == info["sha256"], f"Checksum mismatch for {filename}"

    def test_train_val_test_zero_leakage_against_qa(
        self, splits_dir: Path, qa_path: Path
    ) -> None:
        """Verify strict zero-leakage: no T-Q article appears in train, val, or test_titles."""
        protected = load_protected_articles(qa_path)

        for split_name in ["train.jsonl", "val.jsonl", "test_titles.jsonl"]:
            split_path = splits_dir / split_name
            with open(split_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f]

            for record in records:
                key = (record["law_name"], record["article_number"])
                assert key not in protected, (
                    f"Leaked article found in {split_name}: {key} "
                    f"(Title: {record.get('title')})"
                )

    def test_title_stripped_from_passages_in_splits(self, splits_dir: Path) -> None:
        """Passages in all split pairs must not begin with the header line or article title."""
        for split_name in ["train.jsonl", "val.jsonl", "test_titles.jsonl"]:
            split_path = splits_dir / split_name
            with open(split_path, "r", encoding="utf-8") as f:
                records = [json.loads(line) for line in f]

            for record in records:
                passage = record["passage"].strip()
                assert not passage.startswith("មាត្រា"), (
                    f"Passage still has header line in {split_name} (id={record['id']})"
                )
                assert len(passage) > 0
                assert len(record["query"]) > 0

    def test_split_determinism(self, tmp_path: Path, chunks_dir: Path, qa_path: Path) -> None:
        """prepare_splits must produce bit-for-bit identical outputs for the same seed."""
        out1 = tmp_path / "split1"
        out2 = tmp_path / "split2"

        manifest1 = prepare_splits(chunks_dir, qa_path, out1, seed=42)
        manifest2 = prepare_splits(chunks_dir, qa_path, out2, seed=42)

        for fname in ["train.jsonl", "val.jsonl", "test_titles.jsonl"]:
            content1 = (out1 / fname).read_bytes()
            content2 = (out2 / fname).read_bytes()
            assert content1 == content2, f"{fname} differs between identical seed runs"

        assert manifest1["files"]["train.jsonl"]["sha256"] == manifest2["files"]["train.jsonl"]["sha256"]
