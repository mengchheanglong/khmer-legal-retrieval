"""Unit tests for the legal hierarchical chunker."""

from src.infrastructure.chunking.legal_hierarchical_chunker import LegalHierarchicalChunker


class TestLegalHierarchicalChunker:
    """Tests for Article-level chunking with metadata extraction."""

    def setup_method(self) -> None:
        self.chunker = LegalHierarchicalChunker()

    def test_chunks_english_articles(self, sample_article_text_en: str) -> None:
        """Should split text into individual Article chunks."""
        chunks = self.chunker.chunk(
            text=sample_article_text_en,
            law_name="Civil Code 2007",
            language="en",
        )
        assert len(chunks) == 4  # Articles 1, 2, 315, 316

    def test_article_numbers_extracted(self, sample_article_text_en: str) -> None:
        """Each chunk should have the correct article number."""
        chunks = self.chunker.chunk(
            text=sample_article_text_en,
            law_name="Civil Code 2007",
            language="en",
        )
        article_numbers = [c.metadata.article_number for c in chunks]
        assert article_numbers == [1, 2, 315, 316]

    def test_chapter_metadata_tracked(self, sample_article_text_en: str) -> None:
        """Articles should inherit their parent Chapter context."""
        chunks = self.chunker.chunk(
            text=sample_article_text_en,
            law_name="Civil Code 2007",
            language="en",
        )
        # Articles 1 & 2 are in Chapter 1
        assert "CHAPTER 1" in (chunks[0].metadata.chapter or "")
        # Articles 315 & 316 are in Chapter 2
        assert "CHAPTER 2" in (chunks[2].metadata.chapter or "")

    def test_book_metadata_tracked(self, sample_article_text_en: str) -> None:
        """All articles should have Book 4 as parent context."""
        chunks = self.chunker.chunk(
            text=sample_article_text_en,
            law_name="Civil Code 2007",
            language="en",
        )
        for chunk in chunks:
            assert "BOOK 4" in (chunk.metadata.book or "")

    def test_content_with_context_has_prefix(self, sample_article_text_en: str) -> None:
        """content_with_context should include hierarchical prefix."""
        chunks = self.chunker.chunk(
            text=sample_article_text_en,
            law_name="Civil Code 2007",
            language="en",
        )
        assert "[Civil Code 2007]" in chunks[0].content_with_context

    def test_chunk_ids_are_unique(self, sample_article_text_en: str) -> None:
        """Each chunk should have a unique chunk_id."""
        chunks = self.chunker.chunk(
            text=sample_article_text_en,
            law_name="Civil Code 2007",
            language="en",
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunks_khmer_articles(self, sample_article_text_kh: str) -> None:
        """Should split Khmer text into individual Article chunks."""
        chunks = self.chunker.chunk(
            text=sample_article_text_kh,
            law_name="ក្រមរដ្ឋប្បវេណី",
            language="kh",
        )
        assert len(chunks) == 2  # មាត្រា ១ and មាត្រា ២

    def test_khmer_skips_toc_repeats_and_cross_references(self, sample_extracted_text_kh: str) -> None:
        """Table-of-contents repeats and inline 'មាត្រា ៣៣៦ នៃ...' references are not articles."""
        chunks = self.chunker.chunk(text=sample_extracted_text_kh, law_name="Civil Code 2007", language="kh")
        assert [c.metadata.article_number for c in chunks] == [1, 2, 3, 4]

    def test_khmer_body_sentence_citing_a_book_is_not_a_heading(self, sample_extracted_text_kh: str) -> None:
        """'គន្ថីទី ១ (...) នៃក្រមនេះ ... ។' is article text, not a Book heading."""
        chunks = self.chunker.chunk(text=sample_extracted_text_kh, law_name="Civil Code 2007", language="kh")
        assert "គន្ថីទី ១ (បទប្បញ្ញត្តិទូទៅ) នៃក្រមនេះ ត្រូវយកមកអនុវត្ត ។" in chunks[3].content

    def test_khmer_lower_headings_reset_under_new_book(self, sample_extracted_text_kh: str) -> None:
        """A section from an earlier book must not carry over into a new book."""
        chunks = self.chunker.chunk(text=sample_extracted_text_kh, law_name="Civil Code 2007", language="kh")
        assert chunks[3].metadata.book == "គន្ថីទី ៣ កាតព្វកិច្ច"
        assert chunks[3].metadata.chapter == "ជំពូកទោល បទប្បញ្ញត្តិទូទៅ"
        assert chunks[3].metadata.section is None

    def test_khmer_article_titles_extracted(self, sample_extracted_text_kh: str) -> None:
        """The text after 'មាត្រា N.-' is the article title."""
        chunks = self.chunker.chunk(text=sample_extracted_text_kh, law_name="Civil Code 2007", language="kh")
        assert chunks[0].metadata.article_title == "ច្បាប់ទូទៅនៃនីតិឯកជន"
        assert chunks[2].metadata.article_title == "គោលការណ៍ស្វ័យភាពនៃបុគ្គលឯកជន"

    def test_khmer_hierarchy_metadata(self, sample_extracted_text_kh: str) -> None:
        """Khmer articles inherit Book / Chapter / Section headings, including ជំពូកទោល."""
        chunks = self.chunker.chunk(text=sample_extracted_text_kh, law_name="Civil Code 2007", language="kh")
        assert chunks[0].metadata.book == "គន្ថីទី ១ បទប្បញ្ញត្តិទូទៅ"
        assert chunks[0].metadata.chapter == "ជំពូកទោល បទប្បញ្ញត្តិទូទៅ"
        assert chunks[2].metadata.book == "គន្ថីទី ២ បុគ្គល"
        assert chunks[2].metadata.chapter == "ជំពូកទី ១ បុគ្គលរូបវន្ត"
        assert chunks[2].metadata.section == "ផ្នែកទី ១ សមត្ថភាពសិទ្ធិ"

    def test_khmer_content_stops_before_next_heading(self, sample_extracted_text_kh: str) -> None:
        """Headings between articles belong to the next article, not the previous one's content."""
        chunks = self.chunker.chunk(text=sample_extracted_text_kh, law_name="Civil Code 2007", language="kh")
        assert chunks[1].content.endswith("២-មិនអនុញ្ញាតឱ្យរំលោភសិទ្ធិ ។")
        assert "គន្ថីទី ២" not in chunks[1].content

    def test_khmer_law_name_in_context(self, sample_extracted_text_kh: str) -> None:
        """Khmer chunks carry the Khmer law name in metadata and context prefix."""
        chunks = self.chunker.chunk(
            text=sample_extracted_text_kh, law_name="Civil Code 2007", language="kh", law_name_kh="ក្រមរដ្ឋប្បវេណី"
        )
        assert chunks[0].metadata.law_name_kh == "ក្រមរដ្ឋប្បវេណី"
        assert chunks[0].content_with_context.startswith("[ក្រមរដ្ឋប្បវេណី]")

    def test_empty_text_raises_error(self) -> None:
        """Should raise ChunkingError for empty input."""
        import pytest
        from src.domain.exceptions import ChunkingError

        with pytest.raises(ChunkingError):
            self.chunker.chunk(text="", law_name="Test", language="en")
