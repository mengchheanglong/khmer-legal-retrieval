"""
Hierarchical legal text chunker for Cambodian law.

Splits legal documents by their natural hierarchy:
Book (គន្ថី) → Title (មាតិកា) → Chapter (ជំពូក) → Section (ផ្នែក) → Article (មាត្រា)

Each chunk corresponds to one Article, tagged with its full
hierarchical path for context-aware embedding and retrieval.
"""

import bisect
import hashlib
from typing import Optional

import regex as re

from src.config.logging import get_logger
from src.domain.entities import Language, LegalChunk, LegalMetadata
from src.domain.exceptions import ChunkingError
from src.domain.ports.chunker_port import ChunkerPort

logger = get_logger(__name__)

# ── Khmer numeral conversion ────────────────────────────────────────────
KHMER_NUMERALS = {"០": 0, "១": 1, "២": 2, "៣": 3, "៤": 4, "៥": 5, "៦": 6, "៧": 7, "៨": 8, "៩": 9}


def khmer_to_int(khmer_str: str) -> int:
    """Convert Khmer numeral string or regular digits to integer."""
    if khmer_str.isdigit():
        return int(khmer_str)
    res = 0
    for char in khmer_str:
        if char in KHMER_NUMERALS:
            res = res * 10 + KHMER_NUMERALS[char]
    return res if res > 0 else 0


# ── Regex Patterns ──────────────────────────────────────────────────────

# English Book/Title/Chapter/Section patterns
_EN_BOOK = re.compile(
    r"(?:^|\n)\s*(BOOK\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|\d+|[IVXLCDM]+)[^\n]*)",
    re.IGNORECASE,
)
_EN_TITLE = re.compile(
    r"(?:^|\n)\s*(TITLE\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|\d+|[IVXLCDM]+)[^\n]*)",
    re.IGNORECASE,
)
_EN_CHAPTER = re.compile(
    r"(?:^|\n)\s*(CHAPTER\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|\d+|[IVXLCDM]+)[^\n]*)",
    re.IGNORECASE,
)
_EN_SECTION = re.compile(
    r"(?:^|\n)\s*(SECTION\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|\d+|[IVXLCDM]+|\d+\b)[^\n]*)",
    re.IGNORECASE,
)

# English Article patterns
_EN_ARTICLE_EXPLICIT = re.compile(
    r"(?:^|\n)\s*Article\s+(\d+)[:.]?\s*(?:\n\s*([^\n]+))?",
    re.IGNORECASE,
)
_EN_ARTICLE_NUMBERED = re.compile(
    r"(?:^|\n)\s*(\d+)\.\s*(?:\n\s*)?(?:\(([^\)\n]+)\))?",
)

# Khmer hierarchy patterns: one heading per line, e.g. "គន្ថីទី ១ បទប្បញ្ញត្តិទូទៅ", "ជំពូកទោល ..."
_KH_NUM = r"(?:ទី)?[ \t]*[០-៩\d]+"
# A heading is a short line without the Khmer full stop (។), so a body sentence such as
# "គន្ថីទី ១ (បទប្បញ្ញត្តិទូទៅ) នៃក្រមនេះ ត្រូវ... ។" is never mistaken for a heading
_KH_HEADING_TAIL = r"[^\n។]{0,150}(?=\n|$)"
_KH_BOOK = re.compile(rf"(?:^|\n)[ \t]*(គន្ថី{_KH_NUM}{_KH_HEADING_TAIL})")
_KH_TITLE = re.compile(rf"(?:^|\n)[ \t]*(មាតិកា{_KH_NUM}{_KH_HEADING_TAIL})")
_KH_CHAPTER = re.compile(rf"(?:^|\n)[ \t]*(ជំពូក(?:ទោល|{_KH_NUM}){_KH_HEADING_TAIL})")
_KH_SECTION = re.compile(rf"(?:^|\n)[ \t]*(ផ្នែក{_KH_NUM}{_KH_HEADING_TAIL})")
_KH_SUBSECTION = re.compile(rf"(?:^|\n)[ \t]*(កថាភាគ{_KH_NUM}{_KH_HEADING_TAIL})")
# Article header "មាត្រា ៣៣៦.- title". The punctuation after the number separates a header
# from a cross-reference such as "មាត្រា ៣៣៦ នៃក្រមនេះ".
_KH_ARTICLE = re.compile(r"(?:^|\n)[ \t]*មាត្រា[ \t]*([០-៩\d]+)[ \t]*[.។][ \t]*-?[ \t]*([^\n]*)")


class LegalHierarchicalChunker(ChunkerPort):
    """
    Chunks Cambodian legal text by Article with hierarchical metadata.

    Supports:
    1. Civil Code format: `1. (Title)` / `1.\n(Title)` (~1,300 articles)
    2. Standard statute format: `Article 1: Title` / `Article 1.`
    3. Khmer statute format: `មាត្រា ១.- Title` / `មាត្រា ១។`
    """

    def chunk(
        self,
        text: str,
        law_name: str,
        language: str = "en",
        law_name_kh: Optional[str] = None,
    ) -> list[LegalChunk]:
        """Split legal text into Article-level chunks with metadata."""
        if not text.strip():
            raise ChunkingError("Input text is empty")

        lang = Language(language)

        if lang == Language.ENGLISH:
            return self._chunk_english(text, law_name, law_name_kh)
        elif lang == Language.KHMER:
            return self._chunk_khmer(text, law_name, law_name_kh)
        else:
            raise ChunkingError(f"Unsupported language: {language}")

    def _chunk_english(self, text: str, law_name: str, law_name_kh: Optional[str] = None) -> list[LegalChunk]:
        """Detect format and chunk English legal text."""
        explicit_matches = list(_EN_ARTICLE_EXPLICIT.finditer(text))
        numbered_matches = list(_EN_ARTICLE_NUMBERED.finditer(text))

        # Determine which pattern dominates
        if len(explicit_matches) > 0 and len(explicit_matches) >= len(numbered_matches):
            # Standard "Article XX" format
            matches = [(m.start(), int(m.group(1)), m.group(2) or "") for m in explicit_matches]
            return self._chunk_with_article_matches(
                text=text,
                law_name=law_name,
                language=Language.ENGLISH,
                matches=matches,
                book_pattern=_EN_BOOK,
                title_pattern=_EN_TITLE,
                chapter_pattern=_EN_CHAPTER,
                section_pattern=_EN_SECTION,
                law_name_kh=law_name_kh,
            )
        elif len(numbered_matches) > 0:
            # Numbered "1. (Title)" format (Civil Code)
            filtered = []
            expected_next = 1
            for m in numbered_matches:
                num = int(m.group(1))
                title = m.group(2) or ""
                if num >= expected_next and num <= expected_next + 15:
                    filtered.append((m.start(), num, title))
                    expected_next = num + 1

            if not filtered or len(filtered) < len(numbered_matches) * 0.5:
                filtered = [(m.start(), int(m.group(1)), m.group(2) or "") for m in numbered_matches]

            return self._chunk_with_article_matches(
                text=text,
                law_name=law_name,
                language=Language.ENGLISH,
                matches=filtered,
                book_pattern=_EN_BOOK,
                title_pattern=_EN_TITLE,
                chapter_pattern=_EN_CHAPTER,
                section_pattern=_EN_SECTION,
                law_name_kh=law_name_kh,
            )
        else:
            raise ChunkingError(f"No legal articles recognized in '{law_name}'")

    def _chunk_khmer(self, text: str, law_name: str, law_name_kh: Optional[str] = None) -> list[LegalChunk]:
        """Chunk Khmer legal text by Article (មាត្រា) boundaries."""
        candidates = [
            (m.start(), khmer_to_int(m.group(1)), m.group(2) or "")
            for m in _KH_ARTICLE.finditer(text)
        ]
        matches = self._filter_sequential(candidates)

        if not matches:
            raise ChunkingError(f"No Khmer articles found in '{law_name}'")

        return self._chunk_with_article_matches(
            text=text,
            law_name=law_name,
            language=Language.KHMER,
            matches=matches,
            book_pattern=_KH_BOOK,
            title_pattern=_KH_TITLE,
            chapter_pattern=_KH_CHAPTER,
            section_pattern=_KH_SECTION,
            subsection_pattern=_KH_SUBSECTION,
            stop_at_headings=True,
            reset_lower_levels=True,
            law_name_kh=law_name_kh,
        )

    @staticmethod
    def _filter_sequential(
        candidates: list[tuple[int, int, str]],
        max_gap: int = 15,
        lookahead: int = 10,
    ) -> list[tuple[int, int, str]]:
        """
        Keep article headers that continue the numbering 1, 2, 3, ...

        Drops out-of-sequence headers such as table-of-contents repeats. A header that jumps
        ahead is also skipped when a closer number follows within `lookahead` candidates.
        """
        kept: list[tuple[int, int, str]] = []
        expected = 1
        for i, (pos, num, title) in enumerate(candidates):
            if not expected <= num <= expected + max_gap:
                continue
            upcoming = candidates[i + 1:i + 1 + lookahead]
            if num > expected and any(expected <= c[1] < num for c in upcoming):
                continue
            kept.append((pos, num, title))
            expected = num + 1
        return kept

    def _chunk_with_article_matches(
        self,
        text: str,
        law_name: str,
        language: Language,
        matches: list[tuple[int, int, str]],
        book_pattern: re.Pattern,
        title_pattern: Optional[re.Pattern],
        chapter_pattern: re.Pattern,
        section_pattern: re.Pattern,
        subsection_pattern: Optional[re.Pattern] = None,
        stop_at_headings: bool = False,
        reset_lower_levels: bool = False,
        law_name_kh: Optional[str] = None,
    ) -> list[LegalChunk]:
        """Build LegalChunk objects with fast hierarchy lookup."""
        chunks: list[LegalChunk] = []

        # Pre-index all hierarchy markers
        books_pos = [(m.start(), m.group(1).strip()) for m in book_pattern.finditer(text)]
        titles_pos = [(m.start(), m.group(1).strip()) for m in title_pattern.finditer(text)] if title_pattern else []
        chapters_pos = [(m.start(), m.group(1).strip()) for m in chapter_pattern.finditer(text)]
        sections_pos = [(m.start(), m.group(1).strip()) for m in section_pattern.finditer(text)]
        subsections_pos = [m.start() for m in subsection_pattern.finditer(text)] if subsection_pattern else []

        # Headings that close the preceding article, so they don't leak into its content
        heading_starts = sorted(
            [p for p, _ in books_pos + titles_pos + chapters_pos + sections_pos] + subsections_pos
        ) if stop_at_headings else []

        logger.info(
            "Chunking articles",
            law_name=law_name,
            article_count=len(matches),
        )

        for i, (start_pos, article_num, article_title) in enumerate(matches):
            end_pos = matches[i + 1][0] if i + 1 < len(matches) else len(text)
            next_heading = bisect.bisect_right(heading_starts, start_pos)
            if next_heading < len(heading_starts):
                end_pos = min(end_pos, heading_starts[next_heading])
            content = text[start_pos:end_pos].strip()

            # Fast binary search for parent hierarchy markers
            current_book, current_title, current_chapter, current_section = self._resolve_hierarchy(
                [books_pos, titles_pos, chapters_pos, sections_pos], start_pos, reset_lower_levels
            )

            metadata = LegalMetadata(
                law_name=law_name,
                law_name_kh=law_name_kh,
                book=current_book,
                title=current_title,
                chapter=current_chapter,
                section=current_section,
                article_number=article_num,
                article_title=article_title.strip() if article_title else None,
                language=language,
            )

            # Build context prefix
            context_law_name = law_name_kh if language == Language.KHMER and law_name_kh else law_name
            context_parts = [f"[{context_law_name}]"]
            if current_book:
                context_parts.append(f"[{current_book}]")
            if current_chapter:
                context_parts.append(f"[{current_chapter}]")
            if current_section:
                context_parts.append(f"[{current_section}]")
            context_prefix = " → ".join(context_parts)
            content_with_context = f"{context_prefix}\n{content}"

            chunk_id = self._generate_chunk_id(law_name, article_num, language.value)

            chunks.append(
                LegalChunk(
                    chunk_id=chunk_id,
                    content=content,
                    content_with_context=content_with_context,
                    metadata=metadata,
                )
            )

        logger.info("Successfully chunked", law_name=law_name, chunks=len(chunks))
        return chunks

    @staticmethod
    def _resolve_hierarchy(
        levels: list[list[tuple[int, str]]],
        target_pos: int,
        reset_lower_levels: bool,
    ) -> list[Optional[str]]:
        """
        Find the latest heading of each level (ordered highest first) before target_pos.

        With reset_lower_levels, a heading is dropped when a higher-level heading appears after
        it, e.g. a section from an earlier chapter must not carry into a chapter without sections.
        """
        labels: list[Optional[str]] = []
        higher_pos = -1
        for positions in levels:
            entry = LegalHierarchicalChunker._lookup_latest_entry(positions, target_pos)
            if entry is None or (reset_lower_levels and entry[0] < higher_pos):
                labels.append(None)
            else:
                labels.append(entry[1])
            if entry is not None:
                higher_pos = max(higher_pos, entry[0])
        return labels

    @staticmethod
    def _lookup_latest(positions: list[tuple[int, str]], target_pos: int) -> Optional[str]:
        """Find the latest marker before target_pos using binary search."""
        entry = LegalHierarchicalChunker._lookup_latest_entry(positions, target_pos)
        return entry[1] if entry else None

    @staticmethod
    def _lookup_latest_entry(positions: list[tuple[int, str]], target_pos: int) -> Optional[tuple[int, str]]:
        """Find the latest (offset, label) marker before target_pos using binary search."""
        if not positions:
            return None
        # Extract just the start offsets
        offsets = [p[0] for p in positions]
        idx = bisect.bisect_right(offsets, target_pos) - 1
        if idx >= 0:
            return positions[idx]
        return None

    @staticmethod
    def _generate_chunk_id(law_name: str, article_number: int, language: str) -> str:
        """Generate a deterministic, unique chunk ID."""
        raw = f"{law_name}::{article_number}::{language}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
