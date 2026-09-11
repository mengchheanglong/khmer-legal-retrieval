"""
PyMuPDF extractor for Khmer legal PDFs typeset in legacy Limon fonts.

Implements ExtractorPort. Converts Limon-font spans to Khmer Unicode, drops page
numbers, keeps headings and article headers (with their full titles) on their own
lines, and re-joins body lines that the PDF wrapped mid-word.
"""

from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import regex as re

from src.config.logging import get_logger
from src.domain.exceptions import ExtractionError
from src.domain.ports.extractor_port import ExtractorPort
from src.infrastructure.extractors.limon_converter import is_limon_font, limon_to_unicode

logger = get_logger(__name__)

_SCANNED_THRESHOLD = 10  # characters per page
_PAGE_NUMBER = re.compile(r"^[\s\-–]*[០-៩0-9]+[\s\-–]*$")
_LIST_ITEM = re.compile(r"^(?:[០-៩0-9]+|[ក-អ])\s*-")  # "១-..." or "ក-..."
_STRUCTURAL = re.compile(r"^(?:មាត្រា|គន្ថី|មាតិកា|ជំពូក|ផ្នែក|កថាភាគ)")
_ARTICLE_HEADER = re.compile(r"^មាត្រា\s*[០-៩0-9]+\s*[.។]\s*-")  # "មាត្រា ៣៣៦.- ..."
_ARTICLE_HEADER_ONLY = re.compile(r"^មាត្រា\s*[០-៩0-9]+\s*[.។]\s*-\s*$")  # title set beside or below
_HYPHEN_BREAK = re.compile(r"(?<=[ក-៓]{2})-$")  # "អាណា-" + "ព្យាបាល" split across lines
_KHMER_LETTER_END = re.compile(r"[ក-៓]$")
_KHMER_LETTER_START = re.compile(r"^[ក-៓]")
# Dangling prepositions or conjunctions that cannot end an article title
_DANGLING_CONNECTORS = re.compile(r"(?:អំពី|នៃ|និង|ឬ|ដោយ|ដែល|ក្នុង|ចំពោះ|លើ|ក្រោម|តាម|ដូចជា|ដើម្បី)$")
# A wrapped article title starts where the title text starts in the header line; body paragraphs
# are indented differently (Civil Code: title 156pt vs body 57/99pt; Criminal Code: ~185pt vs 99/135pt).
# Tolerance 16.0pt accommodates Criminal Code wrapped titles (placed at ~184pt vs line-1 start 181-198pt)
# while keeping safely clear of body paragraphs indented at 135pt (gap > 46pt).
_TITLE_ALIGN_TOLERANCE = 16.0
# A whitespace span from a non-Limon font is a placeholder inside a word when the Limon glyphs on
# either side touch (gap ≈ 0pt, e.g. before the subscript in "ផ្តន្ទាទោស"). Real word spaces in the
# Criminal Code leave a visible gap of 2.5-18pt.
_PLACEHOLDER_MAX_GAP = 1.0

Line = tuple[str, bool, float, Optional[float]]  # (text, is_heading, x0, title_x)


class LimonPdfExtractor(ExtractorPort):
    """Extract Khmer Unicode text from legal PDFs typeset in Limon fonts."""

    def extract(
        self,
        pdf_path: Path,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> str:
        """Extract Khmer Unicode text with one heading or paragraph per line."""
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise ExtractionError(str(pdf_path), str(e))

        last_page = end_page if end_page else len(doc)
        logger.info("Extracting Limon PDF", path=str(pdf_path), pages=f"{start_page}-{last_page}")

        lines: list[Line] = []
        for page_num in range(start_page - 1, last_page):
            lines.extend(self._page_lines(doc[page_num]))
        doc.close()

        result = self._join_lines(lines)
        logger.info("Extraction complete", characters=len(result))
        return result

    def is_scanned(self, pdf_path: Path) -> bool:
        """Detect image-only PDFs (e.g. the ODC Khmer Labour Law), which need OCR instead."""
        try:
            doc = fitz.open(str(pdf_path))
        except Exception:
            return False
        sample_pages = min(len(doc), 5)
        total_chars = sum(len(doc[i].get_text("text").strip()) for i in range(sample_pages))
        doc.close()
        return total_chars / max(sample_pages, 1) < _SCANNED_THRESHOLD

    @staticmethod
    def _page_lines(page: fitz.Page) -> list[Line]:
        """Return (text, is_heading, x0, title_x) per visual line, top to bottom, without page numbers."""
        result = []
        for block in page.get_text("dict", sort=True)["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"]]
                text = LimonPdfExtractor._line_text(spans).strip()
                if not text or _PAGE_NUMBER.match(text):
                    continue
                dominant_font = max(spans, key=lambda s: len(s["text"].strip()))["font"]
                is_header = bool(_ARTICLE_HEADER.match(text))
                result.append((
                    text,
                    LimonPdfExtractor._is_heading(text, dominant_font),
                    line["bbox"][0],
                    LimonPdfExtractor._title_x(spans) if is_header else None,
                ))
        return result

    @staticmethod
    def _is_heading(text: str, dominant_font: str) -> bool:
        """
        Headings are set in the bold Limon R fonts. Article headers always count as headings:
        "មាត្រា N.-" is bold but the title after it is often set in the body font.
        """
        return "LimonR" in dominant_font or bool(_ARTICLE_HEADER.match(text))

    @staticmethod
    def _title_x(spans: list[dict]) -> Optional[float]:
        """Left edge of the title text in an article header line: the first non-blank span after '-'."""
        seen_dash = False
        for span in spans:
            if seen_dash and span["text"].strip():
                return span["bbox"][0]
            seen_dash = seen_dash or "-" in span["text"]
        return None

    @staticmethod
    def _line_text(spans: list[dict]) -> str:
        """Convert contiguous Limon spans together so syllables split across spans stay intact."""
        parts: list[str] = []
        limon_run: list[str] = []
        for i, span in enumerate(spans):
            if is_limon_font(span["font"]):
                limon_run.append(span["text"])
                continue
            if LimonPdfExtractor._is_placeholder_space(spans, i):
                continue  # keep the Limon run intact so the word converts as one syllable sequence
            if limon_run:
                parts.append(limon_to_unicode("".join(limon_run)))
                limon_run = []
            parts.append(span["text"])
        if limon_run:
            parts.append(limon_to_unicode("".join(limon_run)))
        return re.sub(r"[ \t]{2,}", " ", "".join(parts))

    @staticmethod
    def _is_placeholder_space(spans: list[dict], i: int) -> bool:
        """True for a whitespace-only non-Limon span squeezed between two touching Limon spans."""
        if spans[i]["text"].strip() or i == 0 or i + 1 >= len(spans):
            return False
        prev_span, next_span = spans[i - 1], spans[i + 1]
        return (
            is_limon_font(prev_span["font"])
            and is_limon_font(next_span["font"])
            and next_span["bbox"][0] - prev_span["bbox"][2] < _PLACEHOLDER_MAX_GAP
        )

    @staticmethod
    def _join_lines(lines: list[Line]) -> str:
        """
        Rebuild the document with one heading, article header or paragraph per line.

        Wrapped article titles (aligned with the title text) join their header, wrapped heading
        names join their heading, and wrapped body lines join their paragraph.
        """
        out: list[str] = []
        prev_heading = True
        in_article_header = False
        title_x: Optional[float] = None
        for text, is_heading, x0, line_title_x in lines:
            new_block = _STRUCTURAL.match(text) or _LIST_ITEM.match(text)
            bare_header = in_article_header and _ARTICLE_HEADER_ONLY.match(out[-1])
            aligned = title_x is not None and abs(x0 - title_x) <= _TITLE_ALIGN_TOLERANCE
            dangling = bool(in_article_header and out and _DANGLING_CONNECTORS.search(out[-1].strip()))
            continues_title = in_article_header and not new_block and (bare_header or aligned or dangling)

            if continues_title or (out and is_heading and prev_heading and not _STRUCTURAL.match(text)):
                if bare_header:
                    title_x = x0  # title placed beside a bare "មាត្រា N.-": later lines align with it
                # Article titles are Khmer phrases; heading names follow "ជំពូកទោល"/"គន្ថីទី ១" with a space
                out[-1] = LimonPdfExtractor._append(out[-1], text, khmer_aware=in_article_header)
                is_heading = True
            elif out and not (is_heading or prev_heading or _LIST_ITEM.match(text)):
                out[-1] = LimonPdfExtractor._append(out[-1], text, khmer_aware=True)
                in_article_header, title_x = False, None
            else:
                out.append(text)
                in_article_header = bool(_ARTICLE_HEADER.match(text))
                title_x = line_title_x if in_article_header else None
            prev_heading = is_heading
        return "\n".join(out)

    @staticmethod
    def _append(left: str, right: str, khmer_aware: bool) -> str:
        """
        Join a wrapped line to the previous one.

        A hyphen at a mid-word break is dropped. With khmer_aware, a break between two Khmer
        letters joins without a space, since Khmer does not separate words with spaces.
        """
        if _KHMER_LETTER_START.match(right):
            if _HYPHEN_BREAK.search(left):
                return left[:-1] + right
            if khmer_aware and _KHMER_LETTER_END.search(left):
                return left + right
        if left.endswith(" ") or right.startswith(" "):
            return left + right
        return left + " " + right
