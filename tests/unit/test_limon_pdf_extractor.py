"""Unit tests for line classification and re-joining in the Limon PDF extractor."""

import pytest

from src.infrastructure.extractors.limon_pdf_extractor import LimonPdfExtractor

# x0 positions (pt) measured in the ODC Khmer PDFs
CIVIL_MARGIN, CIVIL_INDENT, CIVIL_TITLE = 57.0, 99.0, 156.0
CRIMINAL_INDENT, CRIMINAL_TITLE = 135.0, 186.0


@pytest.mark.parametrize(
    ("text", "font", "expected"),
    [
        ("គន្ថីទី ១", "JAHKKG+LimonR1", True),
        # Article header whose title (the longest span) is set in the body font
        ("មាត្រា ៣៣៦.- ការបង្កើតកិច្ចសន្យាដោយសំណើ និង ស្វីការ", "JAHKHM+LimonS5", True),
        ("ក្រមនេះ យោងតាមមាត្រា ៣៣៦ នៃក្រមនេះ ។", "JAHKHM+LimonS5", False),
        ("មាត្រា ៣៣៦ នៃក្រមនេះ ត្រូវអនុវត្ត ។", "JAHKHM+LimonS5", False),  # cross-reference
    ],
)
def test_is_heading(text: str, font: str, expected: bool) -> None:
    assert LimonPdfExtractor._is_heading(text, font) is expected


def test_title_x_is_first_text_span_after_dash() -> None:
    """Criminal Code header spans: 'maRta 1' '>' '-' ' ' 'title'."""
    spans = [
        {"text": "maRta 1", "bbox": (99, 0, 151, 10)},
        {"text": ">", "bbox": (151, 0, 155, 10)},
        {"text": "-", "bbox": (155, 0, 165, 10)},
        {"text": " ", "bbox": (165, 0, 174, 10)},
        {"text": "EdnGnuvtþ", "bbox": (174, 0, 345, 10)},
    ]
    assert LimonPdfExtractor._title_x(spans) == 174
    assert LimonPdfExtractor._title_x(spans[:3]) is None  # bare header, no title span


def test_join_lines_rebuilds_headings_and_paragraphs() -> None:
    """Headings stay on their own lines; wrapped body lines and heading names are joined."""
    lines = [
        ("គន្ថីទី ១", True, 286.0, None),
        ("បទប្បញ្ញត្តិទូទៅ", True, 270.0, None),  # heading name on the next line
        ("ជំពូកទោល", True, 280.0, None),
        ("មាត្រា ១.- ច្បាប់ទូទៅនៃនីតិឯកជន", True, CIVIL_MARGIN, CIVIL_TITLE),
        ("ក្រមនេះ បញ្ញត្តអំពីទំនាក់ទំនងទ្រព្យ", False, CIVIL_INDENT, None),
        ("សម្បត្តិ និង ទំនាក់ទំនងញាតិ ។", False, CIVIL_MARGIN, None),  # mid-word wrap: no space
        ("១-កិច្ចសន្យា", False, CIVIL_INDENT, None),                  # numbered item: new line
        ("មានអានុភាព ។", False, CIVIL_MARGIN, None),
        ("ពី ១", False, CIVIL_MARGIN, None),
        ("(មួយ) ខែ", False, CIVIL_MARGIN, None),                     # not letter-to-letter: space
    ]
    assert LimonPdfExtractor._join_lines(lines) == (
        "គន្ថីទី ១ បទប្បញ្ញត្តិទូទៅ\n"
        "ជំពូកទោល\n"
        "មាត្រា ១.- ច្បាប់ទូទៅនៃនីតិឯកជន\n"
        "ក្រមនេះ បញ្ញត្តអំពីទំនាក់ទំនងទ្រព្យសម្បត្តិ និង ទំនាក់ទំនងញាតិ ។\n"
        "១-កិច្ចសន្យាមានអានុភាព ។ ពី ១ (មួយ) ខែ"
    )


def test_join_lines_keeps_wrapped_article_title_in_header() -> None:
    """Civil Code layout: a line aligned with the title text belongs to the header."""
    lines = [
        ("មាត្រា ២៣២.- ការកាន់កាប់ដោយមានឆន្ទៈយកជាកម្មសិទ្ធិ និង ការកាន់កាប់ដោយ", True, CIVIL_MARGIN, CIVIL_TITLE),
        ("គ្មានឆន្ទៈយកជាកម្មសិទ្ធិ", False, CIVIL_TITLE, None),
        ("១-ការកាន់កាប់ មាន ការកាន់កាប់ដោយមានឆន្ទៈ ។", False, CIVIL_INDENT, None),
    ]
    assert LimonPdfExtractor._join_lines(lines) == (
        "មាត្រា ២៣២.- ការកាន់កាប់ដោយមានឆន្ទៈយកជាកម្មសិទ្ធិ និង ការកាន់កាប់ដោយគ្មានឆន្ទៈយកជាកម្មសិទ្ធិ\n"
        "១-ការកាន់កាប់ មាន ការកាន់កាប់ដោយមានឆន្ទៈ ។"
    )


def test_join_lines_indented_body_is_not_title() -> None:
    """Criminal Code layout: the body's first line is indented (135pt) but not aligned with the title."""
    lines = [
        ("មាត្រា ១.- ដែនអនុវត្តនៃច្បាប់ព្រហ្មទណ្ឌ", True, 99.0, 174.0),
        ("ច្បាប់ព្រហ្មទណ្ឌកំណត់បទល្មើស", False, CRIMINAL_INDENT, None),
        ("ប្រកាសថា ជាអ្នកទទួលខុសត្រូវ ។", False, 99.0, None),
        ("មាត្រា ៣០.- និយមន័យនៃអ្នករាជការ", True, 99.0, CRIMINAL_TITLE),
        ("សាធារណៈដោយការបោះឆ្នោត", False, 184.0, None),  # aligned with the title: continuation
        ("១-អ្នករាជការសាធារណៈ មានន័យថា ៖", False, CRIMINAL_INDENT, None),
    ]
    assert LimonPdfExtractor._join_lines(lines) == (
        "មាត្រា ១.- ដែនអនុវត្តនៃច្បាប់ព្រហ្មទណ្ឌ\n"
        "ច្បាប់ព្រហ្មទណ្ឌកំណត់បទល្មើសប្រកាសថា ជាអ្នកទទួលខុសត្រូវ ។\n"
        "មាត្រា ៣០.- និយមន័យនៃអ្នករាជការសាធារណៈដោយការបោះឆ្នោត\n"
        "១-អ្នករាជការសាធារណៈ មានន័យថា ៖"
    )


def test_join_lines_attaches_title_set_beside_bare_article_header() -> None:
    """'មាត្រា N.-' with its title (and title continuation) as separate lines forms one header."""
    lines = [
        ("មាត្រា ១១៨១.-", True, CIVIL_MARGIN, None),
        ("បទប្បញ្ញត្តិពិសេសចំពោះមតកសាសន៍របស់ជនជាតិខ្មែរដែលរស់", False, 176.0, None),
        ("នៅបរទេស", False, 176.0, None),
        ("បើជនជាតិខ្មែរមានបំណង ។", False, CIVIL_INDENT, None),
    ]
    assert LimonPdfExtractor._join_lines(lines) == (
        "មាត្រា ១១៨១.- បទប្បញ្ញត្តិពិសេសចំពោះមតកសាសន៍របស់ជនជាតិខ្មែរដែលរស់នៅបរទេស\n"
        "បើជនជាតិខ្មែរមានបំណង ។"
    )


def test_join_lines_removes_hyphen_at_mid_word_break() -> None:
    """'អាណា-' at a line end followed by 'ព្យាបាល' becomes 'អាណាព្យាបាល'."""
    lines = [
        ("មាត្រា ១៦.- អត្ថន័យ", True, CIVIL_MARGIN, CIVIL_TITLE),
        ("សំដៅទៅលើជននៅក្រោមអាណា-", False, CIVIL_INDENT, None),
        ("ព្យាបាលទូទៅ ។", False, CIVIL_MARGIN, None),
    ]
    assert LimonPdfExtractor._join_lines(lines) == (
        "មាត្រា ១៦.- អត្ថន័យ\n"
        "សំដៅទៅលើជននៅក្រោមអាណាព្យាបាលទូទៅ ។"
    )


def test_append_keeps_single_letter_list_marker() -> None:
    """A bare list marker like 'ក-' is not a hyphenated word break."""
    assert LimonPdfExtractor._append("ក-", "ការទទួល", khmer_aware=True) == "ក- ការទទួល"


def test_times_new_roman_whitespace_span_does_not_split_subscript() -> None:
    """Criminal Code p.10: a placeholder space between touching Limon glyphs must not split the word."""
    spans = [
        {"font": "JAHKHM+LimonS5", "text": "Ktiyut", "bbox": (300.0, 0, 340.0, 10)},  # គតិយុត
        {"font": "TimesNewRoman", "text": " ", "bbox": (340.0, 0, 346.0, 10)},        # placeholder
        {"font": "JAHKHM+LimonS5", "text": "\xfe", "bbox": (339.9, 0, 346.0, 10)},    # subscript, gap ≈ 0
    ]
    result = LimonPdfExtractor._line_text(spans)
    assert result == "គតិយុត្ត"
    assert " " not in result


def test_non_limon_space_with_visible_gap_is_a_real_word_space() -> None:
    """Criminal Code p.4: a whitespace span with a visible gap (2.5pt+) is a real space between words."""
    spans = [
        {"font": "JAHKHM+LimonS5", "text": "RkmenH", "bbox": (100.0, 0, 140.0, 10)},  # ក្រមនេះ
        {"font": "ArialUnicodeMS", "text": " ", "bbox": (140.0, 0, 142.5, 10)},
        {"font": "JAHKHM+LimonS5", "text": "minGac", "bbox": (142.7, 0, 180.0, 10)},  # មិនអាច
    ]
    assert LimonPdfExtractor._line_text(spans) == "ក្រមនេះ មិនអាច"


def test_join_lines_criminal_three_digit_title_wrapped() -> None:
    """Criminal Code 3-digit article titles wrap to x0 ≈ 184pt; tolerance 16pt must join them."""
    lines = [
        ("មាត្រា ២៧០.- បទបដិសេធក្នុងការផ្តល់សិទ្ធិដោយអ្នករាជការសាធារណៈដោយ", True, 99.2, 197.5),
        ("មូលហេតុនៃការរើសអើង", False, 184.3, None),
        ("ត្រូវផ្តន្ទាទោសដាក់ពន្ធនាគារពី ៦ (ប្រាំមួយ) ខែ ។", False, 135.2, None),
    ]
    assert LimonPdfExtractor._join_lines(lines) == (
        "មាត្រា ២៧០.- បទបដិសេធក្នុងការផ្តល់សិទ្ធិដោយអ្នករាជការសាធារណៈដោយមូលហេតុនៃការរើសអើង\n"
        "ត្រូវផ្តន្ទាទោសដាក់ពន្ធនាគារពី ៦ (ប្រាំមួយ) ខែ ។"
    )


def test_join_lines_civil_art_676_unindented_title() -> None:
    """Civil Art 676 line 2 is at margin 56.7pt following 'អំពី'; must join to the title."""
    lines = [
        ("មាត្រា ៦៧៦.- ការសម្រាលការទទួលខុសត្រូវក្នុងករណីដែលមិនបានរាយការណ៍អំពី", True, 56.7, 143.9),
        ("វត្ថុដែលមានតម្លៃខ្ពស់", False, 56.7, None),
        ("នៅក្នុងករណីដែលអ្នកផ្ញើបានផ្ញើប្រាក់ ។", False, 98.6, None),
    ]
    assert LimonPdfExtractor._join_lines(lines) == (
        "មាត្រា ៦៧៦.- ការសម្រាលការទទួលខុសត្រូវក្នុងករណីដែលមិនបានរាយការណ៍អំពីវត្ថុដែលមានតម្លៃខ្ពស់\n"
        "នៅក្នុងករណីដែលអ្នកផ្ញើបានផ្ញើប្រាក់ ។"
    )


def test_line_text_collapses_multiple_spaces() -> None:
    """Multiple consecutive horizontal spaces must collapse to a single space."""
    spans = [
        {"font": "JAHKKG+LimonR1", "text": "maRta 1>-"},
        {"font": "JAHKHM+LimonS5", "text": "  cMNgeCIg"},
    ]
    result = LimonPdfExtractor._line_text(spans)
    assert "  " not in result
    assert result == "មាត្រា ១.- ចំណងជើង"
