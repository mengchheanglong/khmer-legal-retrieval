"""
Limon (legacy Khmer font) to Khmer Unicode converter.

Official Khmer PDFs on Open Development Cambodia, including the Civil Code (2007)
and Criminal Code (2009), are typeset in the legacy Limon fonts. Their text layer
stores Latin code points that the font draws as Khmer glyphs: "maRta 336>-" renders
as "មាត្រា ៣៣៦.-". The text must be converted to Unicode before it can be searched
or used to train models.

Conversion happens in two steps:
    1. Map legacy code sequences (longest first) to Unicode using the Limon table.
    2. Reorder each syllable from visual order into Unicode logical order. Limon types
       pre-base vowels (េ ែ ៃ) and COENG RO before the consonant they belong to.

Third-party attribution (see resources/THIRD_PARTY_NOTICES.md):
    Mapping data: KhmerConverter 1.5.1 fontdata.xml, (c) 2006-2008 Khmer Software
    Initiative, LGPL-2.1. The reordering functions below (_reorder, _read_cluster,
    _add_vowel, _assemble) are a Python 3 port of KhmerConverter's
    modules/unicodeReorder.py, GPL-2.0-or-later.
"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

FONTDATA_PATH = Path(__file__).parent / "resources" / "khmer_legacy_fontdata.xml"

_XML_ENTITIES = {
    "mark": "&#x17EA;", "coeng": "&#x17D2;", "zwsp": "&#x200B;",
    "zwj": "&#x200C;", "zwnj": "&#x200D;", "nbsp": " ",
}
# PDF text layers report the Limon glyph at code 0xB5 as GREEK SMALL LETTER MU
_PDF_CHAR_FIXES = {"μ": "\xb5"}


def is_limon_font(font_name: str) -> bool:
    """Return True for any Limon font variant, e.g. 'JAHKHM+LimonS5' or 'Limon R1'."""
    return "limon" in font_name.lower()


def limon_to_unicode(text: str) -> str:
    """
    Convert text extracted from Limon-font PDF spans into Khmer Unicode.

    Args:
        text: Characters as reported by the PDF text layer for Limon-font spans.
            Pass contiguous Limon spans together so syllables are not split.

    Returns:
        Khmer Unicode text in logical order.
    """
    return _reorder(_map_legacy(_to_legacy_codes(text))).replace("\xa0", " ")


# ── Step 1: legacy code mapping ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_mapping(font_type: str = "limon") -> tuple[dict[str, str], int]:
    """
    Load legacy→Unicode mappings for a font and all of its ancestors.

    Returns:
        (mapping, longest key length). Keys are legacy code sequences as latin-1 chars.
    """
    xml = FONTDATA_PATH.read_text(encoding="utf-8")
    xml = re.sub(r"<!DOCTYPE.*?\]>", "", xml, flags=re.DOTALL)
    for name, ref in _XML_ENTITIES.items():
        xml = xml.replace(f"&{name};", ref)
    fonts = {font.get("type"): font for font in ET.fromstring(xml).iter("font")}

    chain = []
    current = font_type
    while current:  # e.g. limon → limon parent → common font → digit
        chain.append(fonts[current])
        current = fonts[current].get("inherit")

    mapping: dict[str, str] = {}
    for font in reversed(chain):  # ancestors first, so child fonts override them
        for section in ("maps/global/map", "maps/tounicode/map"):
            for entry in font.findall(section):
                if entry.get("legacy"):
                    key = "".join(chr(int(code, 16)) for code in entry.get("legacy").split(";"))
                    mapping[key] = entry.get("unicode")
    return mapping, max(len(key) for key in mapping)


def _to_legacy_codes(text: str) -> str:
    """Recover the font's single-byte codes from the Unicode reported by the PDF."""
    codes = []
    for ch in text:
        ch = _PDF_CHAR_FIXES.get(ch, ch)
        try:
            codes.append(chr(ch.encode("cp1252")[0]))
        except UnicodeEncodeError:
            codes.append(ch)
    return "".join(codes)


def _map_legacy(codes: str) -> str:
    """Replace legacy code sequences with Unicode, longest match first (still visual order)."""
    mapping, max_len = _load_mapping()
    out, i = [], 0
    while i < len(codes):
        for length in range(min(max_len, len(codes) - i), 0, -1):
            unicode_text = mapping.get(codes[i:i + length])
            if unicode_text is not None:
                out.append(unicode_text)
                i += length
                break
        else:
            out.append(codes[i])
            i += 1
    return "".join(out)


# ── Step 2: cluster reordering (port of KhmerConverter unicodeReorder.py) ───

_BASE, _VOWEL, _SHIFTER, _COENG, _SIGN, _LEFT, _WITHE, _WITHU = 1, 2, 4, 8, 16, 32, 64, 128
_POSRAA, _MUUS, _TRII, _ROBAT = 256, 512, 1024, 2048

_RO, _PO, _NYO, _ZWSP = "រ", "ព", "ញ", "​"
_SRAAA, _SRAE, _SRAOE, _SRAOO = "ា", "េ", "ើ", "ោ"
_SRAYA, _SRAIE, _SRAAU, _SRAII, _SRAU = "ឿ", "ៀ", "ៅ", "ី", "ុ"
_TRIISAP, _MUUSIKATOAN, _SAMYOKSANNYA = "៊", "៉", "័"
# SRA E typed before a consonant combines with a following vowel part into one vowel
_SRA_E_COMBINING = {_SRAII: _SRAOE, _SRAYA: _SRAYA, _SRAIE: _SRAIE, _SRAAA: _SRAOO, _SRAAU: _SRAAU}

_KHMER_TYPES = [_BASE] * 52  # U+1780..U+17B3: consonants and independent vowels
for _offset, _flag in {0x09: _MUUS, 0x0F: _POSRAA, 0x13: _POSRAA, 0x14: _MUUS, 0x17: _POSRAA,
                       0x19: _POSRAA, 0x1A: _POSRAA, 0x1B: _POSRAA, 0x1C: _POSRAA,
                       0x1F: _TRII, 0x20: _TRII, 0x22: _TRII}.items():
    _KHMER_TYPES[_offset] |= _flag
_KHMER_TYPES += [0, 0]  # U+17B4, U+17B5
_KHMER_TYPES += [  # U+17B6..U+17D3
    _VOWEL | _WITHE | _WITHU, _VOWEL | _WITHU, _VOWEL | _WITHE | _WITHU, _VOWEL | _WITHU,
    _VOWEL | _WITHU, _VOWEL, _VOWEL, _VOWEL, _VOWEL | _WITHU, _VOWEL | _WITHE, _VOWEL | _WITHE,
    _VOWEL | _LEFT, _VOWEL | _LEFT, _VOWEL | _LEFT, _VOWEL, _VOWEL | _WITHE,
    _SIGN | _WITHU, _SIGN, _SIGN, _SHIFTER, _SHIFTER, _SIGN, _ROBAT, _SIGN, _SIGN, _SIGN,
    _SIGN | _WITHU, _SIGN, _COENG, _SIGN,
]
_SINGLE_SLOTS = ((_BASE, "base"), (_ROBAT, "robat"), (_SHIFTER, "shifter1"), (_SIGN, "sign"))


@dataclass
class _Cluster:
    """Parts of one Khmer syllable, collected in visual order."""

    base: str = ""
    robat: str = ""
    shifter1: str = ""
    shifter2: str = ""
    coeng1: str = ""
    coeng2: str = ""
    vowel: str = ""
    sign: str = ""
    keep: str = ""  # trailing non-Khmer character (space, punctuation)
    po_sraa: bool = False


def _khmer_type(ch: str) -> int:
    offset = ord(ch) - 0x1780
    return _KHMER_TYPES[offset] if 0 <= offset < len(_KHMER_TYPES) else 0


def _reorder(text: str) -> str:
    """Reorder visually ordered Khmer syllables into Unicode logical order."""
    result = []
    i = -1
    while i < len(text) - 1:
        cluster, i = _read_cluster(text, i)
        result.append(_assemble(cluster))
    return "".join(result)


def _read_cluster(text: str, i: int) -> tuple[_Cluster, int]:
    """Collect one syllable after index i. Returns it with the index of its last character."""
    c = _Cluster()
    last = len(text) - 1
    while i < last:
        i += 1
        ch = text[i]
        kind = _khmer_type(ch)
        slot = next((name for flag, name in _SINGLE_SLOTS if kind & flag), None)
        if slot:
            if getattr(c, slot):
                return c, i - 1  # a second base/robat/shifter/sign starts the next syllable
            setattr(c, slot, ch)
            c.keep = ""
        elif kind & _COENG:
            if i == last:
                c.coeng1 = ch
                return c, i
            if text[i + 1] == _RO and c.base:
                return c, i - 1  # COENG RO is typed before its consonant: next syllable
            if not c.coeng1:
                c.coeng1, c.keep = text[i:i + 2], ""
                i += 1
            elif len(c.coeng1) == 2 and c.coeng1[1] == _RO:
                c.coeng2, c.keep = text[i:i + 2], ""
                i += 1
            else:
                return c, i - 1
        elif kind & _VOWEL:
            if not _add_vowel(c, ch, kind):
                return c, i - 1
        elif ch == _ZWSP:
            c.keep = _ZWSP
        else:
            c.keep = ch
            return c, i
    return c, i


def _add_vowel(c: _Cluster, ch: str, kind: int) -> bool:
    """Merge a dependent vowel into the syllable. Returns False if it starts the next one."""
    if not c.vowel:
        if kind & _LEFT and c.base:
            return False  # pre-base vowel belongs to the following consonant
        c.vowel, c.keep = ch, ""
    elif c.base == _PO and not c.po_sraa and _SRAAA in (ch, c.vowel):
        c.po_sraa = True
        if c.vowel == _SRAAA:
            c.vowel, c.keep = ch, ""
    elif c.vowel == _SRAE and kind & _WITHE:
        c.vowel, c.keep = _SRA_E_COMBINING[ch], ""
    elif (c.vowel == _SRAU and kind & _WITHU) or (_khmer_type(c.vowel) & _WITHU and ch == _SRAU):
        if not _khmer_type(c.vowel) & _WITHU:
            c.vowel = ch
        c.shifter1 = _shifter_for(c.base)
    elif c.vowel == _SRAE and ch == _SRAU:
        c.shifter1 = _shifter_for(c.base)
    else:
        return False
    return True


def _shifter_for(base: str) -> str:
    return _TRIISAP if base and _khmer_type(base) & _TRII else _MUUSIKATOAN


def _assemble(c: _Cluster) -> str:
    """Emit base + robat + shifter + coengs + shifter + vowel + sign in Unicode order."""
    if c.vowel == _SRAU and c.sign == _SAMYOKSANNYA:
        c.vowel, c.shifter1 = "", _MUUSIKATOAN
    if c.shifter1 and len(c.coeng1) == 2:
        if _khmer_type(c.coeng1[1]) & _TRII:
            c.shifter2, c.shifter1 = _TRIISAP, ""
        elif _khmer_type(c.coeng1[1]) & _MUUS:
            c.shifter2, c.shifter1 = _MUUSIKATOAN, ""
    under = c.coeng2 or c.coeng1
    if len(under) == 2 and not _khmer_type(under[1]) & _POSRAA:
        # Limon draws NYO as PO + SRA AA; above most subscripts that pair means NYO
        if (c.po_sraa and c.vowel) or (c.base == _PO and c.vowel == _SRAAA):
            c.base = _NYO
            if c.vowel == _SRAAA and not c.po_sraa:
                c.vowel = ""
    if c.po_sraa and c.vowel == _SRAE:
        c.vowel = _SRAOO
    # With two coengs, coeng1 is always COENG RO, which comes last in Unicode
    return c.base + c.robat + c.shifter1 + c.coeng2 + c.coeng1 + c.shifter2 + c.vowel + c.sign + c.keep
