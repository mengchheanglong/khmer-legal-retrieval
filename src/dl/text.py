"""
Khmer text normalization and word segmentation utilities for deep learning.

Implements:
1. Unicode NFC normalization.
2. Zero-width character stripping (U+200B, U+200C, U+200D, U+FEFF).
3. Whitespace normalization.
4. Cached word segmentation with `khmer-nltk` for BM25 and BiLSTM (A1).
"""

from functools import lru_cache
import re
import unicodedata
from typing import Optional

# Zero-width regex pattern
_ZERO_WIDTH_CHARS = re.compile(r"[\u200b\u200c\u200d\ufeff]")
_MULTI_SPACE = re.compile(r"[ \t]+")


def normalize_khmer_text(text: Optional[str]) -> str:
    """Normalize Khmer text to canonical Unicode NFC form and clean artifacts.

    Applies:
    1. Unicode NFC canonical decomposition followed by canonical composition.
    2. Removal of zero-width spaces (U+200B), non-joiners (U+200C), joiners (U+200D),
       and zero-width no-break spaces (U+FEFF).
    3. Collapse of redundant consecutive whitespace.

    Args:
        text: Raw input string or None.

    Returns:
        Normalized clean string. Returns empty string if input is None or empty.
    """
    if not text:
        return ""

    # 1. Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", text)

    # 2. Strip zero-width characters
    normalized = _ZERO_WIDTH_CHARS.sub("", normalized)

    # 3. Collapse redundant spaces and strip surrounding whitespace
    normalized = _MULTI_SPACE.sub(" ", normalized).strip()

    return normalized


@lru_cache(maxsize=32768)
def _tokenize_cached(normalized_text: str, remove_spaces: bool = True) -> tuple[str, ...]:
    """Internal LRU-cached tokenization function.

    Takes already normalized text and returns an immutable tuple of tokens.

    Args:
        normalized_text: Clean, NFC-normalized text.
        remove_spaces: Whether to discard whitespace tokens.

    Returns:
        Tuple of segmented token strings.
    """
    if not normalized_text:
        return ()

    import khmernltk

    raw_tokens = khmernltk.word_tokenize(normalized_text)

    if remove_spaces:
        return tuple(t.strip() for t in raw_tokens if t.strip())
    return tuple(raw_tokens)


def tokenize_khmer(
    text: Optional[str],
    normalize: bool = True,
    remove_spaces: bool = True,
) -> list[str]:
    """Segment Khmer text into word tokens using khmer-nltk with LRU caching.

    Args:
        text: Input Khmer text.
        normalize: Whether to apply normalize_khmer_text before tokenizing (default: True).
        remove_spaces: Whether to omit whitespace tokens from the output (default: True).

    Returns:
        List of segmented word tokens.
    """
    if not text:
        return []

    clean_text = normalize_khmer_text(text) if normalize else text
    if not clean_text:
        return []

    return list(_tokenize_cached(clean_text, remove_spaces=remove_spaces))


def detokenize_khmer(tokens: list[str]) -> str:
    """Join segmented Khmer tokens into a space-delimited string.

    Args:
        tokens: List of token strings.

    Returns:
        Space-separated string of tokens.
    """
    return " ".join(t for t in tokens if t)
