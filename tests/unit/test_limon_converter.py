"""Unit tests for the Limon legacy-font to Khmer Unicode converter."""

import pytest

from src.infrastructure.extractors.limon_converter import _reorder, is_limon_font, limon_to_unicode

ZWSP = "​"

# Visual-order input → Unicode logical order, taken from KhmerConverter's own test suite
KSI_REORDER_CASES = [
    ("កករ", "កករ"),
    ("បា៉", "ប៉ា"),
    ("បូ៊", "ប៊ូ"),
    ("របំា", "របាំ"),
    ("្រកដាស្របដាល់កណ្ដាល", "ក្រដាសប្រដាល់កណ្ដាល"),
    ("បេង្គាល ខាងេលី េសៀវេភៅ", "បង្គោល ខាងលើ សៀវភៅ"),
    ("សីុបុីអុី", "ស៊ីប៉ីអ៊ី"),
    ("ន្សីុ", "ន្ស៊ី"),
    ("េគ្របែឡងគ្នា", "គេប្រឡែងគ្នា"),
    ("បពា្ញា", "បញ្ញា"),
    ("កេ្រព្ជាាង", "កញ្ជ្រោង"),
    ("្របឹក្សាធម្មនុពា្ញ", "ប្រឹក្សាធម្មនុញ្ញ"),
    ("បូពា៌", "បូព៌ា"),
    ("េ្របី", "ប្រើ"),
    ("ប្ប័ុង", "ប្ប៉័ង"),
    ("េសុីប", "ស៊ើប"),
    ("កំុេជា", "កុំជោ"),
    ("ប្រក", "បក្រ"),
    ("្រស្ត", "ស្ត្រ"),
    ("រ" + ZWSP + "ដ្ឋ" + ZWSP + "ាភិប" + ZWSP + "ាល", "រ" + ZWSP + "ដ្ឋាភិបាល"),
    ("្", "្"),
    ("this is english text", "this is english text"),
]

# Real Limon text-layer strings from the ODC Khmer Civil Code (2007) and Criminal Code (2009)
LIMON_CASES = [
    ("Rkmrdæb,evNI", "ក្រមរដ្ឋប្បវេណី"),
    ("RkmRBhµTNÐ", "ក្រមព្រហ្មទណ្ឌ"),
    ("RkmRBhμTNÐ", "ក្រមព្រហ្មទណ្ឌ"),  # same glyph reported as Greek mu
    ("KnßITI 1", "គន្ថីទី ១"),
    ("maRta 336>- karbegáItkic©snüaedaysMeNI nig sVIkar",
     "មាត្រា ៣៣៦.- ការបង្កើតកិច្ចសន្យាដោយសំណើ និង ស្វីការ"),
    ("RkmenH bBaØtþGMBIeKalkarN_TUeTA", "ក្រមនេះ បញ្ញត្តអំពីគោលការណ៍ទូទៅ"),
    ("1-kic©snüamanGanuPaBedaysarsMeNI nig sVIkarRtUvKña .",
     "១-កិច្ចសន្យាមានអានុភាពដោយសារសំណើ និង ស្វីការត្រូវគ្នា ។"),
]


@pytest.mark.parametrize(("visual", "expected"), KSI_REORDER_CASES)
def test_reorder_matches_khmerconverter_cases(visual: str, expected: str) -> None:
    assert _reorder(visual) == expected


@pytest.mark.parametrize(("limon", "expected"), LIMON_CASES)
def test_limon_text_converts_to_unicode(limon: str, expected: str) -> None:
    assert limon_to_unicode(limon) == expected


def test_empty_and_whitespace_pass_through() -> None:
    assert limon_to_unicode("") == ""
    assert limon_to_unicode(" \n") == " \n"


@pytest.mark.parametrize(
    ("font", "expected"),
    [("JAHKHM+LimonS5", True), ("Limon R1", True), ("Arial-ItalicMT", False), ("Calibri", False)],
)
def test_is_limon_font(font: str, expected: bool) -> None:
    assert is_limon_font(font) is expected


def test_colon_mapping_produces_standard_space() -> None:
    """The legacy mapping for space + colon (0x20 0xB3) must output standard space + '៖', not non-breaking space."""
    res = limon_to_unicode(" \xb3")
    assert res == " ៖"
    assert "\xa0" not in res

