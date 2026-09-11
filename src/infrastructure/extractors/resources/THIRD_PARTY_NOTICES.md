# Third-Party Notices — Khmer legacy font conversion

## KhmerConverter 1.5.1

- **Copyright:** (c) 2006–2008 Khmer Software Initiative (www.khmeros.info).
  Developed by Keo Sophon, San Titvirak, Hok Kakada, Seth Chanratha.
- **Source:** https://github.com/keosophon/KhmerConverter (mirror: https://github.com/olivierberten/KhmerConverter)

| File in this repository | What was reused | License |
|-------------------------|-----------------|---------|
| `khmer_legacy_fontdata.xml` | Unmodified copy of `modules/fontdata.xml` (legacy font ↔ Unicode glyph tables) | LGPL-2.1, per the project's `LICENSE.TXT`, included here as `KHMERCONVERTER_LICENSE.txt` |
| `../limon_converter.py` (reordering functions) | Python 3 port of the cluster reordering rules in `modules/unicodeReorder.py` | GPL-2.0-or-later, per that file's source header |

**Local change to the mapping data:** `khmer_legacy_fontdata.xml` itself is unmodified, but
`limon_converter.py` resolves its `&nbsp;` entity to a normal space (U+0020) instead of U+00A0.
The only affected mapping is Limon `0x20 0xB3` → " ៖"; a normal space keeps tokenization consistent.

**Licence note:** the KhmerConverter source headers reference the GNU GPL v2 or later, but the
project ships an LGPL-2.1 `LICENSE.TXT`. Until that is clarified, treat `limon_converter.py` as
GPL-2.0-or-later. Keep this in mind if a licence is chosen for the rest of this repository.
