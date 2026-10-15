# Current presentation

Present **Khmer_Legal_Retrieval_Defense.pdf**: 14 main slides and 4 appendix slides.
It is based on the latest PDF supplied by the author, with missing experimental
setup/results evidence added and misleading claims corrected.

- `update_defense.py` reproducibly edits the supplied PDF and pulls tables from
  `results/metrics/summary.csv` without changing experiment results.
- `archive/` contains the original supplied PDF and all superseded HTML/PDF decks,
  generators, screenshots, and speaker notes. Do not present those old versions.
- See `../docs/DEFENSE_REVIEW.md` for a 10-minute rehearsal plan and demo steps.

To rebuild on this Windows machine: `python slides/update_defense.py`.
PDF editing dependencies: PyMuPDF and reportlab; the script uses Windows Arial fonts.
The source deck was supplied as PDF, so this is a PDF update rather than a PPTX conversion.
