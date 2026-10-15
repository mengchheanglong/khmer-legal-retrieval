"""Update the canonical defense PDF while preserving the supplied slide design.

The archived user PDF is the visual source; new evidence pages use saved metrics.
Run from the repository root: python slides/update_defense.py
"""
from pathlib import Path
from io import BytesIO
import csv
import pymupdf
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent.parent
SLIDES = ROOT / 'slides'
NAVY = '#17384e'
TEAL = '#09888b'
GRAY = '#526575'


def evidence_page(title: str, subtitle: str, rows: list[list[str]], widths: list[int], notes: list[str]) -> pymupdf.Document:
    """Render an evidence table as a vector PDF page with selectable text."""
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=(960, 540))
    c.setFillColor(TEAL); c.setFont('BodyBold', 10)
    c.drawString(52, 510, 'EXPERIMENTAL EVIDENCE')
    c.setFillColor(NAVY); c.setFont('BodyBold', 25)
    c.drawString(52, 473, title)
    c.setFillColor(GRAY); c.setFont('Body', 12)
    c.drawString(52, 447, subtitle)
    y = 397
    for n, row in enumerate(rows):
        x = 52
        c.setFont('BodyBold' if n == 0 else 'Body', 14)
        c.setFillColor(TEAL if n == 0 else NAVY)
        for value, width in zip(row, widths):
            c.drawString(x, y, value)
            x += width
        y -= 39
    c.setFont('Body', 12); c.setFillColor(GRAY)
    for note in notes:
        c.drawString(52, y - 13, note); y -= 21
    c.showPage(); c.save()
    return pymupdf.open(stream=buf.getvalue(), filetype='pdf')


def replace_text(page: pymupdf.Page, old: str, new: str, size: float = 11) -> None:
    """Replace a single source phrase without altering other slide content."""
    matches = page.search_for(old)
    if not matches:
        raise ValueError(f'Missing text: {old}')
    rect = matches[0]
    page.add_redact_annot(rect, fill=(1,1,1))
    page.apply_redactions(images=0, graphics=0)
    box = pymupdf.Rect(rect.x0, rect.y0 - 1, 908, rect.y1 + 13)
    if page.insert_textbox(box, new, fontsize=size, fontname='helv', color=(.09,.22,.30)) < 0:
        raise ValueError(f'Text did not fit: {new}')


def main() -> None:
    """Build the sole current deck from archived design and saved evidence."""
    pdfmetrics.registerFont(TTFont('Body', 'C:/Windows/Fonts/arial.ttf'))
    pdfmetrics.registerFont(TTFont('BodyBold', 'C:/Windows/Fonts/arialbd.ttf'))
    source = pymupdf.open(SLIDES / 'archive/Khmer_Legal_Retrieval_Defense_original.pdf')
    replace_text(source[2], 'Legacy Limon fonts made ordinary text extraction unreliable.',
                 'Open Development Cambodia | Official Khmer codes | CC-BY-SA-4.0 | Limon font conversion')
    replace_text(source[3], 'So we used the article structure itself to create training examples.',
                 'Article titles provide weak supervision; 100 Civil and 100 Criminal questions form the main test set.')
    replace_text(source[9], 'Error categories among the 86 A3 questions with no relevant article in the top five.',
                 'Heuristic categories for 86 A3 misses; vocabulary mismatch is a residual label, not a proven cause.', 10)
    replace_text(source[9], 'Scope: Civil + Criminal Codes only · This is a research prototype, not a legal-advice system.',
                 'Limits: two codes, 200 questions, one seed; manual extraction error rate still unmeasured.', 10)
    replace_text(source[13], 'Best reported setting: LR 2×10⁻⁵, WD 0.01 · 5 epochs · Tesla T4',
                 'Selected: LR 2e-5, WD 0.01; tied at reported precision with WD 0.00. Best epoch 4 of 5.', 11)
    metrics = list(csv.DictReader((ROOT / 'results/metrics/summary.csv').open(encoding='utf8')))
    names = ['BM25', 'E5 zero-shot', 'A1 BiLSTM', 'A2 XLM-R probe', 'A3 XLM-R fine-tune', 'A4 PrahokBART']
    table = [['System', 'Recall@1', 'Recall@5', 'MRR@10', 'Hit@5']]
    for name, row in zip(names, metrics):
        table.append([name] + [f"{float(row[k]):.4f}" for k in ['tq_recall@1','tq_recall@5','tq_mrr@10','tq_hit@5']])
    results = evidence_page('Results on the same 200 Khmer questions',
        'All 1,976 articles are candidates. No statute filter is applied in the benchmark.', table,
        [285,135,135,135,135], ['Recall@5: mean fraction of relevant articles recovered; Hit@5: any relevant article found.',
        'MRR@10: reciprocal rank of the first relevant article, averaged over questions.',
        'Source: results/metrics/summary.csv. BM25 and zero-shot E5 are additional baselines.'])
    runtime = [['Approach','Trainable params','Training time','Hardware']]
    for name,row in zip(names[2:],metrics[2:]):
        runtime.append([name,f"{float(row['trainable_params'])/1e6:.2f}M",f"{float(row['train_time_sec']):.1f} s",row['hardware']])
    setup = evidence_page('Experimental setup and compute',
        'Fixed seed 42; shared splits and evaluation; model selection uses validation MRR@10.',
        runtime,[270,190,165,230],['A3: AdamW; LR {1e-5, 2e-5, 3e-5} x weight decay {0, 0.01}; 5 epochs.',
        'Batch 16; temperature 0.05; max length 256; warmup then cosine learning-rate schedule.',
        'A2 time covers the head training on cached embeddings; excludes backbone encoding.',
        'Times use different hardware and are not a controlled speed comparison.',
        'Sources: results/metrics/summary.csv and results/tuning/a3.csv.'])
    out=pymupdf.open()
    for i in range(len(source)):
        if i==6: out.insert_pdf(setup)
        if i==8: out.insert_pdf(results)
        out.insert_pdf(source,from_page=i,to_page=i)
    for i,page in enumerate(out):
        if i==0: continue
        # Re-number every footer to avoid stale 12/16-page labels.
        page.add_redact_annot(pymupdf.Rect(795,510,930,534),fill=(1,1,1))
        page.apply_redactions(images=0,graphics=0)
        label=f'{i+1:02} / 14' if i<14 else f'APPENDIX {i-13} | {i+1}/18'
        page.insert_textbox(pymupdf.Rect(785,516,908,533),label,fontsize=8,fontname='helv',align=2,color=(.09,.22,.30))
    out.save(SLIDES/'Khmer_Legal_Retrieval_Defense.pdf',garbage=4,deflate=True)
    print(f'Updated canonical deck: {len(out)} pages (14 main + 4 appendix)')


if __name__=='__main__':
    main()
