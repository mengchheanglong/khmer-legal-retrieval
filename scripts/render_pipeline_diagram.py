"""Render high-resolution, perfectly legible end-to-end system pipeline diagram."""
from pathlib import Path
from playwright.sync_api import sync_playwright

repo_root = Path(__file__).resolve().parent.parent
output_png = repo_root / "results" / "figures" / "system_pipeline.png"

mermaid_code = """
flowchart TD
    subgraph Data["<b>Stage 1: Corpus Ingestion & Font Normalization Pipeline</b>"]
        direction LR
        PDF["<b>Official Khmer PDFs</b><br>Civil Code (2007) & Criminal Code (2009)"]
        Extractor["<b>LimonPdfExtractor</b><br>Page Filtering & Syllable Engine"]
        Converter["<b>LimonConverter</b><br>Unicode Mapping & Syllable Reordering"]
        Chunker["<b>LegalHierarchicalChunker</b><br>Book → Title → Chapter → Article"]
        Corpus["<b>1,976 Statutory Articles</b><br>Canonical Khmer Corpus (data/04_chunks/)"]

        PDF --> Extractor --> Converter --> Chunker --> Corpus
    end

    subgraph Splits["<b>Stage 2: Deterministic Leakage-Guarded Splits (Seed 42)</b>"]
        direction LR
        Guard["<b>Leakage Guard</b><br>Sequester 261 cited articles"]
        TrainPairs["<b>Train: 1,176 pairs</b><br>Val: 147 pairs (80/10/10)"]
        TT["<b>Held-Out Titles (T-T)</b><br>148 Articles (In-Distribution)"]
        TQ["<b>Primary Questions (T-Q)</b><br>200 Human Questions (OOD)"]

        Guard -->|"Exclude 261 Cited"| TrainPairs
        Guard -->|"Held-Out Titles"| TT
        Guard -->|"Human Questions"| TQ
    end

    subgraph DL["<b>Stage 3: PyTorch Deep Learning Retrievers (InfoNCE, &tau; = 0.05)</b>"]
        direction LR
        Loss["<b>InfoNCE Loss</b><br>In-Batch Negatives"]
        A3["<b>A3: XLM-R Full Fine-Tune 🏆</b><br>All 12 layers · 278.04M params · Tesla T4 (Winner)"]
        A2["<b>A2: XLM-R Linear Probe</b><br>Frozen backbone · 0.59M params"]
        A4["<b>A4: PrahokBART Dual Encoder</b><br>Khmer pretrained · 35.83M params"]
        A1["<b>A1: BiLSTM Dual Encoder</b><br>From scratch · 1.97M params"]

        Loss --> A3
        Loss --> A2
        Loss --> A4
        Loss --> A1
    end

    subgraph Bottom[" "]
        direction LR
        subgraph App["<b>Stage 5: Citation-Grounded Assistant Prototype</b>"]
            direction TB
            UserQ["<b>Citizen Legal Query</b><br>Natural language question in Khmer"]
            Retriever["<b>Retriever Module</b><br>A3 Dense (Winning Model) or BM25"]
            RRF["<b>Reciprocal Rank Fusion (RRF)</b><br>Optional Hybrid Ranking (Dense + Sparse)"]
            LLM["<b>DeepSeek (deepseek-chat)</b><br>Statutory-conditioned Khmer generation"]
            Verify["<b>Citation Verifier</b><br>Validates Article citations in context"]
            Answer["<b>Grounded Legal Answer</b><br>Verified citations & source articles"]

            UserQ --> Retriever --> RRF --> LLM --> Verify --> Answer
        end

        subgraph Eval["<b>Stage 4: Dual Benchmark Evaluation & Reporting</b>"]
            direction TB
            Harness["<b>Shared Evaluation Harness</b><br>Recall@k, MRR@10, Hit@5 · 95% Paired Bootstrap CIs"]
            Tables["<b>Leaderboard & Significance</b><br>summary.csv · statistical_significance.md"]
            Plots["<b>Publication Figures</b><br>primary_metrics_bar · learning_curves"]

            Harness --> Tables
            Harness --> Plots
        end
    end

    %% Clean Inter-Stage Routing
    Corpus --> Guard
    TrainPairs --> Loss
    Corpus ==>|"Canonical Statutory Context"| Retriever
    A3 ==>|"Fine-Tuned Weights"| Retriever
    TQ --> Harness
    TT --> Harness
    A3 --> Harness
    A2 --> Harness
    A4 --> Harness
    A1 --> Harness
"""

html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 30px;
            background-color: #ffffff;
            font-family: 'Leelawadee UI', 'Khmer OS', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            justify-content: center;
        }}
        #container {{
            background: #ffffff;
            padding: 10px;
        }}
        .node rect, .node polygon {{
            stroke-width: 1.5px !important;
            rx: 6px !important;
            ry: 6px !important;
        }}
        .cluster rect {{
            stroke-width: 1.8px !important;
            rx: 8px !important;
            ry: 8px !important;
            fill: #f8fafc !important;
            stroke: #cbd5e1 !important;
        }}
        .cluster text {{
            font-size: 15px !important;
            font-weight: 700 !important;
            fill: #0f172a !important;
        }}
        .edgeLabel {{
            background-color: #ffffff !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            color: #334155 !important;
            padding: 3px 6px !important;
            border-radius: 4px !important;
            border: 1px solid #cbd5e1 !important;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div class="mermaid">
{mermaid_code}
        </div>
    </div>
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'base',
            themeVariables: {{
                fontFamily: "'Leelawadee UI', 'Khmer OS', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
                fontSize: '13px',
                primaryColor: '#f8fafc',
                primaryBorderColor: '#64748b',
                primaryTextColor: '#0f172a',
                secondaryColor: '#eff6ff',
                secondaryBorderColor: '#3b82f6',
                secondaryTextColor: '#1e3a8a',
                tertiaryColor: '#f0fdf4',
                tertiaryBorderColor: '#22c55e',
                tertiaryTextColor: '#14532d',
                lineColor: '#64748b',
                textColor: '#0f172a'
            }},
            flowchart: {{
                curve: 'basis',
                nodeSpacing: 30,
                rankSpacing: 40,
                htmlLabels: true
            }}
        }});
    </script>
</body>
</html>
"""

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=2.0)
        page.set_content(html_content)
        page.wait_for_selector(".mermaid svg", timeout=20000)
        svg_element = page.query_selector(".mermaid svg")
        if not svg_element:
            raise RuntimeError("Mermaid SVG failed to render")
        output_png.parent.mkdir(parents=True, exist_ok=True)
        svg_element.screenshot(path=str(output_png))
        browser.close()
    print(f"Successfully generated clean pipeline diagram: {output_png} ({output_png.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
