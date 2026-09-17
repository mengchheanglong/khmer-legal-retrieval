"""
Slide generation and PDF export script.

Reads slides/presentation.html and exports it to slides/final_presentation.pdf
using Microsoft Edge in headless mode.
"""

import subprocess
import sys
from pathlib import Path


def export_pdf_via_edge(html_path: Path, pdf_path: Path) -> bool:
    """Use Microsoft Edge headless to print the HTML presentation to PDF."""
    edge_paths = [
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    ]
    edge_bin = None
    for p in edge_paths:
        if p.exists():
            edge_bin = p
            break

    if not edge_bin:
        print("Microsoft Edge not found at standard paths. Skipping PDF rendering.")
        return False

    file_uri = html_path.resolve().as_uri()
    cmd = [
        str(edge_bin),
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path.resolve()}",
        "--no-pdf-header-footer",
        file_uri,
    ]

    print(f"Rendering PDF with Edge: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if pdf_path.exists():
            size_kb = pdf_path.stat().st_size / 1024
            print(f"Successfully generated PDF: {pdf_path} ({size_kb:.1f} KB)")
            return True
        else:
            print("Edge executed but PDF was not created.")
            return False
    except Exception as e:
        print(f"Failed to render PDF via Edge: {e}")
        return False


def main():
    root = Path(__file__).resolve().parent.parent
    slides_dir = root / "slides"
    html_path = slides_dir / "presentation.html"
    pdf_path = slides_dir / "final_presentation.pdf"

    if not html_path.exists():
        print(f"Error: {html_path} does not exist.")
        sys.exit(1)

    success = export_pdf_via_edge(html_path, pdf_path)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
