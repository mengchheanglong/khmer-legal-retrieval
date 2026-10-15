"""
Slide generation and PDF export script.

Reads slides/presentation.html and slides/presentation_12slides.html,
exporting both 16-slide and 12-slide PDF versions via Playwright (or Edge fallback).
"""

import subprocess
import sys
from pathlib import Path


def export_pdf_via_playwright(html_path: Path, pdf_path: Path) -> bool:
    """Use Playwright to export exact 16:9 vector PDF."""
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 900})
            page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
            page.wait_for_timeout(1000)
            page.pdf(
                path=str(pdf_path.resolve()),
                width="16in",
                height="9in",
                print_background=True,
                margin={"top": "0px", "right": "0px", "bottom": "0px", "left": "0px"},
            )
            browser.close()
        size_kb = pdf_path.stat().st_size / 1024
        print(f"Generated PDF via Playwright: {pdf_path.name} ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"Playwright export failed ({e}), falling back to Edge...")
        return False


def export_pdf_via_edge(html_path: Path, pdf_path: Path) -> bool:
    """Fallback: Use Microsoft Edge headless to print HTML to PDF."""
    edge_paths = [
        Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
        Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
    ]
    edge_bin = next((p for p in edge_paths if p.exists()), None)
    if not edge_bin:
        print("Microsoft Edge not found. Skipping fallback.")
        return False

    cmd = [
        str(edge_bin),
        "--headless",
        "--disable-gpu",
        f"--print-to-pdf={pdf_path.resolve()}",
        "--no-pdf-header-footer",
        html_path.resolve().as_uri(),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if pdf_path.exists():
            size_kb = pdf_path.stat().st_size / 1024
            print(f"Generated PDF via Edge: {pdf_path.name} ({size_kb:.1f} KB)")
            return True
    except Exception as e:
        print(f"Edge export failed: {e}")
    return False


def main():
    root = Path(__file__).resolve().parent.parent
    slides_dir = root / "slides"

    decks = [
        ("presentation.html", "final_presentation.pdf"),
        ("presentation_12slides.html", "final_presentation_12slides.pdf"),
    ]

    for html_name, pdf_name in decks:
        html_path = slides_dir / html_name
        pdf_path = slides_dir / pdf_name
        if not html_path.exists():
            print(f"Warning: {html_path} not found. Skipping.")
            continue

        if not export_pdf_via_playwright(html_path, pdf_path):
            export_pdf_via_edge(html_path, pdf_path)


if __name__ == "__main__":
    main()
