"""
Download script for Cambodian legal documents.

Fetches target official legal PDFs (English & Khmer) from
Open Development Cambodia (ODC) and stores them in data/01_raw/.
"""

import sys
from pathlib import Path
import httpx
from tqdm import tqdm

from src.config.logging import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

DOCUMENTS = [
    {
        "name": "Civil Code of Cambodia (2007)",
        "law_key": "civil_code_2007_en",
        "language": "en",
        "url": "https://data.opendevelopmentcambodia.net/en/dataset/09d1e634-7256-47e9-89e8-1b6494af8883/resource/b3d4b1cd-9bdf-409e-8ada-1a764bc0b793/download/6b6d012f-869c-49b8-ae6e-8a9a4cccae34.pdf",
        "filename": "civil_code_2007_en.pdf",
    },
    {
        "name": "Law on Commercial Enterprises (2005)",
        "law_key": "commercial_enterprises_2005_en",
        "language": "en",
        "url": "https://data.opendevelopmentcambodia.net/en/dataset/966485f2-b2f4-49e3-9125-baeeadf13cb8/resource/ad5c06ef-5d7f-4083-8ef0-ea034488a2bf/download/88376-law-on-commercial-enterprises-2005.pdf",
        "filename": "commercial_enterprises_2005_en.pdf",
    },
    {
        "name": "Law on Commercial Arbitration (2006)",
        "law_key": "commercial_arbitration_2006_en",
        "language": "en",
        "url": "https://data.opendevelopmentmekong.net/dataset/28a1dc26-20a0-4dc8-8d62-8f191348dd53/resource/f48f4119-cfe6-41fe-b54f-48b03341bedc/download/563a6ba2-a45a-42be-8f0d-6baecc84aca7.pdf",
        "filename": "commercial_arbitration_2006_en.pdf",
    },
    {
        "name": "Civil Code of Cambodia (2007) — Khmer",
        "law_key": "civil_code_2007_kh",
        "language": "kh",
        "url": "https://data.opendevelopmentcambodia.net/en/dataset/09d1e634-7256-47e9-89e8-1b6494af8883/resource/e92232fb-eb45-47a0-a7f5-5018e9fd5f39/download/6fa9349f-39c9-440f-881b-9e052653e69c.pdf",
        "filename": "civil_code_2007_kh.pdf",
    },
    {
        "name": "Criminal Code of Cambodia (2009) — Khmer",
        "law_key": "criminal_code_2009_kh",
        "language": "kh",
        "url": "https://data.opendevelopmentcambodia.net/dataset/aac9b82e-4672-4904-9629-e120604f55a7/resource/39521a25-bc64-4d35-908f-f3bcd9a472da/download/2505c33c-a399-42da-8e9e-5b119d293025.pdf",
        "filename": "criminal_code_2009_kh.pdf",
    },
]


def download_file(url: str, destination: Path) -> bool:
    """Download a file via HTTP with streaming and progress bar."""
    if destination.exists() and destination.stat().st_size > 1000:
        logger.info(f"File already exists: {destination.name} ({destination.stat().st_size} bytes)")
        return True

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(".tmp")

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        with httpx.Client(headers=headers, follow_redirects=True, timeout=60.0) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))

                with open(temp_path, "wb") as f, tqdm(
                    desc=destination.name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))

        temp_path.replace(destination)
        logger.info(f"Successfully downloaded: {destination.name} ({destination.stat().st_size} bytes)")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def main() -> None:
    """Download all target PDFs."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    raw_dir = base_dir / "data" / "01_raw"

    logger.info("Starting legal document downloads...")
    success_count = 0

    for doc in DOCUMENTS:
        target_dir = raw_dir / doc["language"]
        target_path = target_dir / doc["filename"]
        logger.info(f"Downloading {doc['name']} to {target_path}...")
        
        if download_file(doc["url"], target_path):
            success_count += 1

    logger.info(f"Downloads completed: {success_count}/{len(DOCUMENTS)} successful.")
    if success_count < len(DOCUMENTS):
        sys.exit(1)


if __name__ == "__main__":
    main()
