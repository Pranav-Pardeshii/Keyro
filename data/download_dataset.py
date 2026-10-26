"""
CMU Keystroke Dynamics Benchmark Dataset Downloader
Dataset: DSL-StrongPasswordData.csv
Reference: Killourhy, K. S., & Maxion, R. A. (2009). Comparing anomaly-detection algorithms for keystroke dynamics.
"""

import os
import sys
import urllib.request
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_URL = "http://www.cs.cmu.edu/~keystroke/DSL-StrongPasswordData.csv"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(DATA_DIR, "DSL-StrongPasswordData.csv")

EXPECTED_COLUMNS = [
    "subject", "sessionIndex", "rep",
    "H.period", "DD.period.t", "UD.period.t",
    "H.t", "DD.t.i", "UD.t.i",
    "H.i", "DD.i.e", "UD.i.e",
    "H.e", "DD.e.five", "UD.e.five",
    "H.five", "DD.five.Shift.r", "UD.five.Shift.r",
    "H.Shift.r", "DD.Shift.r.o", "UD.Shift.r.o",
    "H.o", "DD.o.a", "UD.o.a",
    "H.a", "DD.a.n", "UD.a.n",
    "H.n", "DD.n.l", "UD.n.l",
    "H.l", "DD.l.Return", "UD.l.Return",
    "H.Return"
]

def download_dataset(target_path: str = DATA_FILE, force: bool = False) -> str:
    """
    Downloads the CMU keystroke dataset if not present.
    If the automated download fails, prints clear instructions for manual placement.
    """
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    if os.path.exists(target_path) and not force:
        file_size = os.path.getsize(target_path)
        if file_size > 1_000_000:
            logger.info(f"Dataset already exists at {target_path} ({file_size:,} bytes). Skipping download.")
            return target_path
        else:
            logger.warning(f"File at {target_path} is suspiciously small ({file_size} bytes). Re-downloading.")

    logger.info(f"Downloading CMU Keystroke Benchmark dataset from {DATA_URL}...")
    try:
        req = urllib.request.Request(
            DATA_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) KeyroBenchmark/1.0"}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(target_path, "wb") as out_file:
            total_size = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded * 100 / total_size
                    print(f"\rDownload progress: {downloaded:,} / {total_size:,} bytes ({pct:.1f}%)", end="", flush=True)
            print()
        logger.info(f"Successfully downloaded dataset to {target_path} ({os.path.getsize(target_path):,} bytes)")
        return target_path
    except Exception as e:
        logger.error(f"Failed to automatically download dataset: {e}")
        print("\n" + "=" * 70)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print(f"1. Download the dataset file 'DSL-StrongPasswordData.csv' from:")
        print(f"   {DATA_URL}")
        print(f"2. Save the file into your project at:")
        print(f"   {os.path.abspath(target_path)}")
        print("=" * 70 + "\n")
        raise RuntimeError(f"Could not download dataset. Please place 'DSL-StrongPasswordData.csv' at {target_path}") from e


def verify_dataset(file_path: str = DATA_FILE) -> bool:
    """Verifies that the dataset file exists, has correct headers, and can be read."""
    if not os.path.exists(file_path):
        logger.error(f"Dataset file not found: {file_path}")
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
        headers = [col.strip() for col in header_line.split(",")]

    missing = [c for c in EXPECTED_COLUMNS if c not in headers]
    if missing:
        logger.error(f"Dataset missing expected columns: {missing}")
        return False

    line_count = sum(1 for _ in open(file_path, "r", encoding="utf-8"))
    logger.info(f"Dataset verified: {line_count - 1} records found with {len(headers)} columns.")
    return True


if __name__ == "__main__":
    path = download_dataset()
    if verify_dataset(path):
        print(f"Dataset ready at: {path}")
    else:
        sys.exit(1)
