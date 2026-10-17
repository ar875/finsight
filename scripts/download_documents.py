"""Download the PDFs listed in data/document_links.csv into data/raw (skips files already there).
Some company sites block scripts: the script reports those, and you download them by hand.

    python scripts/download_documents.py
"""
from __future__ import annotations

import csv
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/pdf,*/*",
}


def make_filename(company: str, year: str, doc_type: str, quarter: str) -> str:
    if not re.fullmatch(r"FY\d{4}", year):
        raise ValueError(f"year must look like FY2025, got '{year}'")
    if doc_type == "earnings_call":
        if quarter not in {"1", "2", "3", "4"}:
            raise ValueError("earnings_call needs quarter 1-4")
        return f"{company}_{year}Q{quarter}_earnings_call.pdf"
    if doc_type == "annual_report":
        return f"{company}_{year}_annual_report.pdf"
    raise ValueError(f"unknown doc_type '{doc_type}'")


def download_file(url: str, dest: Path, retries: int = 3) -> str:
    headers = dict(HEADERS)
    if "bseindia.com" in url:
        headers["Referer"] = "https://www.bseindia.com/"
    last_error = ""
    for attempt in range(retries):
        try:
            with requests.get(url, headers=headers, timeout=180, stream=True) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for block in r.iter_content(1 << 16):
                        f.write(block)
            with open(tmp, "rb") as f:
                if f.read(5) != b"%PDF-":      # the site sent an HTML page instead of a PDF
                    tmp.unlink()
                    return "NOT A PDF (site blocked the download: get this one manually)"
            tmp.rename(dest)
            return "ok"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:150]
            time.sleep(3 * (attempt + 1))
    return f"FAILED: {last_error}"


def main(links_csv=ROOT / "data/document_links.csv", out_dir=ROOT / "data/raw") -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(links_csv, encoding="utf-8", newline="") as f:
        links = list(csv.DictReader(f))
    problems = []
    for i, row in enumerate(links, start=1):
        name = make_filename(row["company"], row["year"], row["doc_type"], row["quarter"])
        dest = out / name
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"[{i}/{len(links)}] {name}: already downloaded")
            continue
        status = download_file(row["url"], dest)
        size = f"({dest.stat().st_size / 1e6:.1f} MB)" if dest.exists() else ""
        print(f"[{i}/{len(links)}] {name}: {status} {size}")
        if status != "ok":
            problems.append((name, status))
        time.sleep(1)    # be polite to the servers
    print(f"\nProblems: {len(problems)}")
    for name, status in problems:
        print(f"  - {name}: {status}")
    return len(problems)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
