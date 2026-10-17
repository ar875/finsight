"""Build (or update) the search index from the PDFs in data/raw.

    python scripts/build_index.py                         # uses config.yaml
    python scripts/build_index.py --raw-dir /path/to/pdfs --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from finsight import index_dir, load_config  # noqa: E402
from finsight.indexing.ingest import ingest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", help="folder with the PDFs (default: data.raw_dir in config.yaml)")
    ap.add_argument("--force", action="store_true", help="re-process every file even if unchanged")
    args = ap.parse_args()
    cfg = load_config(overrides={"data": {"raw_dir": args.raw_dir}} if args.raw_dir else None)
    stats = ingest(cfg, force=args.force)
    for k, v in stats.items():
        print(f"{k:>20}: {v}")
    print("Index folder:", index_dir(cfg))


if __name__ == "__main__":
    main()
