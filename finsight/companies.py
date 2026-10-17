"""Company list (data/companies.csv): code, name, sector, aliases separated by ';'."""
from __future__ import annotations

import csv
from pathlib import Path

from .config import ROOT


def load_companies(path: str | Path = ROOT / "data" / "companies.csv") -> dict[str, dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return {row["code"]: {"name": row["name"], "sector": row["sector"],
                              "aliases": [a.strip() for a in row["aliases"].split(";") if a.strip()]}
                for row in csv.DictReader(f)}


COMPANIES = load_companies()
