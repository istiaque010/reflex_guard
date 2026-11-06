#!/usr/bin/env python
"""Rebuild the paper's corpus and compare it with the released file.

    python scripts/rebuild_paper_corpus.py

The released file ``data/corpus/reflexguard_corpus.csv`` is the corpus used in the
paper and is the authoritative copy (see ``data/corpus/SHA256SUMS``). This script
re-runs the same procedure (``reflex_guard.paper_corpus``) from the public sources and
reports, per source, how far the rebuilt corpus agrees with the released one.

What to expect: the Alpaca and In-the-Wild/DAN sets and the Anthropic candidate pool
are recovered exactly, and every per-source count matches. The random draws (the
13,000 Anthropic prompts and the resampled JailbreakBench and MultiJail copies) are
not recovered bit for bit, so use the released file for any comparison with the paper.

Requires network access and ``huggingface_hub``; ``datasets`` is not needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reflex_guard.paper_corpus import PAPER_CORPUS_SHA256, build_paper_corpus, sha256  # noqa: E402

RELEASED = ROOT / "data" / "corpus" / "reflexguard_corpus.csv"


def main() -> None:
    digest = sha256(RELEASED)
    print(f"released corpus: {RELEASED.name}  sha256 {digest}")
    print("  checksum OK" if digest == PAPER_CORPUS_SHA256 else "  CHECKSUM MISMATCH (line endings changed?)")
    ref = pd.read_csv(RELEASED, keep_default_na=False)

    print("rebuilding from the public sources (downloads several datasets)...")
    new = build_paper_corpus()
    print(f"\n{'source':20}{'released':>10}{'rebuilt':>10}{'shared unique prompts':>24}")
    for src in ref["source"].unique():
        a = ref.loc[ref["source"] == src, "Goal"].astype(str)
        b = new.loc[new["source"] == src, "Goal"].astype(str)
        shared = len(set(a) & set(b))
        print(f"{src:20}{len(a):>10}{len(b):>10}{shared:>14} / {a.nunique()}")


if __name__ == "__main__":
    main()
