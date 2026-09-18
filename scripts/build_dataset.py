#!/usr/bin/env python
"""Download and assemble the evaluation corpus (Table 3).

Writes ``data/corpus.parquet`` and ``data/manifest.json``. The manifest records
``n_seed`` for the augmented sources; ``scripts/evaluate.py`` needs it to compute
the design effect, so do not delete it.

    python scripts/build_dataset.py --out data/

Requires network access and ``pip install datasets``. If a Hub identifier has
moved, fix the corresponding ``_load_*`` in ``src/reflex_guard/data.py`` and keep
the returned schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reflex_guard.data import build_corpus, split_corpus  # noqa: E402
from reflex_guard.synthetic import build_synthetic_corpus  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--synthetic", action="store_true",
                    help="generate a tiny offline corpus for plumbing tests only; "
                         "results from it are meaningless and must not be reported")
    ap.add_argument("--scale", type=float, default=1.0, help="synthetic corpus size multiplier")
    args = ap.parse_args()

    if args.synthetic:
        print("BUILDING SYNTHETIC CORPUS - for plumbing tests only.")
        print("Metrics computed from it are meaningless. Do not report them.\n")
        df = build_synthetic_corpus(args.out, seed=args.seed, scale=args.scale)
    else:
        print("building corpus (this downloads several datasets)...")
        df = build_corpus(args.out, seed=args.seed)

    print(f"\ntotal {len(df):,} prompts | {df.label.mean():.1%} harmful")
    print(df.groupby(["source", "label", "role"]).size().to_string())

    train, test = split_corpus(df, seed=args.seed)
    print(f"\nsplit -> train {len(train):,} ({train.label.mean():.1%} harmful)"
          f" | test {len(test):,} ({test.label.mean():.1%} harmful)")
    print("harmful sources per partition (must be disjoint):")
    print("  train:", sorted(train[train.label == 1].source.unique()))
    print("  test :", sorted(test[test.label == 1].source.unique()))

    overlap = set(train[train.label == 1].source) & set(test[test.label == 1].source)
    if overlap:
        raise SystemExit(f"ERROR: harmful source leakage across the split: {overlap}")
    print("\nno harmful-source leakage. next: python scripts/train.py")


if __name__ == "__main__":
    main()
