#!/usr/bin/env python
"""Embed the corpus, fit every classifier, and select the default threshold.

The threshold is chosen by maximising F1 over stratified cross-validation folds
of the TRAINING partition only. The held-out test set is never touched here.
That separation is the whole reason the test-set numbers mean anything.

    python scripts/train.py --data data/ --out artifacts/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reflex_guard.classifiers import (  # noqa: E402
    DISPLAY_NAMES,
    available_classifiers,
    build_classifier,
)
from reflex_guard.data import load_corpus, split_corpus  # noqa: E402
from reflex_guard.embedding import PAPER_PREFIX, BGEEmbedder  # noqa: E402
from reflex_guard.guard import ReflexGuard  # noqa: E402
from reflex_guard.metrics import f1_optimal_threshold  # noqa: E402
from reflex_guard.preprocessing import JailbreakAwarePreprocessor  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prefix", default=PAPER_PREFIX, help="instruction prefix; '' to disable")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--classifiers", nargs="*", default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_corpus(args.data)
    train, test = split_corpus(df, seed=args.seed)
    print(f"train {len(train):,} ({train.label.mean():.1%} harmful) | "
          f"test {len(test):,} ({test.label.mean():.1%} harmful)")

    pre = JailbreakAwarePreprocessor()
    emb = BGEEmbedder(prefix=args.prefix)
    print(f"embedding with {emb.model_name} on {emb.device}, dim={emb.dim}, prefix={args.prefix!r}")

    t0 = time.perf_counter()
    Xtr = emb.encode([pre.process(t).text for t in train.text.tolist()], show_progress=True)
    Xte = emb.encode([pre.process(t).text for t in test.text.tolist()], show_progress=True)
    print(f"embedded {len(Xtr)+len(Xte):,} prompts in {time.perf_counter()-t0:.1f}s")

    ytr = train.label.to_numpy()
    yte = test.label.to_numpy()
    np.savez_compressed(out / "embeddings.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    train.to_parquet(out / "train.parquet", index=False)
    test.to_parquet(out / "test.parquet", index=False)

    names = args.classifiers or available_classifiers()
    summary = {}
    for name in names:
        print(f"\n--- {DISPLAY_NAMES.get(name, name)} ---")

        # Threshold selection: CV on the training partition only.
        oof = np.zeros(len(ytr), dtype=float)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        for k, (tr_i, va_i) in enumerate(skf.split(Xtr, ytr), 1):
            clf = build_classifier(name, seed=args.seed, y=ytr[tr_i])
            clf.fit(Xtr[tr_i], ytr[tr_i])
            oof[va_i] = clf.predict_proba(Xtr[va_i])[:, 1]
            print(f"  fold {k}/{args.folds} done")
        tau = f1_optimal_threshold(ytr, oof)
        print(f"  CV-selected threshold tau* = {tau:.3f}")

        guard = ReflexGuard(classifier=name, threshold=tau, seed=args.seed,
                            embedder=emb, preprocessor=pre, prefix=args.prefix)
        guard.fit(X=Xtr, y=ytr)
        guard.save(out / name)

        np.save(out / name / "test_scores.npy", guard.score_embeddings(Xte))
        np.save(out / name / "oof_scores.npy", oof)
        summary[name] = {"threshold_cv": tau, "display": DISPLAY_NAMES.get(name, name)}

    (out / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote artifacts to {out}/")
    print("threshold values above were selected on training folds; "
          "run scripts/evaluate.py for held-out results.")


if __name__ == "__main__":
    main()
