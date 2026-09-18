#!/usr/bin/env python
"""Held-out evaluation: Table 4, the prevalence analysis, and the efficiency scores.

Confidence intervals are computed at the EFFECTIVE sample size derived from the
corpus manifest, not the nominal one. Two evaluation sources are augmented from a
smaller pool of seeds, so their instances are correlated; point estimates are
unaffected but intervals must widen.

    python scripts/evaluate.py --artifacts artifacts/ --data data/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reflex_guard.classifiers import DISPLAY_NAMES  # noqa: E402
from reflex_guard.data import effective_test_size  # noqa: E402
from reflex_guard.metrics import (  # noqa: E402
    operating_point,
    ppv_at_prevalence,
    res,
    res_j,
    wilson_interval,
)

PREVALENCES = [0.20, 0.10, 0.05, 0.01, 0.005, 0.001]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--data", default="data")
    ap.add_argument("--latency", default=None, help="latency_results.json from latency_bench.py")
    args = ap.parse_args()

    art = Path(args.artifacts)
    blob = np.load(art / "embeddings.npz")
    yte = blob["yte"]

    nominal, independent = effective_test_size(Path(args.data) / "manifest.json")
    deff = nominal / independent if independent else 1.0
    n_pos, n_neg = int((yte == 1).sum()), int((yte == 0).sum())
    print(f"test set: {n_pos} harmful ({independent} independent, design effect "
          f"{deff:.2f}), {n_neg} benign\n")

    lat = {}
    if args.latency and Path(args.latency).exists():
        raw = json.loads(Path(args.latency).read_text(encoding="utf-8"))
        # keys beginning with '_' hold metadata (_environment, _embedding), not models
        lat = {k: v["mean_ms"] for k, v in raw.items()
               if not k.startswith("_") and isinstance(v, dict) and "mean_ms" in v}

    summary = json.loads((art / "training_summary.json").read_text(encoding="utf-8"))
    rows = []
    for name, meta in summary.items():
        score_path = art / name / "test_scores.npy"
        if not score_path.exists():
            continue
        scores = np.load(score_path)
        tau = meta["threshold_cv"]
        op = operating_point(yte, scores, tau)

        # effective denominators: harmful-derived counts deflated, benign untouched
        n_rec = independent
        n_prec = op.counts.tp / deff + op.counts.fp
        n_acc = independent + n_neg

        row = {
            "classifier": DISPLAY_NAMES.get(name, name),
            "tau": round(tau, 3),
            "accuracy": round(op.accuracy, 3),
            "acc_ci": [round(v, 3) for v in wilson_interval(op.accuracy, n_acc)],
            "precision": round(op.precision, 3),
            "prec_ci": [round(v, 3) for v in wilson_interval(op.precision, n_prec)],
            "recall": round(op.recall, 3),
            "rec_ci": [round(v, 3) for v in wilson_interval(op.recall, n_rec)],
            "fpr": round(op.fpr, 4),
            "f1": round(op.f1, 3),
            "youden_j": round(op.youden_j, 4),
            "tp": op.counts.tp, "fp": op.counts.fp,
            "tn": op.counts.tn, "fn": op.counts.fn,
        }
        if name in lat:
            row["latency_ms"] = round(lat[name], 2)
            row["RES"] = round(res(op.recall, lat[name]), 2)
            row["RES_J"] = round(res_j(op.recall, op.fpr, lat[name]), 2)
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    pd.set_option("display.width", 200)
    print(table.drop(columns=["tp", "fp", "tn", "fn"]).to_string(index=False))
    table.to_csv(art / "heldout_results.csv", index=False)

    # -------- prevalence analysis for the best-F1 model (paper's Table 9) -----
    best = table.iloc[0]
    tpr, fpr = float(best.recall), float(best.fpr)
    print(f"\nOperating point of {best.classifier}: TPR={tpr:.3f}, FPR={fpr:.4f}")
    print(f"{'prevalence':>11} {'precision':>10} {'FDR':>7} {'true/10k':>9} {'false/10k':>10}")
    obs = n_pos / (n_pos + n_neg)
    base = [(obs, "  <- test set")] + [(p, "") for p in PREVALENCES]
    prev_rows = []
    for pi, tag in base:
        ppv = ppv_at_prevalence(tpr, fpr, pi)
        tp10k, fp10k = 10000 * pi * tpr, 10000 * (1 - pi) * fpr
        print(f"{pi*100:10.1f}% {ppv:10.3f} {1-ppv:7.3f} {tp10k:9.0f} {fp10k:10.0f}{tag}")
        prev_rows.append({"prevalence": pi, "ppv": ppv, "fdr": 1 - ppv,
                          "true_per_10k": tp10k, "false_per_10k": fp10k})
    pd.DataFrame(prev_rows).to_csv(art / "prevalence_analysis.csv", index=False)

    print(f"\nwrote {art}/heldout_results.csv and {art}/prevalence_analysis.csv")
    if not lat:
        print("note: RES / RES_J omitted - run scripts/latency_bench.py first, "
              "then pass --latency artifacts/latency_results.json")


if __name__ == "__main__":
    main()
