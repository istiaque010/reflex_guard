#!/usr/bin/env python
"""Write this run's results in the layout of the article's Tables 4 to 7.

    python scripts/paper_tables.py --artifacts artifacts/ --data data/paper/

Run it after train.py, latency_bench.py and attack_eval.py. It reads the saved scores and
reports them at the paper's fixed operating points (tau = 0.25 by default, and the sensitive
tau = 0.03), with the same columns as the article, so the two can be compared directly.
It writes, into <artifacts>/paper_tables/:

    table4_heldout.csv          Table 4: held-out metrics at tau, Wilson CIs at the effective n
    table5_attacks.csv          Table 5: attack-family recall at tau and at the sensitive tau
    table6_latency.csv          Table 6: per-request latency (min, mean, P50, P99) on this machine
    table7_threshold_sweep.csv  Table 7: DrAttack-style recall, general recall and benign FPR
    comparison.md               these values next to the values reported in the article

The numbers are what this run measured. They are not adjusted toward the article's values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reflex_guard.classifiers import DISPLAY_NAMES  # noqa: E402
from reflex_guard.data import effective_test_size  # noqa: E402
from reflex_guard.metrics import operating_point, res, res_j, wilson_interval  # noqa: E402

PAPER = ROOT / "results" / "paper_tables"
SWEEP = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
FAMILIES = [("gcg", "GCG-style suffix"), ("base64", "Base64-encoded"), ("drattack", "DrAttack-style scaffolding")]


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--data", default="data/paper")
    ap.add_argument("--tau", type=float, default=0.25, help="default operating point (paper: 0.25)")
    ap.add_argument("--sensitive-tau", type=float, default=0.03, help="sensitive operating point (paper: 0.03)")
    ap.add_argument("--model", default="hist_gradient_boosting", help="classifier used for Tables 5 and 7")
    args = ap.parse_args()

    art = Path(args.artifacts)
    out = art / "paper_tables"
    out.mkdir(parents=True, exist_ok=True)
    yte = np.load(art / "embeddings.npz")["yte"]
    nominal, independent = effective_test_size(Path(args.data) / "manifest.json")
    deff = nominal / independent if independent else 1.0
    n_neg = int((yte == 0).sum())

    lat_raw = json.loads((art / "latency_results.json").read_text(encoding="utf-8"))
    lat = {k: v for k, v in lat_raw.items() if not k.startswith("_") and isinstance(v, dict)}
    summary = json.loads((art / "training_summary.json").read_text(encoding="utf-8"))

    # ---------------- Table 4 ----------------
    rows = []
    for name in summary:
        path = art / name / "test_scores.npy"
        if not path.exists():
            continue
        s = np.load(path)
        op = operating_point(yte, s, args.tau)
        # Wilson intervals at the effective sample size, as in the article
        n_eff = {"accuracy": independent + n_neg, "precision": op.counts.tp / deff + op.counts.fp,
                 "recall": independent}
        mean_ms = lat.get(name, {}).get("mean_ms")
        row = {"classifier": DISPLAY_NAMES.get(name, name)}
        for metric in ("accuracy", "precision", "recall"):
            value = getattr(op, metric)
            low, high = wilson_interval(value, n_eff[metric])
            row.update({metric: round(value, 3), f"{metric}_ci_low": round(low, 3), f"{metric}_ci_high": round(high, 3)})
        row.update({"fpr": round(op.fpr, 3), "f1": round(op.f1, 3), "roc_auc": round(roc_auc_score(yte, s), 3),
                    "res": round(res(op.recall, mean_ms), 2) if mean_ms else None,
                    "res_j": round(res_j(op.recall, op.fpr, mean_ms), 2) if mean_ms else None})
        rows.append(row)
    t4 = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    t4 = t4[["classifier", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "precision", "precision_ci_low",
             "precision_ci_high", "recall", "recall_ci_low", "recall_ci_high", "fpr", "f1", "roc_auc", "res", "res_j"]]
    t4.to_csv(out / "table4_heldout.csv", index=False)

    # ---------------- Table 5 ----------------
    t5 = []
    for key, label in FAMILIES:
        p = art / f"scores_{key}.npy"
        if p.exists():
            s = np.load(p)
            t5.append({"attack_family": label, "samples": len(s),
                       f"recall_tau_{args.tau:g}": pct((s >= args.tau).mean()),
                       f"recall_tau_{args.sensitive_tau:g}": pct((s >= args.sensitive_tau).mean()),
                       "mean_prob": round(float(s.mean()), 2), "sd_prob": round(float(s.std()), 2)})
    t5 = pd.DataFrame(t5)
    t5.to_csv(out / "table5_attacks.csv", index=False)

    # ---------------- Table 6 ----------------
    t6 = pd.DataFrame([{"classifier": DISPLAY_NAMES.get(k, k), "min_ms": round(v["min_ms"], 2),
                        "mean_ms": round(v["mean_ms"], 2), "p50_ms": round(v["p50_ms"], 2),
                        "p99_ms": round(v["p99_ms"], 2)} for k, v in lat.items()]).sort_values("mean_ms")
    t6.to_csv(out / "table6_latency.csv", index=False)

    # ---------------- Table 7 ----------------
    s_m = np.load(art / args.model / "test_scores.npy")
    dr = np.load(art / "scores_drattack.npy") if (art / "scores_drattack.npy").exists() else None
    t7 = []
    for tau in SWEEP + [args.tau]:
        t7.append({"tau": f"{tau:g}" + (" (default)" if tau == args.tau else ""),
                   "drattack_style_recall": pct((dr >= tau).mean()) if dr is not None else None,
                   "general_recall": pct((s_m[yte == 1] >= tau).mean()),
                   "benign_fpr": pct((s_m[yte == 0] >= tau).mean())})
    t7 = pd.DataFrame(t7)
    t7.to_csv(out / "table7_threshold_sweep.csv", index=False)

    # ---------------- comparison with the article ----------------
    lines = [f"# This run vs. the article (tau = {args.tau:g}, sensitive tau = {args.sensitive_tau:g})", "",
             "Values in brackets are the ones reported in the article (`results/paper_tables/`).", ""]
    if PAPER.exists():
        p4 = pd.read_csv(PAPER / "table4_heldout.csv").set_index("classifier")
        p4.index = p4.index.str.replace(" (k=5)", "", regex=False)
        lines += ["## Table 4 (held-out)", "", "| Classifier | Recall | FPR | F1 | ROC-AUC |", "|---|---|---|---|---|"]
        for _, r in t4.iterrows():
            k = r.classifier.replace(" (k=5)", "")
            ref = p4.loc[k] if k in p4.index else None
            cells = [f"{r[c]} [{ref[c]}]" if ref is not None else f"{r[c]}" for c in ("recall", "fpr", "f1", "roc_auc")]
            lines.append(f"| {r.classifier} | " + " | ".join(cells) + " |")
        p5 = pd.read_csv(PAPER / "table5_attacks.csv").set_index("attack_family")
        lines += ["", "## Table 5 (attack families)", "", "| Family | n | Recall at tau | Recall at sensitive tau | "
                  "Mean prob |", "|---|---|---|---|---|"]
        for _, r in t5.iterrows():
            ref = p5.loc[r.attack_family] if r.attack_family in p5.index else None
            a, b = r.iloc[2], r.iloc[3]
            lines.append(f"| {r.attack_family} | {r.samples} [{ref['samples'] if ref is not None else '-'}] | "
                         f"{a} [{ref['recall_tau_0.25'] if ref is not None else '-'}] | "
                         f"{b} [{ref['recall_tau_0.03'] if ref is not None else '-'}] | "
                         f"{r.mean_prob} [{ref['mean_prob'] if ref is not None else '-'}] |")
        p7 = pd.read_csv(PAPER / "table7_threshold_sweep.csv")
        lines += ["", "## Table 7 (threshold sweep)", "", "| tau | DrAttack-style recall | General recall | Benign FPR |",
                  "|---|---|---|---|"]
        for (_, r), (_, q) in zip(t7.iterrows(), p7.iterrows(), strict=True):
            lines.append(f"| {r.tau} | {r.drattack_style_recall} [{q.drattack_style_recall}] | "
                         f"{r.general_recall} [{q.general_recall}] | {r.benign_fpr} [{q.benign_fpr}] |")
    lines += ["", "Latency (Table 6) depends on hardware and is not compared here.", ""]
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    pd.set_option("display.width", 220)
    for title, df in [("Table 4", t4), ("Table 5", t5), ("Table 6", t6), ("Table 7", t7)]:
        print(f"\n=== {title} ===")
        print(df.to_string(index=False))
    print(f"\nwrote {out}/ (table4-7 CSV files and comparison.md)")


if __name__ == "__main__":
    main()
