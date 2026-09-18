#!/usr/bin/env python
"""Attack-family robustness and the threshold sweep (Tables 6, 10, 11).

Reports, per family, detection at the default and sensitive thresholds, the
ablation of jailbreak-aware preprocessing, and the benign false-positive rate at
every threshold so the cost of sensitivity is visible alongside its benefit.

    python scripts/attack_eval.py --artifacts artifacts/ --model hist_gradient_boosting
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reflex_guard.attacks import (  # noqa: E402
    load_gcg_suffixes,
    make_base64_attacks,
    make_drattack_attacks,
    make_gcg_attacks,
)
from reflex_guard.embedding import BGEEmbedder  # noqa: E402
from reflex_guard.guard import ReflexGuard  # noqa: E402
from reflex_guard.preprocessing import JailbreakAwarePreprocessor  # noqa: E402

SWEEP = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.15, 0.20, 0.25]


def recall_at(scores: np.ndarray, tau: float) -> float:
    return float((scores >= tau).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--model", default="hist_gradient_boosting")
    ap.add_argument("--behaviors", default="data/attacks/behaviors.txt")
    ap.add_argument("--gcg-suffixes", default="data/attacks/gcg_suffixes.txt")
    ap.add_argument("--n-gcg", type=int, default=60)
    ap.add_argument("--n-b64", type=int, default=100)
    ap.add_argument("--n-drattack", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    art = Path(args.artifacts)
    emb = BGEEmbedder()
    guard = ReflexGuard.load(art / args.model, embedder=emb)
    tau_default = guard.threshold
    print(f"model={args.model}  default tau={tau_default:.3f}\n")

    behaviors = [ln.strip() for ln in Path(args.behaviors).read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"{len(behaviors)} seed behaviours")

    families: dict[str, list[str]] = {
        "Base64": make_base64_attacks(behaviors, n=args.n_b64, seed=args.seed),
        "DrAttack": make_drattack_attacks(behaviors, n=args.n_drattack, seed=args.seed),
    }
    # The shipped gcg_suffixes.txt is comments only, so test for CONTENT rather
    # than for the file's existence.
    gcg_path = Path(args.gcg_suffixes)
    suffixes = load_gcg_suffixes(gcg_path) if gcg_path.exists() else []
    if suffixes:
        families["GCG"] = make_gcg_attacks(behaviors, suffixes, n=args.n_gcg, seed=args.seed)
        print(f"{len(suffixes)} GCG suffixes loaded")
    else:
        print(f"NOTE: no GCG suffixes in {gcg_path} - skipping the GCG family.\n"
              f"      Generate them against a surrogate model; see docs/REPRODUCING.md.\n"
              f"      Do not hand-write suffixes and report the result as GCG.")

    # benign scores for the false-positive column
    test = pd.read_parquet(art / "test.parquet")
    benign = test[test.label == 0].text.tolist()
    benign_scores = guard.score(benign, show_progress=True)

    rows, sweep_rows = [], []
    for family, prompts in families.items():
        with_pre = guard.score(prompts)
        # ablation: identical pipeline, preprocessing disabled
        guard.pre = JailbreakAwarePreprocessor(decode_base64=False)
        without_pre = guard.score(prompts)
        guard.pre = JailbreakAwarePreprocessor()

        rows.append({
            "attack": family,
            "n": len(prompts),
            "recall@default": recall_at(with_pre, tau_default),
            "recall@0.03": recall_at(with_pre, 0.03),
            "recall@default_no_preproc": recall_at(without_pre, tau_default),
            "mean_prob": float(with_pre.mean()),
            "sd_prob": float(with_pre.std()),
            "min_prob": float(with_pre.min()),
            "max_prob": float(with_pre.max()),
        })
        for tau in SWEEP:
            sweep_rows.append({
                "attack": family,
                "tau": tau,
                "attack_recall": recall_at(with_pre, tau),
                "benign_fpr": recall_at(benign_scores, tau),
            })
        np.save(art / f"scores_{family.lower()}.npy", with_pre)

    res = pd.DataFrame(rows)
    sweep = pd.DataFrame(sweep_rows)
    pd.set_option("display.width", 200)
    print("\n=== attack families ===")
    print(res.to_string(index=False))
    print("\n=== threshold sweep (attack recall vs benign FPR) ===")
    print(sweep.pivot(index="tau", columns="attack",
                      values=["attack_recall", "benign_fpr"]).to_string())

    res.to_csv(art / "attack_results.csv", index=False)
    sweep.to_csv(art / "threshold_sweep.csv", index=False)
    np.save(art / "scores_benign.npy", benign_scores)

    print(f"\nwrote {art}/attack_results.csv and {art}/threshold_sweep.csv")
    print(
        "\nIMPORTANT: if you pick a sensitive threshold by inspecting these same\n"
        "attack scores, the recall you then quote at it is IN-SAMPLE. Say so, or\n"
        "hold out a disjoint attack split for selection."
    )


if __name__ == "__main__":
    main()
