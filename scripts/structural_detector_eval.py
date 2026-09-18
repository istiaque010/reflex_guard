#!/usr/bin/env python
"""Does the structural-pattern detector actually work?

THE SECOND EXPERIMENT THE PAPER IS MISSING. The two-tier threshold policy routes
"DrAttack-like" prompts to a sensitive threshold. That policy is only sound if
the router itself is precise. If the detector fires on benign prompts that
legitimately use research or educational framing, the policy inherits the very
false-positive rate it was designed to avoid.

Measured here:
  * recall on DrAttack-style prompts (does it catch what it should?)
  * false-positive rate on held-out benign prompts (what does it cost?)
  * the resulting END-TO-END false-positive rate of the two-tier policy, which is
    the number that actually matters
  * per-pattern firing rates, so a bad pattern can be identified and removed

    python scripts/structural_detector_eval.py --artifacts artifacts/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reflex_guard import preprocessing as pp  # noqa: E402
from reflex_guard.attacks import make_drattack_attacks  # noqa: E402
from reflex_guard.embedding import BGEEmbedder  # noqa: E402
from reflex_guard.guard import ReflexGuard  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--model", default="hist_gradient_boosting")
    ap.add_argument("--behaviors", default="data/attacks/behaviors.txt")
    ap.add_argument("--sensitive", type=float, default=0.03)
    ap.add_argument("--n-drattack", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    art = Path(args.artifacts)
    det = pp.JailbreakAwarePreprocessor.detect_structural

    test = pd.read_parquet(art / "test.parquet")
    benign = test[test.label == 0].text.tolist()
    harmful = test[test.label == 1].text.tolist()
    behaviors = [ln.strip() for ln in Path(args.behaviors).read_text(encoding="utf-8").splitlines() if ln.strip()]
    drattack = make_drattack_attacks(behaviors, n=args.n_drattack, seed=args.seed)

    fires_dra = np.array([det(t) for t in drattack])
    fires_ben = np.array([det(t) for t in benign])
    fires_har = np.array([det(t) for t in harmful])

    print("=== structural detector in isolation ===")
    print(f"DrAttack prompts      n={len(drattack):5d}  fires {fires_dra.mean():6.1%}   <- recall")
    print(f"benign (held-out)     n={len(benign):5d}  fires {fires_ben.mean():6.1%}   <- cost")
    print(f"other harmful         n={len(harmful):5d}  fires {fires_har.mean():6.1%}")

    print("\n=== per-pattern firing rate on benign traffic ===")
    rows = []
    for i, pat in enumerate(pp._STRUCTURAL_PATTERNS):
        b = float(np.mean([bool(pat.search(t)) for t in benign]))
        d = float(np.mean([bool(pat.search(t)) for t in drattack]))
        rows.append({"pattern": pat.pattern[:58], "benign_fire": b, "drattack_fire": d,
                     "ratio": (d / b) if b else float("inf")})
        print(f"  [{i}] benign {b:6.2%}  drattack {d:6.2%}   {pat.pattern[:56]}")
    pd.DataFrame(rows).to_csv(art / "structural_patterns.csv", index=False)

    # ---- end-to-end cost of the two-tier policy --------------------------
    emb = BGEEmbedder()
    guard = ReflexGuard.load(art / args.model, embedder=emb)
    tau = guard.threshold
    ben_scores = guard.score(benign, show_progress=True)
    dra_scores = guard.score(drattack)

    fpr_single = float((ben_scores >= tau).mean())
    fpr_global = float((ben_scores >= args.sensitive).mean())
    eff_tau = np.where(fires_ben, args.sensitive, tau)
    fpr_two_tier = float((ben_scores >= eff_tau).mean())

    rec_single = float((dra_scores >= tau).mean())
    rec_two_tier = float((dra_scores >= np.where(fires_dra, args.sensitive, tau)).mean())

    print(f"\n=== two-tier policy, end to end (tau={tau:.3f}, sensitive={args.sensitive}) ===")
    print(f"{'policy':<34}{'benign FPR':>12}{'DrAttack recall':>18}")
    print(f"{'single threshold':<34}{fpr_single:>11.2%}{rec_single:>18.2%}")
    print(f"{'global sensitive threshold':<34}{fpr_global:>11.2%}{1.0:>18.2%}")
    print(f"{'two-tier (structural routing)':<34}{fpr_two_tier:>11.2%}{rec_two_tier:>18.2%}")

    extra = fpr_two_tier - fpr_single
    print(f"\nTwo-tier routing costs {extra:+.2%} additional benign false positives "
          f"and recovers {rec_two_tier - rec_single:+.2%} DrAttack recall.")
    if extra > 0.02:
        print("VERDICT: the detector is too loose on benign traffic. Tighten the "
              "patterns (see structural_patterns.csv) or drop the policy.")
    elif rec_two_tier < 0.5:
        print("VERDICT: the detector misses most DrAttack prompts, so routing "
              "rarely triggers. The policy does little either way.")
    else:
        print("VERDICT: routing is favourable on this data. State the measured "
              "numbers in the paper rather than presenting the policy as a proposal.")

    pd.DataFrame([{
        "single_fpr": fpr_single, "global_sensitive_fpr": fpr_global,
        "two_tier_fpr": fpr_two_tier, "single_recall": rec_single,
        "two_tier_recall": rec_two_tier,
        "detector_recall": float(fires_dra.mean()),
        "detector_benign_fire": float(fires_ben.mean()),
    }]).to_csv(art / "structural_detector_eval.csv", index=False)
    print(f"\nwrote {art}/structural_detector_eval.csv")


if __name__ == "__main__":
    main()
