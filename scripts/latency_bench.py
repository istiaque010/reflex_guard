#!/usr/bin/env python
"""Latency benchmark (Table 6): component, end-to-end, and the TAIL.

Reports mean, P50, P95 and P99. The tail is the point: for a lightweight
guardrail the mean flatters the architecture and the P99 does not, and P99 is
what a user actually experiences one request in a hundred.

Measurement protocol: batch size 1 (worst case, matching single-request serving),
warm-up iterations discarded, ``time.perf_counter`` throughout, and a CUDA
synchronise before each timestamp so GPU work is not attributed to the next call.

    python scripts/latency_bench.py --artifacts artifacts/ --n 100 --warmup 10
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reflex_guard.classifiers import DISPLAY_NAMES  # noqa: E402
from reflex_guard.embedding import BGEEmbedder  # noqa: E402
from reflex_guard.guard import ReflexGuard  # noqa: E402


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _stats(samples_ms: list[float]) -> dict:
    a = np.asarray(samples_ms, dtype=float)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "n": int(a.size),
    }


def environment() -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch"] = None
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    art = Path(args.artifacts)
    test = pd.read_parquet(art / "test.parquet")
    rng = np.random.default_rng(args.seed)
    prompts = test.text.iloc[rng.choice(len(test), args.n, replace=False)].tolist()
    warm = test.text.iloc[0]

    emb = BGEEmbedder()
    env = environment()
    print(json.dumps(env, indent=2))

    # ---- shared embedding cost -------------------------------------------
    for _ in range(args.warmup):
        emb.encode([warm], batch_size=1)
    _sync()
    emb_ms = []
    for p in prompts:
        _sync(); t0 = time.perf_counter()
        emb.encode([p], batch_size=1)
        _sync(); emb_ms.append((time.perf_counter() - t0) * 1000)
    emb_stats = _stats(emb_ms)
    print(f"\nembedding only: mean {emb_stats['mean_ms']:.2f} ms  "
          f"P50 {emb_stats['p50_ms']:.2f}  P99 {emb_stats['p99_ms']:.2f}")

    summary = json.loads((art / "training_summary.json").read_text(encoding="utf-8"))
    results = {"_embedding": emb_stats, "_environment": env}
    rows = []
    for name in summary:
        model_dir = art / name
        if not model_dir.exists():
            continue
        guard = ReflexGuard.load(model_dir, embedder=emb)

        for _ in range(args.warmup):
            guard.decide(warm)
        _sync()

        e2e, clf_only = [], []
        for p in prompts:
            _sync(); t0 = time.perf_counter()
            guard.decide(p)                       # preprocess + embed + classify
            _sync(); e2e.append((time.perf_counter() - t0) * 1000)

            v = emb.encode([p], batch_size=1)     # isolate classifier inference
            _sync(); t1 = time.perf_counter()
            guard.score_embeddings(v)
            _sync(); clf_only.append((time.perf_counter() - t1) * 1000)

        st = _stats(e2e)
        st["classifier_only_mean_ms"] = float(np.mean(clf_only))
        st["component_mean_ms"] = emb_stats["mean_ms"] + st["classifier_only_mean_ms"]
        results[name] = st
        rows.append({"classifier": DISPLAY_NAMES.get(name, name), **st})
        print(f"{DISPLAY_NAMES.get(name, name):24s} mean {st['mean_ms']:7.2f}  "
              f"P50 {st['p50_ms']:7.2f}  P99 {st['p99_ms']:7.2f}  "
              f"(clf {st['classifier_only_mean_ms']:.2f})")

    (art / "latency_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    df = pd.DataFrame(rows).sort_values("mean_ms")
    df.to_csv(art / "latency_results.csv", index=False)

    if not df.empty:
        worst = df.p99_ms.max()
        print(f"\nP99 spans {df.p99_ms.min():.0f}-{worst:.0f} ms, i.e. up to "
              f"{worst / df.mean_ms.min():.1f}x the fastest mean.")
        print("Report the distribution, not just the mean.")
    print(f"wrote {art}/latency_results.json")


if __name__ == "__main__":
    main()
