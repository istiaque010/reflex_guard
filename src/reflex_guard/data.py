"""Corpus construction with the known data issues fixed (an alternative to the paper corpus).

NOTE: this is NOT the corpus used in the paper. The paper corpus is released as
``data/corpus/reflexguard_corpus.csv`` and its procedure is in ``paper_corpus.py``.
This module builds a corrected corpus instead: Anthropic red-team attempts only,
MultiJail Bengali text, and an AlpacaEval-plus-Alpaca benign pool. Results on it are
not comparable with the paper's tables.

Five public sources. Training harmful data comes from ONE of them (Anthropic
red-team); the other three harmful sources are evaluation-only, so the reported
test metrics measure cross-source generalisation rather than in-distribution fit.

    source                role        target n    label
    -------------------------------------------------------------
    AlpacaEval            benign      15,000      0
    Anthropic red-team    train       13,000      1
    JailbreakHub / DAN    eval           968      1
    JailbreakBench        eval           800      1   (from 100 seeds)
    MultiJail (Bengali)   eval           800      1   (from 315 seeds)

Two of the evaluation sources are AUGMENTED. ``n_seed`` is carried through the
manifest so that ``scripts/evaluate.py`` can compute the design effect and widen
the confidence intervals accordingly. Do not drop this field.

Dataset identifiers are the best-known public mirrors at time of writing; verify
them against the Hub before relying on a run. If a loader fails, implement the
corresponding ``_load_*`` and keep the returned schema.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["SourceSpec", "SOURCES", "build_corpus", "load_corpus", "split_corpus"]

HF_IDS = {
    "anthropic_redteam": "Anthropic/hh-rlhf",           # data_dir="red-team-attempts"
    "jailbreakbench": "JailbreakBench/JBB-Behaviors",
    "jailbreakhub": "TrustAIRLab/in-the-wild-jailbreak-prompts",
    "multijail": "DAMO-NLP-SG/MultiJail",
    "alpacaeval": "tatsu-lab/alpaca_eval",
    "alpaca52k": "tatsu-lab/alpaca",                    # benign top-up, see below
}

# The JailbreakHub repo publishes four snapshots and has NO default config, so
# omitting the config name raises ValueError. jailbreak_2023_12_25 is the later
# jailbreak-only snapshot (1,405 rows, ~1,356 unique after normalisation).
JAILBREAKHUB_CONFIG = "jailbreak_2023_12_25"


@dataclass
class SourceSpec:
    name: str
    label: int
    target_n: int
    role: str                      # "train" | "eval" | "benign"
    augmented: bool = False
    notes: str = ""
    n_seed: int | None = None      # unique items before augmentation
    extra: dict = field(default_factory=dict)


SOURCES = [
    SourceSpec("alpacaeval", 0, 15_000, "benign", notes="benign baseline"),
    SourceSpec("anthropic_redteam", 1, 13_000, "train", notes="single-turn human openers"),
    SourceSpec("jailbreakhub", 1, 968, "eval", notes="MinHash dedup @ J=0.85"),
    SourceSpec("jailbreakbench", 1, 800, "eval", augmented=True, n_seed=100,
               notes="paraphrase expansion (corrected corpus only)"),
    SourceSpec("multijail", 1, 800, "eval", augmented=True, n_seed=315,
               notes="paraphrase oversampling, Bengali subset"),
]


# --------------------------------------------------------------------- utils
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def _dedup(texts: list[str], min_chars: int = 10) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        t = _norm(t)
        if len(t) < min_chars:
            continue
        key = hashlib.sha1(t.lower().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _subsample(texts: list[str], n: int, seed: int) -> list[str]:
    """Uniform SRS without replacement. Returns everything if n exceeds len."""
    if n >= len(texts):
        return texts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=n, replace=False)
    return [texts[i] for i in sorted(idx)]


# ------------------------------------------------------------------ loaders
def _hf(name: str, *args, **kwargs):
    from datasets import load_dataset

    return load_dataset(name, *args, **kwargs)


def _load_anthropic_redteam(target_n: int, seed: int) -> list[str]:
    """~38,961 single-turn human openers. Column ``transcript``. Not gated.

    The archive is named ``.jsonl.gz`` but contains a single JSON *array*, so we
    parse it directly instead of relying on the ``datasets`` JSON builder falling
    back from ``pyarrow.json.read_json`` to ``json.load``.
    """
    import gzip
    import json as _json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_IDS["anthropic_redteam"],
        filename="red-team-attempts/red_team_attempts.jsonl.gz",
        repo_type="dataset",
    )
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = _json.load(fh)

    opener = re.compile(r"Human:\s*(.*?)(?=\n\nAssistant:|\Z)", re.S)
    texts = []
    for row in rows:
        m = opener.search(row.get("transcript") or "")
        if m:
            texts.append(m.group(1))
    return _subsample(_dedup(texts), target_n, seed)


def _load_jailbreakbench(target_n: int, seed: int) -> list[str]:
    """100 harmful behaviours. Config ``behaviors``, split ``harmful``, column ``Goal``."""
    ds = _hf(HF_IDS["jailbreakbench"], "behaviors", split="harmful")
    return _augment(_dedup([r["Goal"] for r in ds]), target_n, seed)


def _load_jailbreakhub(target_n: int, seed: int) -> list[str]:
    """In-the-wild jailbreak prompts. Requires an explicit config name."""
    try:
        ds = _hf(HF_IDS["jailbreakhub"], JAILBREAKHUB_CONFIG, split="train")
    except ValueError as exc:
        from datasets import get_dataset_config_names

        configs = get_dataset_config_names(HF_IDS["jailbreakhub"])
        raise ValueError(
            f"config {JAILBREAKHUB_CONFIG!r} unavailable for {HF_IDS['jailbreakhub']}. "
            f"Available: {configs}. Pick a 'jailbreak_*' snapshot (the 'regular_*' "
            f"configs are non-jailbreak prompts) and update JAILBREAKHUB_CONFIG."
        ) from exc
    texts = [r["prompt"] for r in ds if r.get("jailbreak", True)]
    return _subsample(_dedup(texts), target_n, seed)


def _load_multijail(target_n: int, seed: int) -> list[str]:
    """315 Bengali harmful queries. Language columns use ISO-639-1 codes."""
    ds = _hf(HF_IDS["multijail"], split="train")
    if "bn" not in ds.column_names:
        raise KeyError(f"no Bengali column in MultiJail; saw {ds.column_names}")
    return _augment(_dedup([r["bn"] or "" for r in ds]), target_n, seed)


def _load_alpacaeval(target_n: int, seed: int) -> list[str]:
    """Benign instructions.

    AlpacaEval proper holds only 805 instructions, far short of the 15,000 the
    corpus spec asks for. Silently returning 805 would leave the test split with
    no benign examples at all and make every FPR and specificity figure
    undefined, so we top up from ``tatsu-lab/alpaca`` (52,002 rows) and say so.

    AlpacaEval also ships a Python loading script, which ``datasets>=3.0``
    refuses outright; the script only reads ``alpaca_eval.json``, so we fetch
    that file directly.
    """
    import json as _json

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_IDS["alpacaeval"], filename="alpaca_eval.json", repo_type="dataset"
    )
    rows = _json.loads(Path(path).read_text(encoding="utf-8"))
    texts = _dedup([r.get("instruction") or "" for r in rows])

    if len(texts) < target_n:
        print(f"    AlpacaEval yields {len(texts)}; topping up from {HF_IDS['alpaca52k']}")
        extra = _hf(HF_IDS["alpaca52k"], split="train")
        texts = _dedup(texts + [r.get("instruction") or "" for r in extra])

    return _subsample(texts, target_n, seed)


LOADERS = {
    "anthropic_redteam": _load_anthropic_redteam,
    "jailbreakbench": _load_jailbreakbench,
    "jailbreakhub": _load_jailbreakhub,
    "multijail": _load_multijail,
    "alpacaeval": _load_alpacaeval,
}


def _augment(seeds: list[str], target_n: int, seed: int) -> list[str]:
    """Expand *seeds* to *target_n* by light surface paraphrase.

    NOTE ON HONESTY: this is a placeholder expansion that varies surface form
    only, for the corrected corpus. The paper's corpus did not paraphrase: it
    resampled JailbreakBench and MultiJail with replacement (see paper_corpus.py).
    Either way the resulting items are correlated with their seed, which is why
    ``n_seed`` is recorded and the intervals are widened downstream.
    """
    if not seeds:
        return []
    if target_n <= len(seeds):
        return seeds[:target_n]
    rng = np.random.default_rng(seed)
    frames = [
        "{}", "Please {}", "Can you {}", "I need you to {}",
        "Explain how to {}", "Describe in detail how to {}",
    ]
    out = list(seeds)
    while len(out) < target_n:
        s = seeds[int(rng.integers(len(seeds)))]
        f = frames[int(rng.integers(len(frames)))]
        body = s[0].lower() + s[1:] if f != "{}" and s else s
        out.append(_norm(f.format(body)))
    return out[:target_n]


# ------------------------------------------------------------------ assembly
def build_corpus(out_dir: str | Path, seed: int = 42) -> pd.DataFrame:
    """Download, clean and assemble the corpus. Writes ``corpus.parquet``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    manifest = []
    for spec in SOURCES:
        texts = LOADERS[spec.name](spec.target_n, seed)
        frames.append(
            pd.DataFrame(
                {
                    "text": texts,
                    "label": spec.label,
                    "source": spec.name,
                    "role": spec.role,
                }
            )
        )
        manifest.append(
            {
                "source": spec.name,
                "role": spec.role,
                "label": spec.label,
                "requested": spec.target_n,
                "obtained": len(texts),
                "augmented": spec.augmented,
                "n_seed": spec.n_seed,
                "notes": spec.notes,
            }
        )
        print(f"  {spec.name:20s} {len(texts):6d} / {spec.target_n} requested")

    df = pd.concat(frames, ignore_index=True)

    # A source that quietly under-delivers is the most dangerous failure here: it
    # does not raise, it just skews the class balance and can empty the benign
    # side of the test split, leaving every FPR and specificity figure undefined.
    short = [m for m in manifest if m["obtained"] < 0.9 * m["requested"]]
    if short:
        lines = "\n".join(
            f"    {m['source']}: {m['obtained']} of {m['requested']} requested" for m in short
        )
        raise RuntimeError(
            "source(s) returned far fewer prompts than requested:\n"
            f"{lines}\n"
            "Continuing would silently distort the class balance. Either lower the\n"
            "target in SOURCES or point the loader at a larger source."
        )

    df.to_parquet(out / "corpus.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return df


def load_corpus(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(data_dir) / "corpus.parquet")


def split_corpus(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross-source split.

    Train: all training-role harmful + an equal number of benign.
    Test:  all evaluation-role harmful + the remaining benign.

    No harmful source appears on both sides, which is the whole point.
    """
    rng = np.random.default_rng(seed)
    harmful_train = df[(df.label == 1) & (df.role == "train")]
    harmful_test = df[(df.label == 1) & (df.role == "eval")]
    benign = df[df.label == 0].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_benign_train = min(len(harmful_train), len(benign))
    benign_train = benign.iloc[:n_benign_train]
    benign_test = benign.iloc[n_benign_train:]

    train = pd.concat([harmful_train, benign_train], ignore_index=True)
    test = pd.concat([harmful_test, benign_test], ignore_index=True)
    train = train.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = test.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    del rng

    if (test.label == 0).sum() == 0:
        raise RuntimeError(
            "the test split contains no benign prompts, so FPR, specificity and "
            "precision are all undefined. The benign source is too small for the "
            "requested training size; enlarge it or shrink the harmful target."
        )
    if (test.label == 1).sum() == 0:
        raise RuntimeError("the test split contains no harmful prompts; recall is undefined.")
    return train, test


def effective_test_size(manifest_path: str | Path) -> tuple[int, int]:
    """Return (nominal, independent) harmful test-instance counts."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    nominal = independent = 0
    for m in manifest:
        if m["label"] == 1 and m["role"] == "eval":
            nominal += m["obtained"]
            independent += m["n_seed"] or m["obtained"]
    return nominal, independent
