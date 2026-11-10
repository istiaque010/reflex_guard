"""Rebuild the 30,568-prompt corpus used in the paper (Section IV-A, Table 3).

This module reproduces the procedure that produced ``data/corpus/reflexguard_corpus.csv``,
including its known flaws, so that the released file can be checked rather than trusted.
The flaws are documented in Section IV-A of the revised manuscript and in
``data/corpus/DATASHEET.md``:

* The harmful training prompts are the opening human turns of the full Anthropic
  HH-RLHF training split, which mixes red-team and helpfulness conversations.
* JailbreakBench (100 behaviors) and MultiJail (315 rows) are resampled with
  replacement to 800 rows each, so the copies are identical.
* MultiJail has no ``query``/``prompt``/``question`` column and no ``language``
  column, so the first column (``id``) was selected. The 800 MultiJail rows therefore
  hold record identifiers, not Bengali text.
* The benign prompts are the first 15,000 Stanford Alpaca instructions (instruction
  plus optional input), labelled ``AlpacaEval`` in the file.

For a corpus without these flaws, use :func:`reflex_guard.data.build_corpus`.
Only ``huggingface_hub`` and ``pandas`` are needed; no ``datasets`` install.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

SEED = 42
PAPER_CORPUS_SHA256 = "b5618cea0120e15666eab5160bd22f38b73f159724797f0f7cbff96f64ec9eaa"
PRIORITY = {"JailbreakBench": 1, "MultiJail_BN": 2, "JailbreakHub_DAN": 3, "Anthropic_RedTeam": 4, "AlpacaEval": 5}
HH_RLHF_TRAIN = [  # the files ``load_dataset("Anthropic/hh-rlhf", split="train")`` reads, in its order
    "harmless-base/train.jsonl.gz",
    "helpful-base/train.jsonl.gz",
    "helpful-online/train.jsonl.gz",
    "helpful-rejection-sampled/train.jsonl.gz",
]


def _hub(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset")


def _first_turn(text: str) -> str:
    if "\n\nAssistant:" in text:
        return text.split("\n\nAssistant:")[0].replace("\n\nHuman:", "").strip()
    return text.replace("\n\nHuman:", "").strip()


def _anthropic() -> pd.DataFrame:
    goals = []
    for f in HH_RLHF_TRAIN:
        with gzip.open(_hub("Anthropic/hh-rlhf", f), "rt", encoding="utf-8") as fh:
            goals += [_first_turn(json.loads(line)["chosen"]) for line in fh]
    df = pd.DataFrame({"Goal": goals})
    df = df[df["Goal"].str.len() > 10][["Goal"]].copy()
    df["label"], df["source"], df["category"] = 1, "Anthropic_RedTeam", "Tuning"
    return df


def _jailbreakbench() -> pd.DataFrame:
    df = pd.read_csv(_hub("JailbreakBench/JBB-Behaviors", "data/harmful-behaviors.csv"))[["Goal"]].copy()
    df["label"], df["source"], df["category"] = 1, "JailbreakBench", "Evaluation"
    return df


def _jailbreakhub() -> pd.DataFrame:
    raw = pd.read_parquet(_hub("walledai/JailbreakHub", "data/train-00000-of-00001.parquet"))
    df = raw.rename(columns={"prompt": "Goal"})[["Goal"]].head(1000).copy()
    df["label"], df["source"], df["category"] = 1, "JailbreakHub_DAN", "Evaluation"
    return df


def _multijail() -> pd.DataFrame:
    raw = pd.read_csv(_hub("DAMO-NLP-SG/MultiJail", "MultiJail.csv"), encoding="utf-8-sig")
    # Paper behaviour: no query/prompt/question column exists, so the first column (id) was used.
    col = next((c for c in ["query", "prompt", "question"] if c in raw.columns), raw.columns[0])
    df = raw.rename(columns={col: "Goal"})[["Goal"]].head(315).copy()
    df["label"], df["source"], df["category"] = 1, "MultiJail_BN", "Evaluation"
    return df


def _alpaca() -> pd.DataFrame:
    raw = pd.read_parquet(_hub("tatsu-lab/alpaca", "data/train-00000-of-00001-a09b74b3ef9c3b56.parquet"))
    raw["Goal"] = raw["instruction"].fillna("") + " " + raw["input"].fillna("")
    df = raw[raw["Goal"].str.len() > 10].head(15000)[["Goal"]].copy()
    df["label"], df["source"], df["category"] = 0, "AlpacaEval", "Evaluation"
    return df


def build_paper_corpus() -> pd.DataFrame:
    """Return the corpus exactly as the paper's data script assembled it."""
    raw = pd.concat([_anthropic(), _jailbreakbench(), _jailbreakhub(), _multijail(), _alpaca()], ignore_index=True)
    raw["priority"] = raw["source"].map(PRIORITY)
    clean = raw.sort_values("priority").drop_duplicates(subset=["Goal"], keep="first")
    pick = lambda s: clean[clean["source"] == s]  # noqa: E731
    parts = [
        pick("JailbreakBench").sample(n=800, replace=True, random_state=SEED),
        pick("MultiJail_BN").sample(n=800, replace=True, random_state=SEED),
        pick("JailbreakHub_DAN"),
        pick("Anthropic_RedTeam").sample(n=13000, random_state=SEED),
        pick("AlpacaEval"),
    ]
    return pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)


ROLE = {"Anthropic_RedTeam": "train", "AlpacaEval": "benign", "JailbreakHub_DAN": "eval",
        "JailbreakBench": "eval", "MultiJail_BN": "eval"}
N_SEED = {"JailbreakBench": 100, "MultiJail_BN": 315}  # seed pools behind the resampled sources
SOURCE_NAME = {"AlpacaEval": "Alpaca"}  # correct name for the mislabelled benign source


def export_for_pipeline(csv_path: str | Path, out_dir: str | Path) -> pd.DataFrame:
    """Convert the released CSV to the ``corpus.parquet`` + ``manifest.json`` layout that
    ``scripts/train.py`` and ``scripts/evaluate.py`` read. The manifest carries ``n_seed``
    so the evaluation uses the effective sample size (1,383), as in the paper."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(csv_path, keep_default_na=False)
    # The released CSV labels the benign prompts "AlpacaEval", an error in the original data
    # script; they are Stanford Alpaca. The CSV keeps the old label so its SHA-256 still matches,
    # and the pipeline uses the correct name.
    df = pd.DataFrame({"text": src["Goal"].astype(str), "label": src["label"].astype(int),
                       "source": src["source"].replace(SOURCE_NAME), "role": src["source"].map(ROLE)})
    manifest = []
    for name, part in df.groupby("source", sort=False):
        manifest.append({"source": name, "role": part["role"].iloc[0], "label": int(part["label"].iloc[0]),
                         "requested": len(part), "obtained": len(part),
                         "augmented": name in N_SEED, "n_seed": N_SEED.get(name),
                         "notes": "released paper corpus; see data/corpus/DATASHEET.md"})
    df.to_parquet(out / "corpus.parquet", index=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return df


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same_rows(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """True if both frames hold the same multiset of rows, ignoring order."""
    key = ["Goal", "label", "source", "category", "priority"]
    fa = a[key].astype(str).value_counts().sort_index()
    fb = b[key].astype(str).value_counts().sort_index()
    return fa.equals(fb)
