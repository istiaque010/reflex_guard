#!/usr/bin/env python
"""Non-embedding controls: does the dense encoder actually earn its place?

THIS IS THE EXPERIMENT THE PAPER IS MISSING. Without it, a result of "embeddings
plus a classifier works well" cannot be attributed to the *embedding*, because no
simpler representation was tried. Three controls, same split, same classifiers,
same threshold protocol:

  1. keyword / regex blocklist      - no learning at all
  2. TF-IDF word+char n-grams       - classical sparse lexical features
  3. hashed bag-of-words            - cheapest possible dense-ish baseline

Run this before submitting. If TF-IDF lands within a point or two of BGE, that is
a finding and belongs in the paper; it does not invalidate the work, but it does
change what the work claims.

    python scripts/representation_baselines.py --artifacts artifacts/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reflex_guard.classifiers import DISPLAY_NAMES, build_classifier  # noqa: E402
from reflex_guard.metrics import f1_optimal_threshold, operating_point  # noqa: E402
from reflex_guard.preprocessing import JailbreakAwarePreprocessor  # noqa: E402

BLOCKLIST = [
    "bomb", "explosive", "detonat", "malware", "ransomware", "keylogger",
    "exploit", "sql injection", "phishing", "counterfeit", "launder",
    "untraceable", "poison", "toxin", "silencer", "meth", "fentanyl",
    "child", "hack into", "steal", "kill", "murder", "weapon", "firearm",
    "ddos", "botnet", "backdoor", "credit card", "ssn", "bypass",
]


def keyword_scores(texts: list[str]) -> np.ndarray:
    """Fraction of blocklist terms present. A deliberately naive control."""
    low = [t.lower() for t in texts]
    return np.array([sum(k in t for k in BLOCKLIST) / len(BLOCKLIST) for t in low])


def make_tfidf() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                                 sublinear_tf=True, strip_accents="unicode")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 max_features=200_000, sublinear_tf=True)),
    ])


#: estimators that accept scipy sparse input directly. For the others we reduce
#: with TruncatedSVD rather than densifying: a 2^18-wide hashed matrix densified
#: to float32 is ~1.2 GB for only 1,200 rows, which is both slow and pointless.
SPARSE_OK = {"logistic_regression"}
SVD_DIM = 384          # matches the BGE dimensionality, so the comparison is fair


def _prepare(Xtr, Xte, cname, seed):
    """Return (Xtr, Xte, note) shaped for *cname*."""
    import scipy.sparse as sp

    already_dense = not sp.issparse(Xtr)
    if already_dense:
        return Xtr, Xte, "dense"          # BGE embeddings need no transform
    if cname in SPARSE_OK:
        return Xtr, Xte, "sparse"
    from sklearn.decomposition import TruncatedSVD

    k = min(SVD_DIM, min(Xtr.shape) - 1)
    svd = TruncatedSVD(n_components=k, random_state=seed)
    return svd.fit_transform(Xtr), svd.transform(Xte), f"svd{k}"


def evaluate(name, Xtr, ytr, Xte, yte, clf_names, seed, folds):
    rows = []
    for cname in clf_names:
        Xa, Xb, note = _prepare(Xtr, Xte, cname, seed)
        oof = np.zeros(len(ytr), dtype=float)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for tr_i, va_i in skf.split(np.zeros(len(ytr)), ytr):
            clf = build_classifier(cname, seed=seed, y=ytr[tr_i])
            clf.fit(Xa[tr_i], ytr[tr_i])
            oof[va_i] = clf.predict_proba(Xa[va_i])[:, 1]
        tau = f1_optimal_threshold(ytr, oof)

        clf = build_classifier(cname, seed=seed, y=ytr)
        clf.fit(Xa, ytr)
        op = operating_point(yte, clf.predict_proba(Xb)[:, 1], tau)
        rows.append({
            "representation": name,
            "features": note,
            "classifier": DISPLAY_NAMES.get(cname, cname),
            "tau": round(tau, 3),
            "accuracy": round(op.accuracy, 4),
            "precision": round(op.precision, 4),
            "recall": round(op.recall, 4),
            "fpr": round(op.fpr, 4),
            "f1": round(op.f1, 4),
        })
        print(f"  {name:>10s} + {DISPLAY_NAMES.get(cname, cname):22s} "
              f"F1={op.f1:.4f}  rec={op.recall:.4f}  fpr={op.fpr:.4f}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--classifiers", nargs="*",
                    default=["logistic_regression", "hist_gradient_boosting"])
    args = ap.parse_args()

    art = Path(args.artifacts)
    train = pd.read_parquet(art / "train.parquet")
    test = pd.read_parquet(art / "test.parquet")
    pre = JailbreakAwarePreprocessor()

    tr_txt = [pre.process(t).text for t in train.text.tolist()]
    te_txt = [pre.process(t).text for t in test.text.tolist()]
    ytr, yte = train.label.to_numpy(), test.label.to_numpy()

    rows = []

    print("\n[1/4] keyword blocklist (no learning)")
    ks = keyword_scores(te_txt)
    best = max((operating_point(yte, ks, t) for t in np.linspace(0.001, 0.2, 200)),
               key=lambda o: o.f1)
    print(f"  best-case F1={best.f1:.4f} rec={best.recall:.4f} fpr={best.fpr:.4f} "
          f"(threshold tuned ON the test set, so this is an upper bound)")
    rows.append({"representation": "keyword", "classifier": "threshold rule",
                 "tau": round(best.threshold, 4), "accuracy": round(best.accuracy, 4),
                 "precision": round(best.precision, 4), "recall": round(best.recall, 4),
                 "fpr": round(best.fpr, 4), "f1": round(best.f1, 4)})

    print("\n[2/4] TF-IDF word(1,2) + char_wb(3,5)")
    vec = make_tfidf()
    Xtr = vec.fit_transform(tr_txt).astype(np.float32)
    Xte = vec.transform(te_txt).astype(np.float32)
    print(f"  feature dim = {Xtr.shape[1]:,}")
    rows += evaluate("tfidf", Xtr, ytr, Xte, yte, args.classifiers, args.seed, args.folds)
    del Xtr, Xte

    print("\n[3/4] hashed bag-of-words (2^18)")
    hv = HashingVectorizer(n_features=2**18, alternate_sign=False, norm="l2")
    Xtr = hv.transform(tr_txt).astype(np.float32)
    Xte = hv.transform(te_txt).astype(np.float32)
    rows += evaluate("hashing", Xtr, ytr, Xte, yte, args.classifiers, args.seed, args.folds)
    del Xtr, Xte

    print("\n[4/4] BGE embeddings (from scripts/train.py)")
    blob = np.load(art / "embeddings.npz")
    rows += evaluate("bge", blob["Xtr"], blob["ytr"], blob["Xte"], blob["yte"],
                     args.classifiers, args.seed, args.folds)

    df = pd.DataFrame(rows)
    df.to_csv(art / "representation_baselines.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== representation comparison ===")
    print(df.sort_values("f1", ascending=False).to_string(index=False))

    bge = df[df.representation == "bge"].f1.max()
    tfidf = df[df.representation == "tfidf"].f1.max()
    print(f"\nbest BGE F1 = {bge:.4f} | best TF-IDF F1 = {tfidf:.4f} | "
          f"delta = {bge - tfidf:+.4f}")
    print(
        "Interpretation: the delta is what the dense semantic representation buys\n"
        "over sparse lexical features on THIS corpus. Report it honestly, whatever\n"
        "sign it has."
    )
    print(f"wrote {art}/representation_baselines.csv")


if __name__ == "__main__":
    main()
