"""CPU latency of the Reflex-Guard pipeline (Section IV-J, Table 8).

    python scripts/cpu_latency.py

This is the script that produced Table 8 and results/cpu_latency/latency_cpu_local.json.

Classifiers are trained on a 26,000-prompt training split of data/corpus/reflexguard_corpus.csv
(the 13,000 Anthropic prompts plus 13,000 benign prompts drawn with seed 42; the remaining 4,568
prompts form the held-out set), with the Table 2 hyperparameters. The original benign train/test
assignment was not preserved; latency does not depend on it. The "_hardware" entry below names the
machine used for Table 8; edit it when you run the script elsewhere.
Latency follows the paper's protocol (batch size 1, preprocessing + "query: " embedding +
classification + threshold), extended to 3 runs x 1,000 held-out prompts, with a bootstrap CI
for P99 and a breakdown by input length. Two settings: all CPU threads, and 2 threads.
Writes results/cpu_latency/latency_cpu_local.json.
"""
import base64
import json
import os
import platform
import re
import time

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

SEED = 42
TAU = 0.25
HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "corpus", "reflexguard_corpus.csv")
OUT = os.path.join(HERE, "..", "results", "cpu_latency", "latency_cpu_local.json")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


df = pd.read_csv(CSV, keep_default_na=False)
df["Goal"] = df["Goal"].astype(str)
benign = df[df.label == 0].sample(frac=1, random_state=SEED)
train = pd.concat([df[df.source == "Anthropic_RedTeam"], benign.iloc[:13000]])
test = pd.concat([df[df.source.isin(["JailbreakHub_DAN", "JailbreakBench", "MultiJail_BN"])], benign.iloc[13000:]])
log("train", len(train), "test", len(test))

B64 = re.compile(r'(?:[A-Za-z0-9+/]{4}){4,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?')


def preprocess(text):
    out = text
    for m in B64.findall(text):
        try:
            out = out.replace(m, f" [DECODED: {base64.b64decode(m).decode('utf-8')}] ")
        except Exception:
            pass
    return out


enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
X = enc.encode(["query: " + t for t in train.Goal], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
y = train.label.values
pos_w = (len(y) - y.sum()) / y.sum()
models = {
    "Logistic Regression": LogisticRegression(solver="saga", max_iter=1000, class_weight="balanced", n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=500, min_samples_split=5, min_samples_leaf=2,
                                            class_weight="balanced", n_jobs=-1, random_state=SEED),
    "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=SEED),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, random_state=SEED),
    "LightGBM": LGBMClassifier(n_estimators=500, learning_rate=0.05, class_weight="balanced", n_jobs=-1,
                               random_state=SEED, verbose=-1),
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             objective="binary:logistic", eval_metric="logloss", scale_pos_weight=pos_w,
                             n_jobs=-1, random_state=SEED),
}
for n, m in models.items():
    t0 = time.time()
    m.fit(X, y)
    log("trained", n, f"{time.time() - t0:.0f}s")

rng = np.random.default_rng(SEED)
prompts = test.Goal.iloc[rng.choice(len(test), size=1000, replace=False)].tolist()
ntok = [len(enc.tokenizer("query: " + preprocess(p), truncation=True, max_length=512)["input_ids"]) for p in prompts]
BUCKETS = [("<=64 tokens", 0, 64), ("65-256 tokens", 65, 256), ("257-512 tokens", 257, 512)]


def stats(a):
    a = np.asarray(a)
    return {"n": int(len(a)), "min": round(float(a.min()), 2), "mean": round(float(a.mean()), 2),
            "p50": round(float(np.percentile(a, 50)), 2), "p95": round(float(np.percentile(a, 95)), 2),
            "p99": round(float(np.percentile(a, 99)), 2)}


out = {"_hardware": {"cpu": "Intel Core Ultra 7 270K Plus (24 cores)", "ram_gb": 64, "os": platform.platform(),
                     "python": platform.python_version(), "torch": torch.__version__},
       "_protocol": "batch size 1; 20 warm-up passes; 3 runs x 1,000 held-out prompts; preprocessing, embedding, "
                    "classification and threshold timed together",
       "_token_buckets": {b: int(sum(lo <= t <= hi for t in ntok)) for b, lo, hi in BUCKETS}}
for setting, threads in [("all_threads", os.cpu_count()), ("2_threads", 2)]:
    torch.set_num_threads(threads)
    for m in models.values():
        for attr in ("n_jobs",):
            if hasattr(m, attr):
                setattr(m, attr, threads)
    for _ in range(20):
        enc.encode("query: warmup", normalize_embeddings=True)
    res = {}
    for name, model in models.items():
        runs = []
        for _ in range(3):
            ms = []
            for p in prompts:
                s = time.perf_counter()
                v = enc.encode("query: " + preprocess(p), normalize_embeddings=True, convert_to_numpy=True).reshape(1, -1)
                _ = model.predict_proba(v)[:, 1] >= TAU
                ms.append((time.perf_counter() - s) * 1000)
            runs.append(ms)
        allms = np.concatenate(runs)
        boot = [np.percentile(rng.choice(allms, size=len(allms)), 99) for _ in range(2000)]
        pooled = stats(allms)
        pooled["p99_ci"] = [round(float(np.percentile(boot, 2.5)), 2), round(float(np.percentile(boot, 97.5)), 2)]
        tok3 = np.array(ntok * 3)
        res[name] = {"pooled": pooled, "per_run_mean": [round(float(np.mean(r)), 2) for r in runs],
                     "per_run_p99": [round(float(np.percentile(r, 99)), 2) for r in runs],
                     "by_length": {b: stats(allms[(tok3 >= lo) & (tok3 <= hi)]) for b, lo, hi in BUCKETS}}
        log(setting, name, "mean", pooled["mean"], "p99", pooled["p99"])
    # embedding-only cost
    em = []
    for p in prompts:
        s = time.perf_counter()
        enc.encode("query: " + preprocess(p), normalize_embeddings=True)
        em.append((time.perf_counter() - s) * 1000)
    res["_embedding_only"] = stats(em)
    out[setting] = {"threads": threads, "results": res}
json.dump(out, open(OUT, "w"), indent=1)
log("saved", OUT)
