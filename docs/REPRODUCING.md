# Reproducing the paper

## 0. Environment

The original runs used Google Colab with Python 3.10.12, scikit-learn 1.2.2, XGBoost 2.0.3,
LightGBM 4.0.0, sentence-transformers 2.2.2, transformers 4.35.2 and PyTorch 2.1.0, on an
NVIDIA Tesla T4 (16 GB). The CPU latency of Table 8 was measured on a desktop computer
without a GPU (Intel Core Ultra 7 270K Plus, 64 GB RAM). Seeds are fixed to 42.

## 1. Corpus

The corpus behind the paper's tables is `data/corpus/reflexguard_corpus.csv`. Check it with
`cd data/corpus && sha256sum -c SHA256SUMS`, and read `data/corpus/DATASHEET.md` for its sources
and known issues. Convert it for the pipeline with

    python scripts/build_dataset.py --paper --out data/paper/

which gives the paper's split: 26,000 training prompts (13,000 harmful, 13,000 benign) and
4,568 held-out prompts (2,568 harmful, 2,000 benign). The manifest records the seed pools of
the resampled sources (JailbreakBench 100, MultiJail 315), so `evaluate.py` reports Wilson
intervals at the effective sample size of 1,383.

`scripts/rebuild_paper_corpus.py` re-runs the corpus procedure from the public sources and
compares the result with the released file. The random draws are not recovered bit for bit,
so always use the released file to compare with the paper.

## 2. Threshold protocol

`train.py` selects tau by maximising F1 (Eq. 8) over stratified cross-validation folds of
the training partition. The test set is not used for selection.

The sensitive threshold tau = 0.03 in Table 7 was chosen by inspecting the DrAttack-style
scores, so the recall quoted at that threshold is in-sample. `attack_eval.py` prints this
warning.

## 3. Attack inputs

`data/attacks/behaviors.txt` (15 seed behaviors) and `data/attacks/gcg_suffixes.txt`
(10 GCG-style suffixes) are the original inputs of Table 5. The suffixes are fixed and
manually constructed; they were not optimised against a model. The GCG-style family crosses
the 15 behaviors with the 10 suffixes and evaluates 60 of the 150 combinations (seed 42);
the DrAttack-style family embeds the behaviors in six benign templates (60 prompts). Both
can be regenerated exactly. The 100 Base64 prompts were not preserved. See
`data/attacks/ATTACK_CONFIGURATION.md`.

## 4. Reported results

The numbers printed in the article are in `results/paper_tables/` (Tables 4 to 8). The
trained classifiers and raw outputs of the original runs behind Tables 4 to 7 were not
preserved, so a fresh run is not expected to match them exactly. Table 8 is reproduced by
`scripts/cpu_latency.py`; its raw output is in `results/cpu_latency/`.
