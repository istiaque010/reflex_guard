# Reproducing the paper

## 0. Environment

The paper's runs used Python 3.10.12, scikit-learn 1.2.2, XGBoost 2.0.3,
LightGBM 4.0.0, sentence-transformers 2.2.2, transformers 4.35.2, PyTorch 2.1.0,
on a Tesla T4 (16 GB) with a dual-core Intel Xeon and 12 GB RAM.

Seeds are fixed to 42 everywhere: dataset sampling, the stratified split,
classifier initialisation, and the CUDA RNG. `train.py:set_seed()` also enables
deterministic cuDNN. Exact latency figures will still differ on other hardware —
that is expected and is the reason the paper does not treat cross-hardware
latency comparisons as controlled.

## 1. Corpus

`scripts/build_dataset.py` downloads five public sources. Hub identifiers live in
`src/reflex_guard/data.py::HF_IDS`; if one has moved, fix the corresponding
`_load_*` function and keep the returned schema (`text`, `label`, `source`,
`role`).

Two sources are augmented from a smaller seed pool:

  JailbreakBench   100 seeds -> 800 instances
  MultiJail (bn)   315 seeds -> 800 instances

`data.py::_augment` performs surface-form expansion only. The paper used
back-translation and synonym replacement; substitute a real paraphraser there if
you need the exact corpus. Either way the resulting items are correlated with
their seed, which is why `n_seed` is written to `manifest.json` and consumed by
`evaluate.py` to widen the confidence intervals. Do not delete that field.

## 2. Threshold protocol

`train.py` selects tau by maximising F1 over stratified CV folds of the TRAINING
partition. The test set is never used for selection.

If you later choose a sensitive threshold by inspecting attack scores, the recall
you quote at that threshold is IN-SAMPLE. Either say so explicitly, or hold out a
disjoint attack split for selection. `attack_eval.py` prints this warning.

## 3. GCG suffixes

Not shipped. Generate with:

    git clone https://github.com/llm-attacks/llm-attacks
    # optimise against Vicuna-7B-v1.3: 500 steps, batch size 512
    # paste the resulting suffixes, one per line, into data/attacks/gcg_suffixes.txt

The paper crosses 15 behaviours with 10 suffixes and evaluates 60 of the 150
combinations. `attack_eval.py` skips the GCG family if the file has no
non-comment lines.

## 4. Expected outputs

    artifacts/heldout_results.csv          Table 4 (with FPR and effective-n CIs)
    artifacts/prevalence_analysis.csv      Table 9
    artifacts/attack_results.csv           Table 6
    artifacts/threshold_sweep.csv          Tables 10, 11
    artifacts/latency_results.csv          Table 7 (incl. P95/P99)
    artifacts/representation_baselines.csv the missing control
    artifacts/structural_detector_eval.csv the missing router evaluation

## 5. If your numbers differ

Likely causes, in order:

1. Different corpus. The augmentation step is not identical to the paper's.
2. Different embedding prefix. `--prefix ''` vs the paper's `query:` moves
   results; measure it rather than assuming.
3. Different hardware. Affects latency only, not detection.
4. A Hub dataset changed upstream. Check `manifest.json` counts against Table 3.

Report what you measured, not what the paper reported.
