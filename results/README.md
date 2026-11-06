# Results

## `paper_tables/`: results as reported in the article

These files hold the numbers printed in the revised manuscript, converted from its tables.
They are the reference values for comparison, not the output of a fresh run.

| File | Article | Content |
|---|---|---|
| `table4_heldout.csv` | Table 4 | Held-out performance of the seven classifiers at tau = 0.25, with 95% Wilson intervals (effective sample size), FPR, F1, ROC-AUC, RES and RES_J |
| `table5_attacks.csv` | Table 5 | Recall of HistGradientBoosting on the GCG-style, Base64 and DrAttack-style families at tau = 0.25 and 0.03, with mean probability (SD) |
| `table6_latency_t4_gpu.csv` | Table 6 | End-to-end latency over 100 prompts on an NVIDIA Tesla T4 GPU, batch size 1 |
| `table7_threshold_sweep.csv` | Table 7 | Threshold sweep for HistGradientBoosting: DrAttack-style recall, general recall and benign FPR |
| `table8_latency_cpu.csv` | Table 8 | End-to-end CPU latency over 3,000 requests (all threads and two threads), with a 95% bootstrap interval for P99 |

Provenance:

- Tables 4 to 7 come from the original runs on Google Colab. The trained classifiers and
  raw evaluation outputs of those runs were not preserved, so the values cannot be
  regenerated bit for bit from this repository. A newly trained classifier is not expected
  to give identical numbers.
- The attack inputs behind Table 5 (15 seed behaviors and 10 GCG-style suffixes) are in
  `data/attacks/`. The 60 GCG-style and 60 DrAttack-style prompts can be regenerated
  exactly from them; the 100 Base64 prompts were not preserved
  (see `data/attacks/ATTACK_CONFIGURATION.md`).
- The DrAttack-style recall at tau = 0.03 is in-sample: it is measured on the same prompts
  used to choose that threshold.

## `cpu_latency/`: raw output behind Table 8

`latency_cpu_local.json` is the unedited output of `scripts/cpu_latency.py` on the machine
named in its `_hardware` entry (Intel Core Ultra 7 270K Plus, 64 GB RAM, no GPU). It
contains per-run means and P99 values, the pooled statistics, the P99 bootstrap interval,
the breakdown by input length, and the embedding-only cost. Latency depends on hardware,
so a rerun on another machine will give different numbers.
