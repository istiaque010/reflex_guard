# Datasheet: `reflexguard_corpus.csv`

The 30,568-prompt corpus used in the paper (Section IV-A, Table 3). This file is the
authoritative copy. Its SHA-256 is in `SHA256SUMS`; the file must keep LF line endings
for the checksum to match (`.gitattributes` marks it `-text`).

## Columns

| Column | Meaning |
|---|---|
| `Goal` | The prompt text |
| `label` | 1 = harmful, 0 = benign |
| `source` | Source label used by the data script (see below) |
| `category` | `Tuning` for the harmful training source, `Evaluation` otherwise |
| `priority` | Order used for cross-source deduplication (1 = kept first) |

## Composition and split

| `source` value | Actual source | Rows | Distinct prompts | Role |
|---|---|---:|---:|---|
| `AlpacaEval` | First 15,000 Stanford Alpaca instructions (`tatsu-lab/alpaca`), instruction plus optional input | 15,000 | 15,000 | Benign; 13,000 train, 2,000 test |
| `Anthropic_RedTeam` | Opening human turns of the Anthropic HH-RLHF training split | 13,000 | 13,000 | Harmful, training only |
| `JailbreakHub_DAN` | First 1,000 prompts of `walledai/JailbreakHub`, exact-deduplicated | 968 | 968 | Harmful, test only |
| `JailbreakBench` | The 100 JBB harmful behaviors, resampled with replacement | 800 | 100 | Harmful, test only |
| `MultiJail_BN` | See the MultiJail caveat below | 800 | 287 | Harmful, test only |

The paper trains on 26,000 prompts (13,000 harmful + 13,000 benign) and holds out 4,568
(2,568 harmful + 2,000 benign).

## Known issues (stated in Section IV-A of the revised manuscript)

1. **Mixed harmful training data.** HH-RLHF mixes red-team and helpfulness conversations,
   and the data script did not filter by subset. Only 3,605 of the 13,000 harmful
   training prompts come from the red-team subset, so the harmful labels are noisy.
2. **MultiJail holds record identifiers.** The MultiJail CSV has no `query`/`prompt`/
   `question` column, so the first column (`id`) was selected. The 800 MultiJail rows are
   the numbers 0 to 314 (287 distinct after resampling), not Bengali text. Results on this
   corpus therefore say nothing about Bengali or other non-English prompts.
3. **Identical copies.** JailbreakBench and MultiJail were resampled with replacement, so
   their copies are identical. The paper's confidence intervals use the effective sample
   size (1,383) for this reason.
4. **Benign source.** All benign prompts come from Stanford Alpaca, so false-positive rates
   on this corpus are in-distribution estimates.
5. **Truncation.** 337 of the 968 DAN prompts and 23 of the 13,000 HH-RLHF prompts exceed
   the encoder's 512-token limit and are truncated.

## Rebuilding

`python scripts/rebuild_paper_corpus.py` re-runs the procedure from the public sources and
compares the result with this file. The Alpaca and DAN sets, all per-source counts and the
Anthropic candidate pool are recovered exactly. The random draws (the 13,000 Anthropic
prompts and the resampled copies) are not recovered bit for bit, so use this file for any
comparison with the paper.

For a corpus without issues 1 to 3, use `reflex_guard.data.build_corpus`
(`python scripts/build_dataset.py`). Results on that corpus are not comparable with the
paper's tables.
