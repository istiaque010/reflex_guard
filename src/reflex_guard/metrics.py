"""Evaluation metrics, including the ones the paper argues are usually omitted.

Three things here are not in a standard metrics module and are the point of the
paper's methodological argument:

* :func:`wilson_interval` takes an **effective** sample size. Augmenting 415 seed
  prompts into 1,600 test instances makes those instances correlated; treating
  them as independent understates sampling error. Point estimates are unaffected,
  interval widths are not.
* :func:`ppv_at_prevalence` converts a measured operating point into the
  precision it would deliver at an arbitrary harmful-prompt prevalence. A
  balanced-benchmark precision does not transfer to deployment.
* :func:`res` and :func:`res_j` are the efficiency scores. ``res`` uses recall
  only and is therefore maximised by blocking everything; ``res_j`` substitutes
  Youden's J and charges for false positives. Report both.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "ConfusionCounts",
    "OperatingPoint",
    "confusion_at",
    "wilson_interval",
    "design_effect",
    "ppv_at_prevalence",
    "res",
    "res_j",
    "threshold_sweep",
]


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def positives(self) -> int:
        return self.tp + self.fn

    @property
    def negatives(self) -> int:
        return self.tn + self.fp


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    fpr: float
    youden_j: float
    counts: ConfusionCounts

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(d.pop("counts"))
        return d


def confusion_at(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> ConfusionCounts:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(scores) >= threshold).astype(int)
    return ConfusionCounts(
        tp=int(np.sum((pred == 1) & (y_true == 1))),
        fp=int(np.sum((pred == 1) & (y_true == 0))),
        tn=int(np.sum((pred == 0) & (y_true == 0))),
        fn=int(np.sum((pred == 0) & (y_true == 1))),
    )


def operating_point(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> OperatingPoint:
    c = confusion_at(y_true, scores, threshold)
    precision = c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 0.0
    recall = c.tp / c.positives if c.positives else 0.0
    fpr = c.fp / c.negatives if c.negatives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return OperatingPoint(
        threshold=threshold,
        accuracy=(c.tp + c.tn) / c.n if c.n else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        youden_j=recall - fpr,
        counts=c,
    )


def wilson_interval(p: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for proportion *p* observed with sample size *n*.

    Pass an **effective** ``n`` (nominal divided by the design effect) when the
    sample contains correlated items; the point estimate is unchanged and only
    the width grows.
    """
    if n <= 0:
        return (0.0, 1.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def design_effect(nominal: int, independent: int) -> float:
    """Ratio of nominal to independent sample size (>= 1)."""
    if independent <= 0:
        raise ValueError("independent sample size must be positive")
    return nominal / independent


def ppv_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    """Precision implied by an operating point at a given harmful prevalence.

    PPV(pi) = pi*TPR / (pi*TPR + (1-pi)*FPR)   -- Bayes' rule.
    """
    num = prevalence * tpr
    den = num + (1 - prevalence) * fpr
    return num / den if den else 0.0


def expected_cascade_latency(
    stage1_ms: float, stage2_ms: float, escalation_rate: float
) -> float:
    """Mean per-request latency of a two-stage cascade.

    Every request pays ``stage1_ms``; an ``escalation_rate`` fraction also pays
    ``stage2_ms``. This is the framing under which a high-recall pre-filter with
    a non-trivial false-positive rate is still worth deploying.
    """
    return stage1_ms + escalation_rate * stage2_ms


def res(recall: float, latency_ms: float) -> float:
    """Reflex Efficiency Score: recall-only. Maximised by blocking everything."""
    return recall * 100.0 / math.log2(latency_ms + 1.0)


def res_j(tpr: float, fpr: float, latency_ms: float) -> float:
    """False-positive-aware efficiency score using Youden's J = TPR - FPR."""
    return (tpr - fpr) * 100.0 / math.log2(latency_ms + 1.0)


def threshold_sweep(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: np.ndarray | list[float],
) -> list[OperatingPoint]:
    """Operating points across a grid of thresholds."""
    return [operating_point(y_true, scores, float(t)) for t in thresholds]


def f1_optimal_threshold(
    y_true: np.ndarray, scores: np.ndarray, grid: np.ndarray | None = None
) -> float:
    """Threshold maximising F1 (Eq. 11).

    Select this on training/validation folds only. Choosing it on the evaluation
    set and then reporting performance at it yields an in-sample number.
    """
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    points = threshold_sweep(y_true, scores, grid)
    return max(points, key=lambda p: p.f1).threshold
