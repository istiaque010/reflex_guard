"""End-to-end Reflex-Guard pipeline: preprocess -> embed -> classify -> threshold."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .classifiers import build_classifier
from .embedding import PAPER_PREFIX, BGEEmbedder
from .preprocessing import JailbreakAwarePreprocessor, PreprocessResult

__all__ = ["Decision", "ReflexGuard"]

DEFAULT_THRESHOLD = 0.25
SENSITIVE_THRESHOLD = 0.03


@dataclass
class Decision:
    """Per-prompt guardrail decision."""

    blocked: bool
    score: float
    threshold_used: float
    preprocessing: PreprocessResult

    @property
    def action(self) -> str:
        return "BLOCK" if self.blocked else "ALLOW"


class ReflexGuard:
    """Embedding-based prompt guardrail.

    Parameters
    ----------
    classifier:
        Name from :mod:`reflex_guard.classifiers`, or a pre-fitted estimator
        exposing ``predict_proba``.
    threshold:
        Default operating point. Select it on training/validation folds, never on
        the evaluation set.
    two_tier:
        If True, prompts whose structural-pattern flag fires are scored against
        ``sensitive_threshold`` instead of ``threshold``.

        This policy is NOT validated. The structural detector's own false-positive
        rate is unmeasured in the paper, and applying the sensitive threshold to
        all traffic roughly doubles the benign false-positive rate. Measure your
        detector with ``scripts/structural_detector_eval.py`` before enabling it.
    """

    def __init__(
        self,
        classifier: str | object = "hist_gradient_boosting",
        threshold: float = DEFAULT_THRESHOLD,
        two_tier: bool = False,
        sensitive_threshold: float = SENSITIVE_THRESHOLD,
        embedder: BGEEmbedder | None = None,
        preprocessor: JailbreakAwarePreprocessor | None = None,
        seed: int = 42,
        prefix: str = PAPER_PREFIX,
    ) -> None:
        self.classifier_name = classifier if isinstance(classifier, str) else type(classifier).__name__
        self.clf = build_classifier(classifier, seed=seed) if isinstance(classifier, str) else classifier
        self.threshold = threshold
        self.two_tier = two_tier
        self.sensitive_threshold = sensitive_threshold
        self.seed = seed
        self._prefix = prefix
        self._embedder = embedder
        self.pre = preprocessor or JailbreakAwarePreprocessor()
        self._fitted = not isinstance(classifier, str)

    # ------------------------------------------------------------- lazy embed
    @property
    def embedder(self) -> BGEEmbedder:
        if self._embedder is None:
            self._embedder = BGEEmbedder(prefix=self._prefix)
        return self._embedder

    # -------------------------------------------------------------- transform
    def embed_texts(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Preprocess then embed. This is the featurisation used everywhere."""
        processed = [self.pre.process(t).text for t in texts]
        return self.embedder.encode(processed, show_progress=show_progress)

    # ------------------------------------------------------------------- fit
    def fit(self, texts: list[str] | None = None, y=None, *, X: np.ndarray | None = None) -> ReflexGuard:
        """Fit on raw texts (embedded here) or on precomputed embeddings *X*."""
        if X is None:
            if texts is None:
                raise ValueError("provide either texts or X")
            X = self.embed_texts(texts, show_progress=True)
        y = np.asarray(y).astype(int)
        if self.classifier_name == "xgboost":
            self.clf = build_classifier("xgboost", seed=self.seed, y=y)
        self.clf.fit(X, y)
        self._fitted = True
        return self

    # --------------------------------------------------------------- predict
    def score_embeddings(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ReflexGuard is not fitted")
        return self.clf.predict_proba(X)[:, 1]

    def score(self, texts: list[str] | str, show_progress: bool = False) -> np.ndarray:
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        probs = self.score_embeddings(self.embed_texts(items, show_progress))
        return probs[0] if single else probs

    def decide(self, text: str) -> Decision:
        """Full single-prompt path, including the two-tier threshold if enabled."""
        pr = self.pre.process(text)
        vec = self.embedder.encode([pr.text])
        prob = float(self.score_embeddings(vec)[0])
        tau = self.sensitive_threshold if (self.two_tier and pr.structural_flag) else self.threshold
        return Decision(blocked=prob >= tau, score=prob, threshold_used=tau, preprocessing=pr)

    def decide_batch(self, texts: list[str]) -> list[Decision]:
        results = self.pre.process_batch(texts)
        vecs = self.embedder.encode([r.text for r in results])
        probs = self.score_embeddings(vecs)
        out = []
        for pr, prob in zip(results, probs, strict=True):
            tau = self.sensitive_threshold if (self.two_tier and pr.structural_flag) else self.threshold
            out.append(Decision(bool(prob >= tau), float(prob), tau, pr))
        return out

    # ------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.clf, d / "classifier.joblib")
        (d / "config.json").write_text(
            json.dumps(
                {
                    "classifier_name": self.classifier_name,
                    "threshold": self.threshold,
                    "sensitive_threshold": self.sensitive_threshold,
                    "two_tier": self.two_tier,
                    "seed": self.seed,
                    "prefix": self._prefix,
                    "embedding_model": self.embedder.model_name if self._embedder else None,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path, embedder: BGEEmbedder | None = None) -> ReflexGuard:
        d = Path(directory)
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        guard = cls(
            classifier=joblib.load(d / "classifier.joblib"),
            threshold=cfg["threshold"],
            two_tier=cfg["two_tier"],
            sensitive_threshold=cfg["sensitive_threshold"],
            embedder=embedder,
            seed=cfg["seed"],
            prefix=cfg.get("prefix", PAPER_PREFIX),
        )
        guard.classifier_name = cfg["classifier_name"]
        return guard
