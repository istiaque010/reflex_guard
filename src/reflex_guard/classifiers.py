"""Classifier zoo with the hyperparameters of Table 2.

Every constructor is seeded from a single ``seed`` argument so that a run is
reproducible end to end. XGBoost and LightGBM are optional dependencies; if they
are absent the corresponding factories raise a clear ImportError rather than
silently disappearing from the comparison.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

__all__ = ["CLASSIFIERS", "build_classifier", "available_classifiers"]


def _logistic_regression(seed: int):
    return LogisticRegression(
        solver="saga",
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def _random_forest(seed: int):
    return RandomForestClassifier(
        n_estimators=500,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )


def _xgboost(seed: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError("XGBoost is required for the 'xgboost' classifier: pip install xgboost") from exc
    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        # scale_pos_weight is set at fit time from the observed class ratio;
        # see ReflexGuard.fit. 1.0 here keeps the estimator picklable standalone.
        scale_pos_weight=1.0,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def _lightgbm(seed: int):
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LightGBM is required for the 'lightgbm' classifier: pip install lightgbm") from exc
    return LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )


def _hist_gradient_boosting(seed: int):
    return HistGradientBoostingClassifier(
        max_iter=500,
        learning_rate=0.05,
        random_state=seed,
    )


def _adaboost(seed: int):
    return AdaBoostClassifier(
        n_estimators=300,
        learning_rate=0.5,
        random_state=seed,
    )


def _knn(seed: int):  # noqa: ARG001 - KNN is deterministic
    return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)


CLASSIFIERS: dict[str, Callable[[int], object]] = {
    "logistic_regression": _logistic_regression,
    "random_forest": _random_forest,
    "xgboost": _xgboost,
    "lightgbm": _lightgbm,
    "hist_gradient_boosting": _hist_gradient_boosting,
    "adaboost": _adaboost,
    "knn": _knn,
}

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "hist_gradient_boosting": "HistGradientBoosting",
    "adaboost": "AdaBoost",
    "knn": "KNN (k=5)",
}


def available_classifiers() -> list[str]:
    """Names that can actually be built in this environment."""
    ok = []
    for name, factory in CLASSIFIERS.items():
        try:
            factory(0)
            ok.append(name)
        except ImportError:
            continue
    return ok


def build_classifier(name: str, seed: int = 42, y: np.ndarray | None = None):
    """Instantiate a classifier by name.

    If *y* is supplied and the estimator is XGBoost, ``scale_pos_weight`` is set
    to the negative/positive ratio, which is the 'auto' of Table 2.
    """
    if name not in CLASSIFIERS:
        raise KeyError(f"unknown classifier {name!r}; choose from {sorted(CLASSIFIERS)}")
    clf = CLASSIFIERS[name](seed)
    if name == "xgboost" and y is not None:
        pos = int(np.sum(y == 1))
        neg = int(np.sum(y == 0))
        if pos:
            clf.set_params(scale_pos_weight=neg / pos)
    return clf
