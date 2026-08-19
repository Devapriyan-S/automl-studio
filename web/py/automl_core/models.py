"""The candidate model zoo.

Every estimator here ships with scikit-learn and therefore also works inside
Pyodide (WebAssembly), which is what lets the browser demo train real models
with no backend. That constraint rules out XGBoost/LightGBM, so gradient
boosting is provided by :class:`HistGradientBoostingClassifier`, which is
competitive with them on tabular data of this size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from .task import BINARY, MULTICLASS, REGRESSION, TaskSpec


@dataclass
class Candidate:
    key: str
    label: str
    estimator: Any
    # Cheap models run in the "fast" preset; everything runs in "full".
    tier: str  # fast | full
    blurb: str


def _classifiers(spec: TaskSpec, seed: int) -> list[Candidate]:
    # class_weight only helps when the classes are actually skewed; forcing it
    # on balanced data just adds variance.
    cw = "balanced" if spec.is_imbalanced else None
    return [
        Candidate(
            "logreg", "Logistic Regression",
            LogisticRegression(max_iter=2000, class_weight=cw, random_state=seed),
            "fast", "Linear baseline — fast, interpretable coefficients.",
        ),
        Candidate(
            "rf", "Random Forest",
            RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                   class_weight=cw, random_state=seed, n_jobs=1),
            "fast", "Bagged trees — strong default, resists overfitting.",
        ),
        Candidate(
            "hgb", "Gradient Boosting",
            HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                           early_stopping=True, random_state=seed),
            "fast", "Boosted trees — usually the top scorer on tabular data.",
        ),
        Candidate(
            "et", "Extra Trees",
            ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2,
                                 class_weight=cw, random_state=seed, n_jobs=1),
            "full", "Randomised splits — lower variance than Random Forest.",
        ),
        Candidate(
            "knn", "k-Nearest Neighbours",
            KNeighborsClassifier(n_neighbors=5),
            "full", "Instance-based — good when classes form tight clusters.",
        ),
        Candidate(
            "nb", "Gaussian Naive Bayes",
            GaussianNB(),
            "full", "Probabilistic baseline — very fast, strong on text.",
        ),
    ]


def _regressors(seed: int) -> list[Candidate]:
    return [
        Candidate(
            "ridge", "Ridge Regression",
            Ridge(alpha=1.0, random_state=seed),
            "fast", "Regularised linear baseline.",
        ),
        Candidate(
            "rf", "Random Forest",
            RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                  random_state=seed, n_jobs=1),
            "fast", "Bagged trees — captures non-linear structure.",
        ),
        Candidate(
            "hgb", "Gradient Boosting",
            HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1,
                                          early_stopping=True, random_state=seed),
            "fast", "Boosted trees — usually the top scorer.",
        ),
        Candidate(
            "et", "Extra Trees",
            ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2,
                                random_state=seed, n_jobs=1),
            "full", "Randomised splits — lower variance.",
        ),
        Candidate(
            "enet", "Elastic Net",
            ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=seed),
            "full", "L1+L2 penalty — performs feature selection.",
        ),
        Candidate(
            "knn", "k-Nearest Neighbours",
            KNeighborsRegressor(n_neighbors=5),
            "full", "Instance-based non-parametric regressor.",
        ),
    ]


def get_candidates(spec: TaskSpec, preset: str = "fast", seed: int = 42) -> list[Candidate]:
    """Return the models to try for ``spec``.

    ``preset`` is ``"fast"`` (3 models, seconds) or ``"full"`` (6 models).
    """
    pool = _regressors(seed) if spec.kind == REGRESSION else _classifiers(spec, seed)
    if preset == "fast":
        return [c for c in pool if c.tier == "fast"]
    return pool
