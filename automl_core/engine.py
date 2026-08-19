"""The AutoML orchestrator.

Usage::

    engine = AutoMLEngine().fit(df, target="churned")
    engine.leaderboard          # every model, ranked
    engine.predict([{...}])     # accepts raw dict rows

Methodology note: model *selection* uses cross-validation on a training split,
and the headline numbers come from a held-out test split the search never saw.
Reporting the CV score of the winning model as if it were test performance is
the single most common way portfolio projects overstate their results.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from .models import get_candidates
from .preprocess import build_preprocessor
from .profiling import DataProfile, profile_dataframe
from .task import REGRESSION, TaskSpec, detect_task

SCHEMA_VERSION = 2


@dataclass
class ModelResult:
    key: str
    label: str
    blurb: str
    cv_mean: float
    cv_std: float
    fit_seconds: float
    test_metrics: dict[str, float] = field(default_factory=dict)
    failed: bool = False
    error: str | None = None
    # cv_mean minus the same metric on the held-out split. A large positive
    # gap means the model memorised the training folds.
    overfit_gap: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "blurb": self.blurb,
            "cv_mean": round(self.cv_mean, 4),
            "cv_std": round(self.cv_std, 4),
            "fit_seconds": round(self.fit_seconds, 2),
            "test_metrics": {k: round(v, 4) for k, v in self.test_metrics.items()},
            "failed": self.failed,
            "error": self.error,
            "overfit_gap": round(self.overfit_gap, 4),
            "overfit": self.overfit_gap > 0.15,
        }


class AutoMLEngine:
    """Fits a family of models to any tabular dataframe and keeps the best."""

    def __init__(
        self,
        preset: str = "fast",
        cv_folds: int = 5,
        test_size: float = 0.2,
        seed: int = 42,
        progress: Any = None,
    ):
        self.preset = preset
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.seed = seed
        # ``progress`` is an optional callable(stage: str, pct: float) so the
        # browser UI can show a live progress bar during a long fit.
        self.progress = progress or (lambda stage, pct: None)

        self.profile: DataProfile | None = None
        self.task: TaskSpec | None = None
        self.target: str | None = None
        self.feature_columns: list[str] = []
        self.results: list[ModelResult] = []
        self.best_: Pipeline | None = None
        self.best_key: str | None = None
        self.label_encoder_: LabelEncoder | None = None
        self._fit_X: pd.DataFrame | None = None
        self._fit_y: pd.Series | None = None

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        feature_columns: list[str] | None = None,
    ) -> "AutoMLEngine":
        if target not in df.columns:
            raise ValueError(f"Target {target!r} is not a column in the data.")

        self.progress("Profiling columns", 0.05)
        df = df.copy()
        # Rows without a label teach nothing and break scoring.
        df = df[df[target].notna()]
        if len(df) < 20:
            raise ValueError(
                f"Only {len(df)} labelled rows — need at least 20 to train."
            )

        self.profile = profile_dataframe(df)
        self.target = target
        self.task = detect_task(df[target])

        candidates = feature_columns or [
            c for c in self.profile.usable_features if c != target
        ]
        self.feature_columns = candidates
        if not self.feature_columns:
            raise ValueError("No usable feature columns were found.")

        X = df[self.feature_columns]
        y_raw = df[target]

        # Classifiers want contiguous integer labels; we keep the encoder so
        # predictions can be mapped back to the user's original strings.
        if self.task.is_classification:
            self.label_encoder_ = LabelEncoder()
            y = pd.Series(self.label_encoder_.fit_transform(y_raw), index=y_raw.index)
        else:
            y = pd.to_numeric(y_raw, errors="coerce")

        self.progress("Splitting train/test", 0.10)
        stratify = y if self._can_stratify(y) else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=stratify
        )

        pre = build_preprocessor(df, self.profile, self.feature_columns)
        cv = self._make_cv(y_tr)
        metric = self.task.primary_metric
        pool = get_candidates(self.task, self.preset, self.seed)

        self.results = []
        best_score, best_pipe, best_key = -np.inf, None, None

        for i, cand in enumerate(pool):
            self.progress(f"Training {cand.label}", 0.15 + 0.70 * i / len(pool))
            started = time.perf_counter()
            pipe = Pipeline([("pre", pre), ("model", cand.estimator)])
            try:
                scores = cross_val_score(
                    pipe, X_tr, y_tr, cv=cv, scoring=self._scoring(metric),
                    error_score="raise",
                )
                pipe.fit(X_tr, y_tr)
                elapsed = time.perf_counter() - started
                test_metrics = self._evaluate(pipe, X_te, y_te)
                comparable = test_metrics.get(self._scoring(metric))
                result = ModelResult(
                    key=cand.key, label=cand.label, blurb=cand.blurb,
                    cv_mean=float(scores.mean()), cv_std=float(scores.std()),
                    fit_seconds=elapsed,
                    test_metrics=test_metrics,
                    overfit_gap=(float(scores.mean()) - comparable
                                 if comparable is not None else 0.0),
                )
                if result.cv_mean > best_score:
                    best_score, best_pipe, best_key = result.cv_mean, pipe, cand.key
            except Exception as exc:  # one bad model must not sink the run
                result = ModelResult(
                    key=cand.key, label=cand.label, blurb=cand.blurb,
                    cv_mean=float("-inf"), cv_std=0.0,
                    fit_seconds=time.perf_counter() - started,
                    failed=True, error=f"{type(exc).__name__}: {exc}",
                )
            self.results.append(result)

        if best_pipe is None:
            errors = "; ".join(r.error or "" for r in self.results if r.failed)
            raise RuntimeError(f"Every candidate model failed. {errors}")

        self.progress("Refitting winner on all data", 0.92)
        # The leaderboard's test metrics are honest because they came from a
        # model that never saw the test split. The *shipped* model is refit on
        # everything, which is standard practice and strictly better at predicting.
        self.best_key = best_key
        self.best_ = best_pipe
        self.best_.fit(X, y)
        # Kept for permutation importance, which needs real data to shuffle.
        self._fit_X, self._fit_y = X, y

        self.progress("Done", 1.0)
        return self

    # -------------------------------------------------------------- helpers

    def _can_stratify(self, y: pd.Series) -> bool:
        if not self.task.is_classification:
            return False
        # Stratifying needs at least one sample per class on each side.
        return int(y.value_counts().min()) >= 2

    def _make_cv(self, y: pd.Series):
        if self.task.is_classification:
            # n_splits can never exceed the rarest class's member count.
            folds = max(2, min(self.cv_folds, int(y.value_counts().min())))
            return StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.seed)
        folds = max(2, min(self.cv_folds, len(y) // 5))
        return KFold(n_splits=folds, shuffle=True, random_state=self.seed)

    def _scoring(self, metric: str) -> str:
        # roc_auc is undefined for >2 classes, so fall back where needed.
        if metric == "roc_auc" and self.task.kind != "binary_classification":
            return "f1_macro"
        return metric

    def _evaluate(self, pipe: Pipeline, X, y) -> dict[str, float]:
        pred = pipe.predict(X)
        if self.task.kind == REGRESSION:
            rmse = float(np.sqrt(mean_squared_error(y, pred)))
            return {
                "r2": float(r2_score(y, pred)),
                "rmse": rmse,
                "mae": float(mean_absolute_error(y, pred)),
            }
        out = {
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
            "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        }
        if self.task.kind == "binary_classification" and hasattr(pipe, "predict_proba"):
            try:
                out["roc_auc"] = float(roc_auc_score(y, pipe.predict_proba(X)[:, 1]))
            except Exception:
                pass
        return out

    # ------------------------------------------------------------- inspect

    @property
    def warnings(self) -> list[str]:
        """Honest caveats about this run, for display next to the scores."""
        out: list[str] = []
        if self.profile is None:
            return out
        n = self.profile.n_rows
        if n < 100:
            out.append(
                f"Only {n} rows — scores on datasets this small are unstable "
                f"and can swing wildly with a different random split."
            )
        n_feat = len(self.feature_columns)
        if n_feat > n / 10:
            out.append(
                f"{n_feat} features for {n} rows. With so few rows per feature, "
                f"models can memorise noise; consider collecting more data."
            )
        best = next((r for r in self.results if r.key == self.best_key), None)
        if best is not None and best.overfit_gap > 0.15:
            out.append(
                f"The winner scores {best.overfit_gap:.2f} higher in "
                f"cross-validation than on held-out data — it is overfitting. "
                f"Trust the test score, not the CV score."
            )
        if self.task is not None and self.task.is_imbalanced:
            out.append(
                "Classes are imbalanced, so plain accuracy is misleading. "
                f"The ranking metric is {self.task.primary_metric}."
            )
        if self.profile.duplicate_rows > 0:
            out.append(
                f"{self.profile.duplicate_rows} duplicate rows — these can leak "
                f"across the train/test split and inflate scores."
            )
        return out

    @property
    def leaderboard(self) -> list[dict[str, Any]]:
        ranked = sorted(self.results, key=lambda r: (not r.failed, r.cv_mean), reverse=True)
        return [{**r.to_dict(), "rank": i + 1, "is_best": r.key == self.best_key}
                for i, r in enumerate(ranked)]

    @property
    def input_schema(self) -> list[dict[str, Any]]:
        """Field descriptors the frontend uses to build an input form.

        This is what makes the UI dynamic: the form is generated from whatever
        the user uploaded, never hardcoded.
        """
        if self.profile is None:
            raise RuntimeError("Call fit() first.")
        schema = []
        for name in self.feature_columns:
            col = self.profile.get(name)
            field_type = {"numeric": "number", "categorical": "select",
                          "datetime": "date", "text": "textarea"}.get(col.role, "text")
            entry: dict[str, Any] = {
                "name": name,
                "type": field_type,
                "role": col.role,
                "required": col.pct_missing == 0,
            }
            if col.role == "categorical":
                entry["options"] = col.stats.get("levels", [])
                entry["default"] = col.stats.get("top_value")
            elif col.role == "numeric":
                entry.update({
                    "min": col.stats.get("min"),
                    "max": col.stats.get("max"),
                    "default": col.stats.get("median"),
                })
            schema.append(entry)
        return schema

    def feature_importance(self, top_n: int = 15, n_repeats: int = 5) -> list[dict[str, Any]]:
        """Permutation importance on the fitted winner.

        Permutation is model-agnostic, so the same call works whether the
        winner was a linear model or a forest.
        """
        if self.best_ is None:
            raise RuntimeError("Call fit() first.")
        r = permutation_importance(
            self.best_, self._fit_X, self._fit_y, n_repeats=n_repeats,
            random_state=self.seed, scoring=self._scoring(self.task.primary_metric),
        )
        pairs = sorted(
            zip(self.feature_columns, r.importances_mean, r.importances_std),
            key=lambda t: t[1], reverse=True,
        )
        return [{"feature": f, "importance": round(float(m), 5),
                 "std": round(float(s), 5)} for f, m, s in pairs[:top_n]]

    # ------------------------------------------------------------- predict

    def predict(self, rows: list[dict] | pd.DataFrame) -> list[Any]:
        """Predict on raw rows, tolerating missing and extra columns."""
        if self.best_ is None:
            raise RuntimeError("Call fit() first.")
        X = self._coerce_rows(rows)
        pred = self.best_.predict(X)
        if self.label_encoder_ is not None:
            pred = self.label_encoder_.inverse_transform(pred.astype(int))
        return [v.item() if hasattr(v, "item") else v for v in pred]

    def predict_proba(self, rows: list[dict] | pd.DataFrame) -> list[dict[str, float]]:
        if self.best_ is None:
            raise RuntimeError("Call fit() first.")
        if not self.task.is_classification:
            raise ValueError("Probabilities are only defined for classification.")
        if not hasattr(self.best_, "predict_proba"):
            raise ValueError(f"{self.best_key} does not expose probabilities.")
        proba = self.best_.predict_proba(self._coerce_rows(rows))
        labels = [str(c) for c in self.label_encoder_.classes_]
        return [dict(zip(labels, (round(float(p), 4) for p in row))) for row in proba]

    def _coerce_rows(self, rows) -> pd.DataFrame:
        df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
        # Re-add any column the caller omitted as NaN; the fitted imputer in
        # the pipeline will fill it. Extra columns are simply dropped.
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        return df[self.feature_columns]
