"""Decide what kind of learning problem a target column represents."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .profiling import MAX_DISCRETE_NUMERIC_LEVELS

BINARY = "binary_classification"
MULTICLASS = "multiclass_classification"
REGRESSION = "regression"


@dataclass
class TaskSpec:
    kind: str
    n_classes: int | None
    class_labels: list[str] | None
    is_imbalanced: bool
    primary_metric: str
    reason: str

    @property
    def is_classification(self) -> bool:
        return self.kind in {BINARY, MULTICLASS}


def detect_task(y: pd.Series) -> TaskSpec:
    """Infer the task type from the target column alone.

    The rules are deliberately conservative: a numeric target only becomes a
    classification problem when it has very few distinct integer levels, since
    silently turning a regression into a 12-class problem is far more
    damaging than the reverse.
    """
    y = y.dropna()
    if y.empty:
        raise ValueError("Target column is entirely missing.")

    n_unique = y.nunique()
    if n_unique < 2:
        raise ValueError(
            f"Target has only {n_unique} distinct value — nothing to learn."
        )

    numeric = pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)
    integral = numeric and y.mod(1).eq(0).all()

    if numeric and not integral:
        kind, reason = REGRESSION, "Target is continuous (non-integer values)."
    elif numeric and n_unique > MAX_DISCRETE_NUMERIC_LEVELS:
        kind = REGRESSION
        reason = f"Target is numeric with {n_unique} distinct values."
    elif n_unique == 2:
        kind, reason = BINARY, "Target has exactly 2 distinct values."
    else:
        kind = MULTICLASS
        reason = f"Target is discrete with {n_unique} classes."

    if kind == REGRESSION:
        return TaskSpec(
            kind=kind,
            n_classes=None,
            class_labels=None,
            is_imbalanced=False,
            primary_metric="r2",
            reason=reason,
        )

    counts = y.value_counts()
    # A 3:1 split already distorts accuracy enough that it stops being a
    # trustworthy headline metric, so we switch to a balanced one.
    imbalanced = float(counts.iloc[0] / counts.iloc[-1]) > 3.0

    return TaskSpec(
        kind=kind,
        n_classes=int(n_unique),
        class_labels=[str(v) for v in counts.index.tolist()],
        is_imbalanced=imbalanced,
        primary_metric="roc_auc" if kind == BINARY else "f1_macro",
        reason=reason + (" Classes are imbalanced." if imbalanced else ""),
    )
