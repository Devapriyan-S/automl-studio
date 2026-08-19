"""Column-type inference and data-quality profiling.

This is the module that makes the rest of the library *dynamic*: nothing
downstream needs to know what the columns are called or what they contain,
because :func:`profile_dataframe` works it out from the data itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd

# A column with more than this fraction of unique values is treated as a
# free-text / identifier column rather than a low-cardinality category.
HIGH_CARDINALITY_RATIO = 0.5

# Numeric columns with at most this many distinct values are usually encoded
# categories (e.g. a 0/1/2 "class" column), not true measurements.
MAX_DISCRETE_NUMERIC_LEVELS = 12

# Object columns whose average token count exceeds this look like sentences.
TEXT_TOKEN_THRESHOLD = 3.0

# An all-unique integer column is treated as an identifier only when its value
# range is at most this multiple of its row count — i.e. it is a dense
# sequence. Prices, timestamps and measurements have a far sparser range.
IDENTIFIER_DENSITY = 1.5


@dataclass
class ColumnProfile:
    """Everything the pipeline needs to know about a single column."""

    name: str
    role: str  # numeric | categorical | datetime | text | identifier | constant
    dtype: str
    n_missing: int
    pct_missing: float
    n_unique: int
    sample_values: list[Any] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataProfile:
    """Profile of an entire dataframe."""

    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    duplicate_rows: int
    memory_mb: float

    def by_role(self, role: str) -> list[str]:
        return [c.name for c in self.columns if c.role == role]

    def get(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)

    @property
    def usable_features(self) -> list[str]:
        """Columns worth feeding to a model.

        Identifiers and constants are excluded: an identifier leaks row
        identity and a constant carries no signal, so both only add noise.
        """
        return [c.name for c in self.columns
                if c.role in {"numeric", "categorical", "datetime", "text"}]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "duplicate_rows": self.duplicate_rows,
            "memory_mb": self.memory_mb,
            "columns": [c.to_dict() for c in self.columns],
        }


def _looks_like_datetime(s: pd.Series) -> bool:
    """Try parsing a sample as dates, and demand most of them succeed."""
    sample = s.dropna().astype(str).head(200)
    if sample.empty:
        return False
    # Bare integers parse as dates ("2019" -> 2019-01-01) but almost never
    # *are* dates, so require at least one separator to be present.
    if not sample.str.contains(r"[-/:]").any():
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return parsed.notna().mean() > 0.8


def _mean_token_count(s: pd.Series) -> float:
    sample = s.dropna().astype(str).head(500)
    if sample.empty:
        return 0.0
    return float(sample.str.split().str.len().mean())


def infer_role(s: pd.Series, n_rows: int) -> str:
    """Classify one column into a modelling role."""
    non_null = s.dropna()

    if non_null.empty:
        return "constant"

    n_unique = non_null.nunique()
    if n_unique <= 1:
        return "constant"

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    if pd.api.types.is_bool_dtype(s):
        return "categorical"

    if pd.api.types.is_numeric_dtype(s):
        # Integer-valued columns with very few levels are encoded categories.
        is_integral = pd.api.types.is_integer_dtype(s) or (
            non_null.mod(1).eq(0).all()
        )
        if is_integral and n_unique <= MAX_DISCRETE_NUMERIC_LEVELS:
            return "categorical"
        # An all-unique integer column *might* be a row index — but it might
        # equally be a rounded price or a timestamp, which are real features.
        # The distinguishing trait of an ID is that it densely fills a
        # contiguous range (0..N-1, 1000..1000+N), whereas a measurement is
        # spread thinly across a range far wider than the row count.
        if is_integral and n_unique == len(non_null) and n_unique > 20:
            span = float(non_null.max() - non_null.min()) + 1
            if span <= n_unique * IDENTIFIER_DENSITY:
                return "identifier"
        return "numeric"

    # Everything below here is object / string-like.
    if _looks_like_datetime(s):
        return "datetime"

    unique_ratio = n_unique / max(len(non_null), 1)

    if _mean_token_count(non_null) >= TEXT_TOKEN_THRESHOLD:
        return "text"

    if unique_ratio > HIGH_CARDINALITY_RATIO and n_unique > 50:
        return "identifier"

    return "categorical"


def _numeric_stats(s: pd.Series) -> dict[str, Any]:
    non_null = s.dropna()
    if non_null.empty:
        return {}
    q1, q3 = non_null.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {
        "mean": float(non_null.mean()),
        "std": float(non_null.std()) if len(non_null) > 1 else 0.0,
        "min": float(non_null.min()),
        "q1": float(q1),
        "median": float(non_null.median()),
        "q3": float(q3),
        "max": float(non_null.max()),
        "skew": float(non_null.skew()) if len(non_null) > 2 else 0.0,
        "n_outliers": int(((non_null < lo) | (non_null > hi)).sum()),
    }


def _categorical_stats(s: pd.Series) -> dict[str, Any]:
    counts = s.dropna().value_counts()
    if counts.empty:
        return {}
    return {
        "top_value": str(counts.index[0]),
        "top_freq": int(counts.iloc[0]),
        "imbalance_ratio": float(counts.iloc[0] / counts.iloc[-1]),
        "levels": [str(v) for v in counts.head(20).index.tolist()],
    }


def profile_column(s: pd.Series, n_rows: int) -> ColumnProfile:
    role = infer_role(s, n_rows)
    n_missing = int(s.isna().sum())

    stats: dict[str, Any] = {}
    if role == "numeric":
        stats = _numeric_stats(s)
    elif role == "categorical":
        stats = _categorical_stats(s)

    warnings: list[str] = []
    pct_missing = 100.0 * n_missing / n_rows if n_rows else 0.0
    if pct_missing > 50:
        warnings.append(f"{pct_missing:.0f}% missing — consider dropping")
    elif pct_missing > 10:
        warnings.append(f"{pct_missing:.0f}% missing — will be imputed")
    if role == "identifier":
        warnings.append("Looks like an ID — excluded from features")
    if role == "constant":
        warnings.append("Single value — no predictive signal")
    if stats.get("imbalance_ratio", 0) > 20:
        warnings.append("Heavily imbalanced categories")
    if abs(stats.get("skew", 0.0)) > 2:
        warnings.append("Strongly skewed — scaling applied")

    return ColumnProfile(
        name=str(s.name),
        role=role,
        dtype=str(s.dtype),
        n_missing=n_missing,
        pct_missing=round(pct_missing, 2),
        n_unique=int(s.nunique(dropna=True)),
        sample_values=[
            None if pd.isna(v) else (v.item() if hasattr(v, "item") else str(v))
            for v in s.dropna().head(5).tolist()
        ],
        stats=stats,
        warnings=warnings,
    )


def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    """Build a full :class:`DataProfile` for ``df``."""
    n_rows = len(df)
    return DataProfile(
        n_rows=n_rows,
        n_cols=df.shape[1],
        columns=[profile_column(df[c], n_rows) for c in df.columns],
        duplicate_rows=int(df.duplicated().sum()),
        memory_mb=round(df.memory_usage(deep=True).sum() / 1024**2, 3),
    )
