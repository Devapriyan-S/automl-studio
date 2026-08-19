"""Build a preprocessing ColumnTransformer straight from a DataProfile.

Keeping preprocessing inside a scikit-learn Pipeline (rather than mutating the
dataframe up front) is what lets a saved model accept raw user rows later:
unseen categories, missing fields and column reordering are all handled by the
fitted transformer instead of by the caller.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .profiling import DataProfile

# Above this many levels, one-hot encoding explodes the feature space, so we
# fall back to ordinal codes and let tree models split on them.
ONEHOT_MAX_LEVELS = 20


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Expand datetime columns into numeric calendar parts.

    Cyclical fields (month, weekday, hour) are emitted as sin/cos pairs so that
    December and January sit next to each other instead of 11 units apart.
    """

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        out = pd.DataFrame(index=X.index)
        for col in self.columns_:
            dt = pd.to_datetime(X[col], errors="coerce", format="mixed")
            out[f"{col}__year"] = dt.dt.year
            out[f"{col}__month_sin"] = np.sin(2 * np.pi * dt.dt.month / 12)
            out[f"{col}__month_cos"] = np.cos(2 * np.pi * dt.dt.month / 12)
            out[f"{col}__day"] = dt.dt.day
            out[f"{col}__dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
            out[f"{col}__dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
            out[f"{col}__hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
            out[f"{col}__hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
            out[f"{col}__is_weekend"] = (dt.dt.dayofweek >= 5).astype(float)
        return out.fillna(0.0).to_numpy()

    def get_feature_names_out(self, input_features=None):
        suffixes = ["year", "month_sin", "month_cos", "day", "dow_sin",
                    "dow_cos", "hour_sin", "hour_cos", "is_weekend"]
        return np.array([f"{c}__{s}" for c in self.columns_ for s in suffixes])


class AsCategoricalStrings(BaseEstimator, TransformerMixin):
    """Normalise any categorical column to nullable strings.

    scikit-learn's imputers reject boolean dtype outright, and a column mixing
    ints with strings encodes inconsistently. Casting every non-null value to
    ``str`` up front makes the downstream encoders behave identically whether
    the user uploaded ``True``, ``"True"``, or ``1``.
    """

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for col in df.columns:
            values = df[col].astype(object)
            present = values.notna()
            values[present] = values[present].astype(str)
            df[col] = values
        return df

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features if input_features is not None else self.columns_)


class TextCleaner(BaseEstimator, TransformerMixin):
    """Coerce a text column to a NaN-free 1-D string array.

    TfidfVectorizer raises on NaN and on non-string input, so this guard is
    what keeps a text column with a few blank cells from failing the run.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(X)
        return s.fillna("").astype(str).to_numpy()


def _numeric_branch() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])


def _low_card_branch() -> Pipeline:
    return Pipeline([
        ("cast", AsCategoricalStrings()),
        ("impute", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore" is the reason a saved model survives contact
        # with categories it never saw during training.
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 min_frequency=0.01)),
    ])


def _high_card_branch() -> Pipeline:
    return Pipeline([
        ("cast", AsCategoricalStrings()),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                  unknown_value=-1)),
    ])


def _text_branch(n_components: int = 50) -> Pipeline:
    return Pipeline([
        ("clean", TextCleaner()),
        ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                                  strip_accents="unicode", min_df=2)),
        ("svd", TruncatedSVD(n_components=n_components, random_state=0)),
    ])


def build_preprocessor(
    df: pd.DataFrame,
    profile: DataProfile,
    feature_columns: list[str],
) -> ColumnTransformer:
    """Assemble the transformer for the chosen feature columns."""
    numeric, low_card, high_card, datetimes, texts = [], [], [], [], []

    for name in feature_columns:
        col = profile.get(name)
        if col is None:
            continue
        if col.role == "numeric":
            numeric.append(name)
        elif col.role == "datetime":
            datetimes.append(name)
        elif col.role == "text":
            texts.append(name)
        elif col.role == "categorical":
            (low_card if col.n_unique <= ONEHOT_MAX_LEVELS else high_card).append(name)

    transformers: list[tuple] = []
    if numeric:
        transformers.append(("num", _numeric_branch(), numeric))
    if low_card:
        transformers.append(("cat", _low_card_branch(), low_card))
    if high_card:
        transformers.append(("cat_hi", _high_card_branch(), high_card))
    if datetimes:
        transformers.append(("dt", DateTimeFeatures(), datetimes))
    for i, tcol in enumerate(texts):
        # Text branches take a 1-D Series, hence the bare string selector.
        n_comp = min(50, max(2, len(df) // 10))
        transformers.append((f"txt{i}", _text_branch(n_comp), tcol))

    if not transformers:
        raise ValueError(
            "No usable feature columns. All candidates were IDs, constants, "
            "or entirely missing."
        )

    return ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
