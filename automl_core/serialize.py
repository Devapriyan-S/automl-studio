"""Persist a fitted engine as a single portable artefact.

The pickle holds the *whole* pipeline — preprocessing plus estimator — so a
loaded model accepts the same raw rows the original did. Alongside it we store
a JSON sidecar of metadata (schema, leaderboard, task) that a frontend can read
without unpickling anything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from .engine import SCHEMA_VERSION, AutoMLEngine


def bundle_metadata(engine: AutoMLEngine) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "target": engine.target,
        "task": {
            "kind": engine.task.kind,
            "primary_metric": engine.task.primary_metric,
            "n_classes": engine.task.n_classes,
            "class_labels": engine.task.class_labels,
            "reason": engine.task.reason,
        },
        "best_model": engine.best_key,
        "feature_columns": engine.feature_columns,
        "input_schema": engine.input_schema,
        "leaderboard": engine.leaderboard,
        "data_profile": engine.profile.to_dict(),
    }


def save(engine: AutoMLEngine, path: str | Path) -> Path:
    """Write ``<path>.joblib`` and ``<path>.meta.json``."""
    path = Path(path).with_suffix("")
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "schema_version": SCHEMA_VERSION,
            "pipeline": engine.best_,
            "label_encoder": engine.label_encoder_,
            "feature_columns": engine.feature_columns,
            "task_kind": engine.task.kind,
        },
        path.with_suffix(".joblib"),
        compress=3,
    )
    path.with_suffix(".meta.json").write_text(
        json.dumps(bundle_metadata(engine), indent=2, default=str)
    )
    return path.with_suffix(".joblib")


class LoadedModel:
    """A saved pipeline restored for inference only."""

    def __init__(self, blob: dict[str, Any], meta: dict[str, Any] | None = None):
        if blob.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Model was saved with schema v{blob.get('schema_version')}, "
                f"this build expects v{SCHEMA_VERSION}."
            )
        self.pipeline = blob["pipeline"]
        self.label_encoder = blob["label_encoder"]
        self.feature_columns = blob["feature_columns"]
        self.task_kind = blob["task_kind"]
        self.meta = meta or {}

    def predict(self, rows: list[dict]) -> list[Any]:
        import numpy as np
        import pandas as pd

        df = pd.DataFrame(rows)
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = np.nan
        pred = self.pipeline.predict(df[self.feature_columns])
        if self.label_encoder is not None:
            pred = self.label_encoder.inverse_transform(pred.astype(int))
        return [v.item() if hasattr(v, "item") else v for v in pred]


def load(path: str | Path) -> LoadedModel:
    path = Path(path)
    blob = joblib.load(path)
    meta_path = path.with_suffix("").with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return LoadedModel(blob, meta)
