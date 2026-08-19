"""JSON bridge between the browser UI and automl_core.

Every function here takes and returns plain JSON strings. Keeping the boundary
that narrow means the JavaScript side never touches a Python object, and the
same engine code runs unmodified on a server.
"""

import io
import json
import traceback

import pandas as pd

from automl_core import AutoMLEngine, profile_dataframe

_state = {"df": None, "engine": None}


def _json(payload) -> str:
    return json.dumps(payload, default=str)


def _fail(exc: Exception) -> str:
    return _json({
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    })


def load_csv(text: str) -> str:
    """Parse raw CSV text and return a full data profile."""
    try:
        df = pd.read_csv(io.StringIO(text))
        df.columns = [str(c).strip() for c in df.columns]
        if df.empty:
            raise ValueError("The file parsed to zero rows.")
        _state["df"] = df
        _state["engine"] = None

        profile = profile_dataframe(df)
        head = df.head(12)
        return _json({
            "ok": True,
            "profile": profile.to_dict(),
            # Any column can be a target; we suggest the ones that make sense.
            "target_candidates": [
                c.name for c in profile.columns
                if c.role in {"numeric", "categorical"} and c.n_unique > 1
            ],
            "preview": {
                "columns": list(df.columns),
                "rows": head.astype(object).where(head.notna(), None).values.tolist(),
            },
        })
    except Exception as exc:
        return _fail(exc)


def train(target: str, preset: str = "fast") -> str:
    """Fit the model family against ``target``."""
    try:
        df = _state["df"]
        if df is None:
            raise RuntimeError("Upload a CSV first.")

        def report(stage, pct):
            # reportProgress is installed on the worker's global scope by main.js.
            from js import reportProgress  # type: ignore
            reportProgress(stage, float(pct))

        engine = AutoMLEngine(preset=preset, progress=report).fit(df, target=target)
        _state["engine"] = engine

        return _json({
            "ok": True,
            "task": {
                "kind": engine.task.kind,
                "reason": engine.task.reason,
                "primary_metric": engine.task.primary_metric,
                "n_classes": engine.task.n_classes,
                "class_labels": engine.task.class_labels,
            },
            "leaderboard": engine.leaderboard,
            "warnings": engine.warnings,
            "input_schema": engine.input_schema,
            "feature_columns": engine.feature_columns,
            "dropped_columns": [
                {"name": c.name, "reason": c.role}
                for c in engine.profile.columns
                if c.role in {"identifier", "constant"} and c.name != target
            ],
            "importance": engine.feature_importance(top_n=12, n_repeats=3),
        })
    except Exception as exc:
        return _fail(exc)


def predict(rows_json: str) -> str:
    """Predict on user-entered rows, with probabilities where available."""
    try:
        engine = _state["engine"]
        if engine is None:
            raise RuntimeError("Train a model first.")

        rows = json.loads(rows_json)
        # HTML inputs arrive as strings; numeric fields must be coerced back.
        numeric = {f["name"] for f in engine.input_schema if f["type"] == "number"}
        for row in rows:
            for key, value in list(row.items()):
                if value in ("", None):
                    row[key] = None
                elif key in numeric:
                    try:
                        row[key] = float(value)
                    except (TypeError, ValueError):
                        row[key] = None

        out = {"ok": True, "predictions": engine.predict(rows)}
        if engine.task.is_classification:
            try:
                out["probabilities"] = engine.predict_proba(rows)
            except Exception:
                out["probabilities"] = None
        return _json(out)
    except Exception as exc:
        return _fail(exc)
