"""FastAPI service around automl_core.

The browser demo proves the engine needs no server; this proves the same
package drops into one unchanged. Models are held in a process-local registry,
which is fine for a demo and is the first thing you would swap for Redis or S3
before putting it in front of real traffic.

    uvicorn api.main:app --reload
    open http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import io
import uuid
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from automl_core import AutoMLEngine, profile_dataframe

app = FastAPI(
    title="AutoML Studio API",
    version="1.0.0",
    description="Train a scikit-learn model on any uploaded CSV, then serve predictions from it.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo service; scope this to your own origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# model_id -> fitted engine. Process-local and therefore lost on restart.
REGISTRY: dict[str, AutoMLEngine] = {}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class TrainRequest(BaseModel):
    target: str = Field(..., description="Column to predict.")
    preset: str = Field("fast", pattern="^(fast|full)$")


class PredictRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1, max_length=1000)


async def _read_csv(file: UploadFile) -> pd.DataFrame:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // 1024**2} MB.")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}") from exc
    if df.empty:
        raise HTTPException(400, "CSV parsed to zero rows.")
    df.columns = [str(c).strip() for c in df.columns]
    return df


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "models_loaded": len(REGISTRY)}


@app.post("/profile", summary="Infer column roles without training")
async def profile(file: UploadFile = File(...)) -> dict[str, Any]:
    df = await _read_csv(file)
    return profile_dataframe(df).to_dict()


@app.post("/train", summary="Fit a model family and keep the winner")
async def train(target: str, preset: str = "fast", file: UploadFile = File(...)) -> dict[str, Any]:
    df = await _read_csv(file)
    try:
        engine = AutoMLEngine(preset=preset).fit(df, target=target)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    model_id = uuid.uuid4().hex[:12]
    REGISTRY[model_id] = engine
    return {
        "model_id": model_id,
        "task": engine.task.kind,
        "best_model": engine.best_key,
        "leaderboard": engine.leaderboard,
        "warnings": engine.warnings,
        "input_schema": engine.input_schema,
    }


@app.get("/models/{model_id}/schema", summary="Fields a client should collect")
def schema(model_id: str) -> dict[str, Any]:
    engine = REGISTRY.get(model_id)
    if engine is None:
        raise HTTPException(404, f"No model {model_id!r}. It may have expired on restart.")
    return {
        "model_id": model_id,
        "target": engine.target,
        "task": engine.task.kind,
        "input_schema": engine.input_schema,
    }


@app.post("/models/{model_id}/predict", summary="Predict on raw rows")
def predict(model_id: str, body: PredictRequest) -> dict[str, Any]:
    engine = REGISTRY.get(model_id)
    if engine is None:
        raise HTTPException(404, f"No model {model_id!r}. It may have expired on restart.")

    out: dict[str, Any] = {"predictions": engine.predict(body.rows)}
    if engine.task.is_classification:
        try:
            out["probabilities"] = engine.predict_proba(body.rows)
        except ValueError:
            out["probabilities"] = None
    return out
