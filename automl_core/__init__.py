"""automl_core — train a model on *any* tabular CSV, with no code changes.

    from automl_core import AutoMLEngine
    engine = AutoMLEngine().fit(df, target="price")
    engine.leaderboard
    engine.predict([{"rooms": 3, "city": "Chennai"}])

The library is deliberately dependency-light (numpy, pandas, scikit-learn,
scipy) so the exact same code runs on a server and inside the browser via
Pyodide/WebAssembly.
"""

from .engine import AutoMLEngine, ModelResult
from .profiling import ColumnProfile, DataProfile, profile_dataframe
from .serialize import LoadedModel, load, save
from .task import TaskSpec, detect_task

__version__ = "1.0.0"
__all__ = [
    "AutoMLEngine",
    "ModelResult",
    "DataProfile",
    "ColumnProfile",
    "profile_dataframe",
    "TaskSpec",
    "detect_task",
    "save",
    "load",
    "LoadedModel",
]
