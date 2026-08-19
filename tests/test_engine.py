"""End-to-end check: one code path, six very different datasets."""
import sys, pathlib, traceback
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from automl_core import AutoMLEngine, save, load
from tests.make_datasets import ALL

def run(name, factory):
    df, target = factory()
    eng = AutoMLEngine(preset="fast").fit(df, target=target)
    lb = eng.leaderboard
    best = next(r for r in lb if r["is_best"])
    metric = eng.task.primary_metric
    tm = best["test_metrics"]
    head = tm.get(metric, tm.get("accuracy", tm.get("r2")))

    print(f"\n{'='*74}\n{name.upper()}  ({df.shape[0]}x{df.shape[1]})")
    print(f"  task     : {eng.task.kind}  |  {eng.task.reason}")
    print(f"  features : {len(eng.feature_columns)} used, "
          f"{df.shape[1]-1-len(eng.feature_columns)} auto-dropped "
          f"({[c.name for c in eng.profile.columns if c.role in ('identifier','constant')]})")
    print(f"  winner   : {best['label']:22} cv={best['cv_mean']:.3f}  test-{metric}={head:.3f}")
    for r in lb:
        flag = "X" if r["failed"] else ("*" if r["is_best"] else " ")
        print(f"    {flag} {r['label']:24} cv={r['cv_mean']:7.3f}  {r['fit_seconds']:.2f}s"
              + (f"  ERR {r['error'][:50]}" if r["failed"] else ""))

    # Round-trip a raw row through save/load, dropping a field on purpose.
    row = df.drop(columns=[target]).iloc[0].to_dict()
    row.pop(eng.feature_columns[0], None)          # simulate a missing input
    row["totally_new_column"] = "surprise"          # simulate an extra input
    pred = eng.predict([row])
    p = pathlib.Path("/tmp/claude-1000/models") / name
    save(eng, p)
    reloaded = load(p.with_suffix(".joblib"))
    pred2 = reloaded.predict([row])
    assert str(pred[0]) == str(pred2[0]), f"save/load mismatch: {pred} vs {pred2}"
    print(f"  predict  : {pred[0]!r}  (missing+extra cols tolerated, reload matches)")
    print(f"  schema   : {len(eng.input_schema)} form fields auto-generated")
    return True

ok, fail = 0, 0
for name, factory in ALL.items():
    try:
        run(name, factory); ok += 1
    except Exception:
        fail += 1
        print(f"\n{'='*74}\n{name.upper()}  *** FAILED ***")
        traceback.print_exc()

print(f"\n{'='*74}\nRESULT: {ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
