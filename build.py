#!/usr/bin/env python3
"""Copy automl_core into web/py/ so the browser build stays in sync.

The Pyodide worker fetches these files at runtime and writes them into its
virtual filesystem. Running this script is the only step needed to publish
engine changes to the web demo — there is no second copy to maintain by hand.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "automl_core"
DEST = ROOT / "web" / "py" / "automl_core"


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    names = []
    for f in sorted(SRC.glob("*.py")):
        shutil.copy2(f, DEST / f.name)
        names.append(f.name)

    # The worker needs the file list to know what to fetch; generating it here
    # means adding a module never requires touching JavaScript.
    manifest = {"package": "automl_core", "files": names}
    (DEST.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum((DEST / n).stat().st_size for n in names)
    print(f"Copied {len(names)} modules ({total/1024:.1f} KB) -> {DEST}")
    for n in names:
        print(f"  {n}")


if __name__ == "__main__":
    main()
