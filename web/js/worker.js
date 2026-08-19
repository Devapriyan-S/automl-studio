/*
 * Pyodide worker.
 *
 * Training a model takes seconds of solid CPU. Running it on the main thread
 * would freeze the page — no progress bar, no scrolling, and the browser
 * eventually offering to kill the tab. So the whole Python runtime lives here
 * in a worker and talks to the UI over postMessage.
 */

const PYODIDE_VERSION = "0.28.0";
const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

importScripts(`${PYODIDE_CDN}pyodide.js`);

let pyodide = null;
let bridge = null;

const post = (type, payload = {}) => self.postMessage({ type, ...payload });

/* Called from Python during a fit to drive the progress bar. */
self.reportProgress = (stage, pct) => post("progress", { stage, pct });

async function boot() {
  post("boot", { stage: "Downloading Python runtime", pct: 0.05 });
  pyodide = await loadPyodide({
    indexURL: PYODIDE_CDN,
    stdout: (line) => post("log", { line }),
    stderr: (line) => post("log", { line, isError: true }),
  });

  post("boot", { stage: "Loading numpy, pandas, scikit-learn", pct: 0.35 });
  // scipy and joblib arrive as scikit-learn dependencies, but naming them
  // explicitly means a resolver change upstream cannot silently drop them.
  await pyodide.loadPackage(["numpy", "pandas", "scipy", "scikit-learn", "joblib"]);

  post("boot", { stage: "Installing automl_core", pct: 0.8 });
  const manifest = await (await fetch("../py/manifest.json")).json();

  pyodide.FS.mkdirTree("/lib/automl_core");
  await Promise.all(
    manifest.files.map(async (name) => {
      const src = await (await fetch(`../py/automl_core/${name}`)).text();
      pyodide.FS.writeFile(`/lib/automl_core/${name}`, src);
    })
  );
  const bridgeSrc = await (await fetch("../py/bridge.py")).text();
  pyodide.FS.writeFile("/lib/bridge.py", bridgeSrc);

  pyodide.runPython(`import sys; sys.path.insert(0, "/lib")`);
  bridge = pyodide.pyimport("bridge");

  const versions = pyodide.runPython(`
import sklearn, pandas, numpy, sys
f"Python {sys.version.split()[0]} · scikit-learn {sklearn.__version__} · pandas {pandas.__version__} · numpy {numpy.__version__}"
`);
  post("ready", { versions });
}

const HANDLERS = {
  loadCsv: ({ text }) => bridge.load_csv(text),
  train: ({ target, preset }) => bridge.train(target, preset),
  predict: ({ rows }) => bridge.predict(JSON.stringify(rows)),
};

self.onmessage = async ({ data }) => {
  const { id, action, ...args } = data;
  try {
    if (!bridge) throw new Error("Python runtime is still starting up.");
    const handler = HANDLERS[action];
    if (!handler) throw new Error(`Unknown action: ${action}`);
    // Every bridge function returns a JSON string, so the structured-clone
    // boundary only ever carries plain data — never a Python proxy object.
    post("result", { id, result: JSON.parse(await handler(args)) });
  } catch (err) {
    post("result", { id, result: { ok: false, error: String(err.message || err) } });
  }
};

boot().catch((err) =>
  post("bootError", { error: String(err.message || err), stack: String(err.stack || "") })
);
