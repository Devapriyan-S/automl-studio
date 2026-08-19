/*
 * AutoML Studio — UI controller.
 *
 * Holds no ML logic of its own: it renders whatever shape the Python engine
 * reports back. That is the whole point — the form fields, the leaderboard
 * columns and the metric names all come from the uploaded data, so the same
 * page handles a churn CSV and a house-price CSV without a code change.
 */

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html !== undefined) n.innerHTML = html;
  return n;
};
/* User data goes through this before ever touching innerHTML. */
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

/* Scientific notation is right for a 1e-9 importance score and wrong for a
   house price, so the threshold sits well above everyday currency values. */
const fmt = (v, digits = 3) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const n = Number(v);
  const abs = Math.abs(n);
  if (abs !== 0 && (abs >= 1e9 || abs < 1e-4)) return n.toExponential(2);
  if (abs >= 1000) return n.toLocaleString("en-US", { maximumFractionDigits: 0 });
  return n.toFixed(digits).replace(/\.?0+$/, (m) => (m.includes(".") ? "" : m));
};

const METRIC_LABEL = {
  roc_auc: "ROC AUC", f1_macro: "F1 (macro)", r2: "R²", rmse: "RMSE",
  mae: "MAE", accuracy: "Accuracy", balanced_accuracy: "Balanced acc.",
  precision_macro: "Precision", recall_macro: "Recall",
};
const TASK_LABEL = {
  binary_classification: "Binary classification",
  multiclass_classification: "Multi-class classification",
  regression: "Regression",
};

/* ── Worker plumbing ──────────────────────────────────────── */

const worker = new Worker("js/worker.js");
let nextId = 0;
const pending = new Map();

const call = (action, args = {}) =>
  new Promise((resolve) => {
    const id = ++nextId;
    pending.set(id, resolve);
    worker.postMessage({ id, action, ...args });
  });

worker.onmessage = ({ data }) => {
  switch (data.type) {
    case "boot":
      $("#boot-stage").textContent = data.stage;
      $("#boot-bar").style.width = `${data.pct * 100}%`;
      break;
    case "ready":
      $("#boot-bar").style.width = "100%";
      $("#runtime-badge").textContent = data.versions;
      setTimeout(() => { $("#boot").hidden = true; $("#app").hidden = false; }, 320);
      break;
    case "bootError":
      $("#boot-stage").textContent = "Could not start Python";
      $("#boot-error").hidden = false;
      $("#boot-error").textContent =
        `${data.error}\n\nIf this page was just redeployed, the CDN may still ` +
        `be propagating — wait a moment and reload. Otherwise check that ` +
        `cdn.jsdelivr.net is reachable from your network.`;
      break;
    case "progress":
      $("#train-bar").style.width = `${data.pct * 100}%`;
      $("#train-stage").textContent = data.stage;
      break;
    case "result": {
      const resolve = pending.get(data.id);
      if (resolve) { pending.delete(data.id); resolve(data.result); }
      break;
    }
    case "log":
      (data.isError ? console.warn : console.log)("[python]", data.line);
      break;
  }
};

/* ── State ────────────────────────────────────────────────── */

const state = { profile: null, training: null };

const showError = (sel, message) => {
  const node = $(sel);
  node.hidden = false;
  node.textContent = message;
};
const clearError = (sel) => { $(sel).hidden = true; };

const reveal = (sel) => {
  const node = $(sel);
  if (node.hidden) node.hidden = false;
};

/* ── Step 1: upload ───────────────────────────────────────── */

const dropzone = $("#dropzone");
const fileInput = $("#file-input");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
});
["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("over"); }));
["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("over"); }));

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) readFile(file);
});
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) readFile(file);
});

document.querySelectorAll("[data-sample]").forEach((btn) =>
  btn.addEventListener("click", () => {
    const sample = SAMPLES[btn.dataset.sample];
    ingest(sample.csv, sample.suggestedTarget);
  }));

function readFile(file) {
  if (!/\.csv$/i.test(file.name) && file.type !== "text/csv") {
    showError("#upload-error", `"${file.name}" is not a CSV. Export your sheet as CSV and retry.`);
    return;
  }
  const reader = new FileReader();
  reader.onerror = () => showError("#upload-error", "Could not read that file.");
  reader.onload = () => ingest(reader.result);
  reader.readAsText(file);
}

async function ingest(csvText, suggestedTarget) {
  clearError("#upload-error");
  dropzone.querySelector(".dz-title").textContent = "Parsing…";

  const res = await call("loadCsv", { text: csvText });
  dropzone.querySelector(".dz-title").textContent = "Drop a CSV here, or click to browse";

  if (!res.ok) { showError("#upload-error", res.error); return; }

  state.profile = res.profile;
  renderProfile(res);
  renderTargets(res, suggestedTarget);

  reveal("#step-profile");
  reveal("#step-configure");
  $("#step-results").hidden = true;
  $("#step-predict").hidden = true;
  $("#step-profile").scrollIntoView({ behavior: "smooth", block: "start" });
}

/* ── Step 2: profile ──────────────────────────────────────── */

function renderProfile({ profile, preview }) {
  const usable = profile.columns.filter((c) =>
    ["numeric", "categorical", "datetime", "text"].includes(c.role)).length;
  const missing = profile.columns.reduce((a, c) => a + c.n_missing, 0);

  const stats = [
    [profile.n_rows.toLocaleString(), "Rows"],
    [profile.n_cols, "Columns"],
    [usable, "Usable features"],
    [missing.toLocaleString(), "Missing cells"],
    [profile.duplicate_rows.toLocaleString(), "Duplicate rows"],
  ];
  $("#dataset-stats").replaceChildren(...stats.map(([v, l]) => {
    const s = el("div", "stat");
    s.append(el("div", "stat-val", esc(v)), el("div", "stat-lab", esc(l)));
    return s;
  }));

  $("#column-grid").replaceChildren(...profile.columns.map((c) => {
    const card = el("div", `col-card role-${c.role}`);
    card.append(el("div", "col-name", esc(c.name)), el("div", "col-role", esc(c.role)));

    const bits = [`${c.n_unique.toLocaleString()} unique`];
    if (c.pct_missing > 0) bits.push(`${c.pct_missing}% missing`);
    if (c.role === "numeric" && c.stats.mean !== undefined) {
      bits.push(`mean ${fmt(c.stats.mean, 2)}`, `range ${fmt(c.stats.min, 1)}–${fmt(c.stats.max, 1)}`);
    }
    if (c.role === "categorical" && c.stats.top_value) {
      bits.push(`top: ${c.stats.top_value}`);
    }
    card.append(el("div", "col-meta", bits.map(esc).join(" · ")));
    c.warnings.forEach((w) => card.append(el("div", "col-warn", `⚠ ${esc(w)}`)));
    return card;
  }));

  const table = $("#preview-table");
  const thead = el("thead");
  const hrow = el("tr");
  preview.columns.forEach((c) => hrow.append(el("th", null, esc(c))));
  thead.append(hrow);
  const tbody = el("tbody");
  preview.rows.forEach((row) => {
    const tr = el("tr");
    row.forEach((v) => tr.append(el("td", typeof v === "number" ? "num" : null,
      v === null ? '<span style="opacity:.35">null</span>' : esc(v))));
    tbody.append(tr);
  });
  table.replaceChildren(thead, tbody);
}

/* ── Step 3: configure ────────────────────────────────────── */

function renderTargets({ profile, target_candidates }, suggested) {
  const select = $("#target-select");
  select.replaceChildren(...profile.columns
    .filter((c) => c.role !== "constant")
    .map((c) => {
      const opt = el("option");
      opt.value = c.name;
      opt.textContent = `${c.name}  (${c.role}, ${c.n_unique} unique)`;
      return opt;
    }));

  const pick = suggested && target_candidates.includes(suggested)
    ? suggested
    : target_candidates.at(-1) ?? profile.columns.at(-1).name;
  select.value = pick;
  describeTarget();
}

$("#target-select").addEventListener("change", describeTarget);

function describeTarget() {
  const name = $("#target-select").value;
  const col = state.profile.columns.find((c) => c.name === name);
  if (!col) return;
  // Mirrors detect_task() so the user sees the consequence before training.
  const guess = col.role === "numeric" && col.n_unique > 12
    ? "→ regression (predict a number)"
    : col.n_unique === 2
      ? "→ binary classification"
      : `→ ${col.n_unique}-class classification`;
  $("#target-help").textContent =
    `${col.n_unique} distinct values, ${col.pct_missing}% missing ${guess}`;
}

$("#train-btn").addEventListener("click", async () => {
  const btn = $("#train-btn");
  const target = $("#target-select").value;
  const preset = $("#preset-select").value;

  clearError("#train-error");
  btn.disabled = true;
  btn.textContent = "Training…";
  $("#train-progress").hidden = false;
  $("#train-bar").style.width = "0%";

  const res = await call("train", { target, preset });

  btn.disabled = false;
  btn.textContent = "Train models →";
  $("#train-progress").hidden = true;

  if (!res.ok) {
    showError("#train-error", res.error);
    console.error(res.traceback);
    return;
  }

  state.training = res;
  renderResults(res);
  renderPredictForm(res.input_schema);
  reveal("#step-results");
  reveal("#step-predict");
  $("#step-results").scrollIntoView({ behavior: "smooth", block: "start" });
});

/* ── Step 4: results ──────────────────────────────────────── */

function renderResults(res) {
  const banner = $("#task-banner");
  const dropped = res.dropped_columns.length
    ? ` Dropped ${res.dropped_columns.map((d) => `${d.name} (${d.reason})`).join(", ")}.`
    : "";
  banner.replaceChildren(
    el("div", "task-kind", esc(TASK_LABEL[res.task.kind] ?? res.task.kind)),
    el("div", "task-reason",
      `${esc(res.task.reason)} Ranked by ${esc(METRIC_LABEL[res.task.primary_metric] ?? res.task.primary_metric)}. ` +
      `Using ${res.feature_columns.length} features.${esc(dropped)}`));

  $("#warnings").replaceChildren(...res.warnings.map((w) =>
    el("div", "alert alert-warn", `⚠ ${esc(w)}`)));

  // Metric columns are whatever the engine actually reported for this task.
  const metricKeys = [...new Set(res.leaderboard.flatMap((r) => Object.keys(r.test_metrics)))];
  const table = $("#leaderboard");
  const thead = el("thead");
  const hrow = el("tr");
  ["#", "Model", "CV score", ...metricKeys.map((k) => METRIC_LABEL[k] ?? k), "Time"]
    .forEach((h) => hrow.append(el("th", null, esc(h))));
  thead.append(hrow);

  const tbody = el("tbody");
  res.leaderboard.forEach((r) => {
    const tr = el("tr", r.failed ? "failed" : r.is_best ? "best" : "");
    tr.append(el("td", "num", String(r.rank)));

    const nameCell = el("td");
    const tags = [
      r.is_best ? '<span class="tag tag-best">best</span>' : "",
      r.overfit && !r.failed ? '<span class="tag tag-overfit">overfit</span>' : "",
      r.failed ? '<span class="tag tag-fail">failed</span>' : "",
    ].join(" ");
    nameCell.append(el("div", "model-label", `${esc(r.label)} ${tags}`));
    nameCell.append(el("div", "model-blurb", esc(r.failed ? r.error : r.blurb)));
    tr.append(nameCell);

    tr.append(el("td", "num", r.failed ? "—" : `${fmt(r.cv_mean)} <span style="opacity:.5">± ${fmt(r.cv_std, 2)}</span>`));
    metricKeys.forEach((k) => tr.append(el("td", "num", fmt(r.test_metrics[k]))));
    tr.append(el("td", "num", `${r.fit_seconds}s`));
    tbody.append(tr);
  });
  table.replaceChildren(thead, tbody);

  const max = Math.max(...res.importance.map((i) => Math.abs(i.importance)), 1e-9);
  $("#importance").replaceChildren(...res.importance.map((i) => {
    const row = el("div", "imp-row");
    const track = el("div", "imp-track");
    const bar = el("div", "imp-bar");
    bar.style.width = `${Math.max(1.5, (Math.abs(i.importance) / max) * 100)}%`;
    track.append(bar);
    row.append(el("div", "imp-name", esc(i.feature)), track,
                el("div", "imp-val", fmt(i.importance, 4)));
    return row;
  }));
}

/* ── Step 5: predict ──────────────────────────────────────── */

function renderPredictForm(schema) {
  $("#predict-form").replaceChildren(...schema.map((f) => {
    const wrap = el("label", "field");
    wrap.append(el("span", "field-label", esc(f.name)));

    let input;
    if (f.type === "select") {
      input = el("select");
      input.append(el("option", null, "— leave blank —"));
      (f.options ?? []).forEach((o) => {
        const opt = el("option");
        opt.value = o; opt.textContent = o;
        input.append(opt);
      });
      if (f.default !== undefined && f.default !== null) input.value = f.default;
    } else if (f.type === "textarea") {
      input = el("textarea");
      input.placeholder = "Type some text…";
    } else if (f.type === "number") {
      input = el("input");
      input.type = "number";
      input.step = "any";
      if (f.default !== null && f.default !== undefined) input.value = fmt(f.default, 2);
      if (f.min !== null) input.placeholder = `${fmt(f.min, 1)} – ${fmt(f.max, 1)}`;
    } else {
      input = el("input");
      input.type = f.type === "date" ? "date" : "text";
    }
    input.name = f.name;
    input.dataset.field = f.name;
    wrap.append(input);
    wrap.append(el("span", "field-help", esc(f.role)));
    return wrap;
  }));
}

$("#predict-btn").addEventListener("click", async () => {
  const row = {};
  $("#predict-form").querySelectorAll("[data-field]").forEach((input) => {
    row[input.dataset.field] = input.value;
  });

  const btn = $("#predict-btn");
  btn.disabled = true;
  btn.textContent = "Predicting…";
  const res = await call("predict", { rows: [row] });
  btn.disabled = false;
  btn.textContent = "Predict";

  const panel = $("#prediction");
  panel.hidden = false;

  if (!res.ok) {
    panel.replaceChildren(el("div", "alert alert-error", esc(res.error)));
    return;
  }

  const value = res.predictions[0];
  const nodes = [
    el("div", "pred-label", `Predicted ${esc(state.training.task ? $("#target-select").value : "value")}`),
    el("div", "pred-value", esc(typeof value === "number" ? fmt(value, 2) : value)),
  ];

  const proba = res.probabilities?.[0];
  if (proba) {
    nodes.push(el("div", "pred-label", "Class probabilities"));
    Object.entries(proba)
      .sort((a, b) => b[1] - a[1])
      .forEach(([label, p]) => {
        const row = el("div", "proba-row");
        const track = el("div", "proba-track");
        const bar = el("div", "proba-bar");
        bar.style.width = `${p * 100}%`;
        track.append(bar);
        row.append(el("div", "proba-name", esc(label)), track,
                    el("div", "proba-val", `${(p * 100).toFixed(1)}%`));
        nodes.push(row);
      });
  }
  panel.replaceChildren(...nodes);
});
