"""LatentLens — interactive lab for shortcut learning and distribution shift."""
from __future__ import annotations

import json
import traceback

import numpy as np
import pandas as pd
import streamlit as st

from src import nebius_client as nb
from src.dataset import BG_NAMES, CLASSES
from src.experiment import (CONDITIONS, DEFAULT_CONDITIONS, EVAL_LABELS, ExperimentConfig, build_eval_sets, config_from_dict,
                            list_runs, load_model, load_results, predict, run_experiment)
from src.model import LAYER_NAMES
from src.report import plain_language_summary, runs_to_rows, to_csv, to_json, to_markdown
from src.visualization import (SHORT, accuracy_bars, confusion_figure, embedding_figure, feature_map_figure,
                               follow_bars, gap_bars, image_grid, learning_curves)

st.set_page_config(page_title="LatentLens", page_icon="🔬", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1250px;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    .ll-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;}
    .ll-muted {color: rgba(128,128,128,1); font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ state
ss = st.session_state
if "pending_run" in ss:  # set after an experiment finishes; applied before the selector is drawn
    ss["run_id"] = ss.pop("pending_run")
ss.setdefault("chat", [])
runs_available = list_runs()
if runs_available and ss.get("run_id") not in runs_available:
    ss["run_id"] = runs_available[0]


def current_results():
    rid = ss.get("run_id")
    if not rid:
        return None
    try:
        return load_results(rid)
    except Exception as exc:  # corrupted / unreadable file
        st.sidebar.error(f"Could not read {rid}: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def cached_eval_sets(cfg_json: str, seed: int):
    return build_eval_sets(config_from_dict(json.loads(cfg_json)), seed)


@st.cache_resource(show_spinner=False)
def cached_model(run_id: str, model_path: str):
    return load_model(run_id, model_path)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🔬 LatentLens")
st.sidebar.caption("Shortcut learning & distribution shift lab")
if runs_available:
    st.sidebar.selectbox("Experiment run", runs_available, key="run_id",
                         format_func=lambda r: f"{r}  [precomputed example]" if r.startswith("example_") else r,
                         help="Results are stored in results/<run_id>/ and survive page reloads.")
else:
    st.sidebar.info("No experiment has been run yet. Open **Experiment Lab**.")
ncfg = nb.get_config()
if ss.get("model_override"):  # chosen after "Test connection" found the default id is no longer served
    ncfg.model = ss["model_override"]
st.sidebar.markdown("---")
st.sidebar.markdown("**Nemotron / Nebius**")
st.sidebar.write("API key: " + ("✅ set" if ncfg.configured else "⚠️ not set"))
st.sidebar.caption(f"Model: `{ncfg.model}`")
results = current_results()

tabs = st.tabs(["Overview", "Experiment Lab", "Model Comparison", "Representation Explorer",
                "Nemotron Research Assistant", "Reproducibility & Export"])

# ------------------------------------------------------------------ overview
with tabs[0]:
    st.title("LatentLens")
    st.subheader("Do targeted training interventions reduce shortcut learning?")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            """
**Shortcut learning** happens when a network solves a task using an easy cue that is *correlated* with the label in the
training data but is not what we actually care about. **Distribution shift** is what exposes it: when that correlation
changes at test time, accuracy can collapse.

Here, a small CNN must classify the **shape** (circle, square, triangle) of an object. During training the **background
colour** is statistically tied to the class (e.g. red ↔ circle). A model that learns "red ⇒ circle" looks excellent on
matching data and fails when backgrounds are reversed.

**Research question:** *Can targeted training interventions reduce shortcut learning and improve a CNN's generalization
when background–label correlations change?*

| Condition | What changes |
|---|---|
| **A. Baseline** | Train on correlated-background data |
| **B. Background randomization** | Same objects, backgrounds independent of class |
| **C. Colour augmentation** | Same data as A + random channel permutation / grayscale / brightness |

All conditions use the identical CNN, number of training images, optimizer and test sets. Whether B or C help is an
**empirical question** — this app reports what is measured, nothing is assumed.
            """
        )
    with c2:
        st.markdown("##### Experiment status")
        if results is None:
            st.warning("No results yet.")
        else:
            st.write(f"Run `{results['run_id']}` — **{results['status']}**")
            if results["run_id"].startswith("example_"):
                st.info("📦 **Precomputed example** shipped with the repository (generated earlier by "
                        "`python -m src.experiment`). It was not produced by a click in this session. "
                        "Run your own in Experiment Lab.")
            if results["status"] == "failed":
                st.error(results.get("error"))
            if results.get("summary"):
                st.dataframe(pd.DataFrame(
                    {SHORT[c]: {"ID acc": f"{s['id_acc_mean']:.3f}", "Shifted acc": f"{s['shifted_acc_mean']:.3f}",
                                "Gap": f"{s['gap_mean']:+.3f}"} for c, s in results["summary"].items()}).T,
                    width="stretch")
        st.markdown("##### How to start")
        st.markdown("1. Open **Experiment Lab**, keep the defaults (about a minute on a laptop CPU).\n"
                    "2. Click **Run experiment**.\n3. Inspect **Model Comparison** and **Representation Explorer**.\n"
                    "4. Ask the **Nemotron assistant** to interpret the measured numbers.\n5. Export from **Reproducibility & Export**.")

# ------------------------------------------------------------------ experiment lab
with tabs[1]:
    st.header("Experiment Lab")
    if ss.get("flash"):
        kind, msg = ss.pop("flash")
        (st.success if kind == "ok" else st.error)(msg)
    with st.form("lab"):
        c1, c2, c3 = st.columns(3)
        correlation = c1.slider("Background–label correlation (training)", 0.34, 1.0, 0.9, 0.01,
                                help="Share of training images whose background matches their class. 0.33 ≈ no correlation.")
        seed = c2.number_input("Random seed", 0, 10_000, 0, 1)
        n_seeds = c3.slider("Number of seeds", 1, 5, 1, help="More seeds = mean ± std but longer runtime.")
        c4, c5, c6 = st.columns(3)
        epochs = c4.slider("Epochs", 2, 40, 20)
        n_train = c5.select_slider("Training images", [300, 600, 900, 1200, 1800, 2400, 3600], 1800)
        conds = c6.multiselect("Conditions", list(CONDITIONS), default=list(DEFAULT_CONDITIONS),
                               format_func=lambda c: CONDITIONS[c])
        with st.expander("Advanced settings"):
            a1, a2, a3 = st.columns(3)
            lr = a1.number_input("Learning rate", 1e-4, 1e-1, 3e-3, format="%.4f")
            batch = a2.select_slider("Batch size", [16, 32, 64, 128], 64)
            n_test = a3.select_slider("Test images per set", [300, 600, 900, 1500, 3000], 900)
            b1, b2, b3 = st.columns(3)
            aug_perm = b1.slider("Aug: channel-permutation prob (C)", 0.0, 1.0, 0.8, 0.05)
            aug_gray = b2.slider("Aug: grayscale prob (C)", 0.0, 1.0, 0.2, 0.05)
            aug_br = b3.slider("Aug: brightness jitter (C)", 0.0, 0.5, 0.2, 0.05)
            aug_hue = st.slider("Aug: hue-rotation prob (D, opt-in condition)", 0.0, 1.0, 1.0, 0.05,
                                help="Condition D rotates every image's hue by a random angle. Add it to Conditions to run it.")
        est = len(conds) * n_seeds * epochs * n_train / 1800 * 1.2
        st.caption(f"Rough CPU time estimate: ~{max(est, 1):.0f}–{max(est * 2.5, 2):.0f} s "
                   "(depends on your machine). Training never touches the test sets.")
        go_btn = st.form_submit_button("▶ Run experiment", type="primary")
    if go_btn:
        cfg = ExperimentConfig(seed=int(seed), n_seeds=int(n_seeds), correlation=float(correlation),
                               n_train=int(n_train), n_val=450, n_test=int(n_test), epochs=int(epochs),
                               batch_size=int(batch), lr=float(lr), conditions=list(conds),
                               aug_perm_p=float(aug_perm), aug_gray_p=float(aug_gray), aug_brightness=float(aug_br),
                               aug_hue_p=float(aug_hue))
        try:
            cfg.validate()
        except ValueError as exc:
            st.error(f"Invalid settings: {exc}")
        else:
            bar = st.progress(0.0, text="Starting…")
            res = run_experiment(cfg, progress=lambda f, m: bar.progress(min(float(f), 1.0), text=m))
            if res["status"] == "complete":
                ss["flash"] = ("ok", f"Run {res['run_id']} finished in {res['total_seconds']} s. "
                                     "Open **Model Comparison** to see the measured results.")
            else:
                ss["flash"] = ("err", f"Run {res['run_id']} failed: {res['error']}")
            ss["pending_run"] = res["run_id"]
            st.rerun()
    if results and results["status"] == "failed":
        st.error(f"Selected run failed: {results['error']}")
        with st.expander("Traceback"):
            st.code(results.get("traceback", "n/a"))

# ------------------------------------------------------------------ comparison
with tabs[2]:
    st.header("Model Comparison")
    if not results or not results.get("summary"):
        st.info("Run an experiment first (Experiment Lab).")
    else:
        summ = results["summary"]
        if results["status"] != "complete":
            st.warning(f"This run is **{results['status']}**: showing the results that were completed before it stopped.")
        rows = []
        for c, s in summ.items():
            rows.append({"Condition": CONDITIONS[c], "Seeds": s["n_seeds"],
                         "ID acc": f"{s['id_acc_mean']:.3f} ± {s['id_acc_std']:.3f}",
                         "Shifted acc": f"{s['shifted_acc_mean']:.3f} ± {s['shifted_acc_std']:.3f}",
                         "Neutral acc": f"{s['neutral_acc_mean']:.3f} ± {s['neutral_acc_std']:.3f}",
                         "Robustness gap": f"{s['gap_mean']:+.3f} ± {s['gap_std']:.3f}",
                         "BG-follow rate": f"{s['bg_follow_mean']:.3f}"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("± = sample standard deviation across seeds (0 for a single seed). Chance = 0.333. "
                   "Gap = ID − shifted. The ID set follows the *baseline's* correlated distribution for every condition.")
        c1, c2 = st.columns(2)
        c1.plotly_chart(accuracy_bars(summ), width="stretch")
        c2.plotly_chart(gap_bars(summ), width="stretch")
        c3, c4 = st.columns(2)
        c3.plotly_chart(follow_bars(summ), width="stretch")
        c4.plotly_chart(learning_curves(results["runs"]), width="stretch")

        st.subheader("Confusion matrices & per-class accuracy")
        which = st.radio("Test set", list(EVAL_LABELS), format_func=EVAL_LABELS.get, horizontal=True)
        cols = st.columns(len(summ))
        for col, (c, s) in zip(cols, summ.items()):
            col.plotly_chart(confusion_figure(s[f"{which}_confusion"], SHORT[c]), width="stretch")
        pc = []
        for c in summ:
            cm = np.array(summ[c][f"{which}_confusion"])
            pc.append({"Condition": SHORT[c], **{k: f"{cm[i, i] / cm[i].sum():.3f}" for i, k in enumerate(CLASSES)}})
        st.dataframe(pd.DataFrame(pc), hide_index=True, width="stretch")

        st.subheader("Interpretation grounded in measured values")
        for line in plain_language_summary(results):
            st.markdown("- " + line)
        st.caption("Generated deterministically from the numbers above. Measured results are not proof of *why* "
                   "a model changed behaviour; mechanisms remain hypotheses.")
        with st.expander("Per-seed results"):
            st.dataframe(pd.DataFrame(runs_to_rows(results)), width="stretch")

# ------------------------------------------------------------------ explorer
with tabs[3]:
    st.header("Representation Explorer")
    models_avail = [r for r in (results or {}).get("runs", []) if r.get("model_path")]
    if not models_avail:
        st.info("Run an experiment first; trained models are saved with each run.")
    else:
        cfg_json = json.dumps(results["config"])
        c1, c2, c3, c4 = st.columns(4)
        label = c1.selectbox("Model", [f"{SHORT[r['condition']]} · seed {r['seed']}" for r in models_avail])
        run = models_avail[[f"{SHORT[r['condition']]} · seed {r['seed']}" for r in models_avail].index(label)]
        which = c2.selectbox("Test set", list(EVAL_LABELS), index=1, format_func=EVAL_LABELS.get)
        view = c3.selectbox("Show", ["Random examples", "Misclassified", "Followed the background"])
        layer = c4.selectbox("Layer", list(LAYER_NAMES), index=1)
        shuffle = st.number_input("Example shuffle seed", 0, 9999, 0, 1,
                                  help="Change to see a different random sample of images.")
        try:
            model = cached_model(results["run_id"], run["model_path"])
            ds = cached_eval_sets(cfg_json, run["seed"])[which]
            pred = predict(model, ds.x)
        except Exception as exc:
            st.error(f"Could not load model or data: {exc}")
        else:
            wrong = np.where(pred != ds.y)[0]
            follow = np.where((pred == ds.bg) & (pred != ds.y))[0]
            pool = {"Random examples": np.arange(len(ds.y)), "Misclassified": wrong, "Followed the background": follow}[view]
            st.caption(f"Accuracy on this set: {(pred == ds.y).mean():.3f} · errors: {len(wrong)} · "
                       f"errors that match the background-implied class: {len(follow)}")
            if len(pool) == 0:
                st.success("No examples in this category for this model/test set.")
            else:
                rng = np.random.default_rng(int(shuffle))
                pick = np.sort(rng.choice(pool, size=min(12, len(pool)), replace=False))
                caps = [f"{'✓' if pred[i] == ds.y[i] else '✗'} true {CLASSES[ds.y[i]]}<br>pred {CLASSES[pred[i]]} · bg {BG_NAMES[ds.bg[i]]}"
                        for i in pick]
                st.plotly_chart(image_grid(ds.x[pick], caps), width="stretch")
                st.markdown("##### Feature maps for one example")
                sel = st.selectbox("Example", [int(i) for i in pick],
                                   format_func=lambda i: f"#{i}: true {CLASSES[ds.y[i]]} → pred {CLASSES[pred[i]]}")
                st.plotly_chart(feature_map_figure(model, ds.x[sel], layer), width="stretch")
                st.caption("Shows the 16 channels with the highest mean activation at the chosen layer (min–max scaled per channel).")
            with st.expander("Low-dimensional embedding (PCA of penultimate features)"):
                color_by = st.radio("Colour by", ["true shape", "background colour"], horizontal=True)
                n_emb = min(450, len(ds.y))
                st.plotly_chart(embedding_figure(model, ds.x[:n_emb], ds.y[:n_emb], ds.bg[:n_emb], color_by),
                                width="stretch")
            st.markdown(
                "**What these show — and what they don't.** Example grids show *where* the model fails. Feature maps are "
                "activations, not explanations: a channel that lights up on the background suggests the network "
                "represents it, but does **not** prove the prediction was *caused* by it. The PCA is a linear 2-D "
                "projection and can hide structure. Treat all three as diagnostic evidence; use the quantitative "
                "shifted-set metrics for claims, and interventions (e.g. conditions B/C) to test causes.")

# ------------------------------------------------------------------ assistant
with tabs[4]:
    st.header("Nemotron Research Assistant")
    st.caption("Explains and critiques the *measured* results. It does not compute metrics or run experiments — PyTorch does.")
    if not ncfg.configured:
        st.warning("**NEBIUS_API_KEY is not set**, so the assistant is disabled. Create a key in the Nebius Token Factory "
                   "console, copy `.env.example` to `.env`, add `NEBIUS_API_KEY=...` and restart the app. "
                   "Everything else in LatentLens works without it.")
    st.write(f"Endpoint: `{ncfg.base_url}` · Model: `{ncfg.model}`")
    if ncfg.configured and st.button("Test connection"):
        try:
            ids = nb.list_models(ncfg)
            if ncfg.model in ids:
                st.success(f"Connected. `{ncfg.model}` is available ({len(ids)} models listed).")
            else:
                near = [i for i in ids if "nemotron" in i.lower()][:8]
                alt = nb.pick_model(ids, ncfg.model)
                st.warning(f"Connected, but `{ncfg.model}` was not in the model list. Nemotron models seen: {near}.")
                if alt:
                    ss["suggested_model"] = alt
                else:
                    st.error("No Nemotron model is currently listed for this key.")
        except nb.NebiusError as exc:
            st.error(str(exc))
    if ss.get("suggested_model") and ss["suggested_model"] != ncfg.model:
        if st.button(f"Use `{ss['suggested_model']}` for this session"):
            ss["model_override"] = ss.pop("suggested_model")
            st.rerun()
    ready = bool(results and results.get("summary")) and ncfg.configured
    if not (results and results.get("summary")):
        st.info("Run an experiment so the assistant has measured results to discuss.")
    cols = st.columns(len(nb.QUICK_PROMPTS))
    quick = None
    for col, (label, prompt) in zip(cols, nb.QUICK_PROMPTS.items()):
        if col.button(label, disabled=not ready, width="stretch"):
            quick = prompt
    typed = st.chat_input("Ask about the latest experiment…", disabled=not ready)
    question = quick or typed
    if question:
        try:
            with st.spinner("Asking Nemotron…"):
                answer = nb.ask(results, question, ss["chat"], cfg=ncfg)
            ss["chat"] += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        except nb.NebiusError as exc:
            st.error(f"Assistant unavailable: {exc}")
    for m in ss["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if ss["chat"] and st.button("Clear conversation"):
        ss["chat"] = []
        st.rerun()

# ------------------------------------------------------------------ export
with tabs[5]:
    st.header("Reproducibility & Export")
    if not results:
        st.info("No run selected.")
    else:
        c1, c2 = st.columns(2)
        c1.markdown("##### Configuration")
        c1.json(results["config"])
        c2.markdown("##### Environment")
        c2.json(results.get("environment", {}))
        seeds_used = [results["config"]["seed"] + i for i in range(results["config"]["n_seeds"])]
        st.write(f"Seeds used: **{', '.join(map(str, seeds_used))}**. Datasets, initial weights and shuffling are all "
                 "derived from these seeds. Reproduce from the command line: "
                 "`python -m src.experiment --seed S --n-seeds N --correlation C`.")
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇ Results (JSON)", to_json(results), f"{results['run_id']}.json", "application/json")
        d2.download_button("⬇ Per-seed metrics (CSV)", to_csv(results), f"{results['run_id']}.csv", "text/csv",
                           disabled=not results.get("runs"))
        d3.download_button("⬇ Report (Markdown)", to_markdown(results), f"{results['run_id']}_report.md", "text/markdown")
        with st.expander("Preview report"):
            st.markdown(to_markdown(results))
