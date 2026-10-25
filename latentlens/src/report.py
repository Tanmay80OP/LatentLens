"""Export helpers: CSV, JSON and a Markdown report built only from measured values."""
from __future__ import annotations

import csv
import io
import json
from typing import List

from .experiment import CONDITIONS, EVAL_LABELS


def runs_to_rows(results: dict) -> List[dict]:
    rows = []
    for r in results.get("runs", []):
        row = {"run_id": results.get("run_id"), "condition": r["condition"], "seed": r["seed"]}
        for key, m in r["metrics"].items():
            row[f"{key}_accuracy"] = round(m["accuracy"], 6)
            row[f"{key}_ci95_low"] = round(m["ci95"][0], 6)
            row[f"{key}_ci95_high"] = round(m["ci95"][1], 6)
            row[f"{key}_bg_follow_rate"] = round(m["bg_follow_rate"], 6)
            for i, v in enumerate(m["per_class"]):
                row[f"{key}_acc_class{i}"] = round(v, 6)
        row["robustness_gap"] = round(r["robustness_gap"], 6)
        row["train_seconds"] = r["train_seconds"]
        rows.append(row)
    return rows


def to_csv(results: dict) -> str:
    rows = runs_to_rows(results)
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def to_json(results: dict) -> str:
    return json.dumps(results, indent=2)


def to_markdown(results: dict) -> str:
    cfg = results.get("config", {})
    L = [f"# LatentLens experiment report — `{results.get('run_id')}`", ""]
    L += [f"*Created:* {results.get('created_at')}  |  *Status:* **{results.get('status')}**", ""]
    if results.get("error"):
        L += [f"> **Run error:** {results['error']}", ""]
    L += ["## Research question", "",
          "Can targeted training interventions reduce shortcut learning and improve a CNN's generalization "
          "when background–label correlations change?", "", "## Configuration (reproducible)", ""]
    L += [f"- `{k}`: {v}" for k, v in cfg.items()]
    env = results.get("environment", {})
    L += [f"- environment: Python {env.get('python')}, torch {env.get('torch')}, device {env.get('device')}", ""]
    summ = results.get("summary") or {}
    if summ:
        L += ["## Measured results (mean ± std over seeds; std = 0 if one seed)", "",
              "| Condition | Seeds | ID acc | Shifted acc | Neutral acc | Robustness gap | BG-follow rate |",
              "|---|---|---|---|---|---|---|"]
        for c, s in summ.items():
            L.append(
                f"| {CONDITIONS.get(c, c)} | {s['n_seeds']} | {s['id_acc_mean']:.3f} ± {s['id_acc_std']:.3f} "
                f"| {s['shifted_acc_mean']:.3f} ± {s['shifted_acc_std']:.3f} "
                f"| {s['neutral_acc_mean']:.3f} ± {s['neutral_acc_std']:.3f} "
                f"| {s['gap_mean']:+.3f} ± {s['gap_std']:.3f} | {s['bg_follow_mean']:.3f} |")
        L += ["", "Chance level is 0.333. Robustness gap = ID accuracy − shifted accuracy. "
              "BG-follow rate = fraction of shifted-set predictions equal to the class implied by the background.", ""]
        base = summ.get("A_baseline")
        if base:
            L += ["## Differences versus baseline (measured)", ""]
            for c, s in summ.items():
                if c == "A_baseline":
                    continue
                L.append(f"- {CONDITIONS.get(c, c)}: shifted accuracy {s['shifted_acc_mean'] - base['shifted_acc_mean']:+.3f}, "
                         f"ID accuracy {s['id_acc_mean'] - base['id_acc_mean']:+.3f}, "
                         f"gap {s['gap_mean'] - base['gap_mean']:+.3f}")
            L.append("")
    L += ["## Caveats", "",
          "- Synthetic data, one small CNN, few seeds: results illustrate the phenomenon and do not establish general claims.",
          "- Accuracy changes do not prove *why* a model's behaviour changed; feature maps are diagnostic only.",
          "- The 'ID' test set uses the baseline's correlated distribution for all conditions; for B it is therefore itself a shift.",
          "- Test sets were never used to update weights or choose hyper-parameters.", ""]
    return "\n".join(L)


def paired_shifted_diffs(results: dict, cond: str, base: str = "A_baseline") -> List[float]:
    """Per-seed difference in shifted accuracy (cond − base), for seeds present in both conditions."""
    by = {}
    for r in results.get("runs", []):
        by.setdefault(r["condition"], {})[r["seed"]] = r["metrics"]["shifted"]["accuracy"]
    a, b = by.get(cond, {}), by.get(base, {})
    return [a[s] - b[s] for s in sorted(set(a) & set(b))]


def plain_language_summary(results: dict) -> List[str]:
    """Deterministic sentences built only from measured numbers (no model involved)."""
    summ = results.get("summary") or {}
    if not summ:
        return []
    out = []
    n_seeds = max(s["n_seeds"] for s in summ.values())
    n_test = results["runs"][0]["metrics"]["shifted"]["n"] if results.get("runs") else None
    base = summ.get("A_baseline")
    for c, s in summ.items():
        out.append(
            f"**{CONDITIONS.get(c, c)}** scored {s['id_acc_mean']:.1%} on the in-distribution test set and "
            f"{s['shifted_acc_mean']:.1%} on the shifted set (backgrounds always pointing to a wrong class), "
            f"a robustness gap of {s['gap_mean']:+.1%}. On the shifted set it predicted the background-implied "
            f"class {s['bg_follow_mean']:.1%} of the time (chance ≈ 33.3% for a model that ignores shape and background).")
    if base:
        for c, s in summ.items():
            if c == "A_baseline":
                continue
            d = s["shifted_acc_mean"] - base["shifted_acc_mean"]
            direction = "higher" if d > 0 else "lower" if d < 0 else "equal"
            note = ""
            if n_seeds == 1 and n_test:
                # rough 95% sampling error of a difference of two independent proportions on n_test images
                se = 1.96 * (2 * 0.25 / n_test) ** 0.5
                if abs(d) < se:
                    note = f" This difference is smaller than ≈{se:.1%}, the approximate test-set sampling error, so it is not distinguishable from noise with this single seed."
            if n_seeds > 1:
                diffs = paired_shifted_diffs(results, c)
                if len(diffs) > 1:
                    same_sign = all(x > 0 for x in diffs) or all(x < 0 for x in diffs)
                    note = (f" The sign was the same in all {len(diffs)} paired seeds "
                            f"(per-seed differences: {', '.join(f'{x:+.1%}' for x in diffs)})."
                            if same_sign else
                            f" The sign was not consistent across the {len(diffs)} paired seeds "
                            f"(per-seed differences: {', '.join(f'{x:+.1%}' for x in diffs)}), so treat it as unclear.")
            out.append(f"Versus baseline, {CONDITIONS.get(c, c)} had {abs(d):.1%} {direction} shifted accuracy.{note}")
    if n_seeds == 1:
        out.append("Only one seed was run, so seed-to-seed variability is unmeasured; run several seeds before drawing conclusions.")
    return out
