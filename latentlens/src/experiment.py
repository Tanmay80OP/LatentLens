"""Training + evaluation pipeline for the shortcut-learning experiment.

Conditions (same CNN, same number of training images, same objects/labels/optimizer):
  A_baseline   - trained on background-correlated data
  B_bg_random  - trained on the same objects, backgrounds independent of class
  C_color_aug  - trained on the *correlated* data of A, plus on-the-fly colour augmentation
                 (random channel permutation, random grayscale, brightness jitter)

Evaluation sets (identical for every condition; never used for weight updates):
  id       - correlated backgrounds at the configured strength (the baseline's distribution)
  shifted  - backgrounds ALWAYS point to a wrong class (correlation reversed)  [primary shift]
  neutral  - backgrounds independent of the class (correlation removed)
Robustness gap = accuracy(id) - accuracy(shifted).
Shortcut-follow rate = share of `shifted` predictions equal to the class the background implies.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import CLASSES, N_CLASSES, SyntheticDataset, make_dataset
from .model import SmallCNN, count_parameters

CONDITIONS = {
    "A_baseline": "A. Baseline (correlated backgrounds)",
    "B_bg_random": "B. Background randomization",
    "C_color_aug": "C. Colour augmentation",
    "D_hue_aug": "D. Hue-rotation augmentation (opt-in)",
}
# Conditions run by default (the original A/B/C study). D is an opt-in follow-up experiment.
DEFAULT_CONDITIONS = ("A_baseline", "B_bg_random", "C_color_aug")
EVAL_SETS = ("id", "shifted", "neutral")
EVAL_LABELS = {
    "id": "In-distribution (correlated bg)",
    "shifted": "Shifted (reversed bg)",
    "neutral": "Neutral (independent bg)",
}
RESULTS_ROOT = Path(os.environ.get("LATENTLENS_RESULTS", Path(__file__).resolve().parent.parent / "results"))


# --------------------------------------------------------------------------- config
@dataclass
class ExperimentConfig:
    seed: int = 0
    n_seeds: int = 1  # seeds used: seed, seed+1, ...
    correlation: float = 0.9
    n_train: int = 1800
    n_val: int = 450
    n_test: int = 900
    epochs: int = 20
    batch_size: int = 64
    lr: float = 3e-3
    image_size: int = 32
    conditions: List[str] = field(default_factory=lambda: list(DEFAULT_CONDITIONS))
    # colour-augmentation strengths for condition C
    aug_perm_p: float = 0.8
    aug_gray_p: float = 0.2
    aug_brightness: float = 0.2
    # hue-rotation augmentation strength for condition D
    aug_hue_p: float = 1.0

    def validate(self) -> None:
        if not 0.0 <= self.correlation <= 1.0:
            raise ValueError("correlation must be between 0 and 1")
        if self.n_train < 30 or self.n_val < 30 or self.n_test < 30:
            raise ValueError("n_train / n_val / n_test must each be at least 30")
        if self.epochs < 1 or self.batch_size < 1 or self.n_seeds < 1:
            raise ValueError("epochs, batch_size and n_seeds must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        for name in ("aug_perm_p", "aug_gray_p", "aug_hue_p"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not self.conditions:
            raise ValueError("select at least one condition")
        bad = [c for c in self.conditions if c not in CONDITIONS]
        if bad:
            raise ValueError(f"unknown condition(s): {bad}")

    @property
    def seeds(self) -> List[int]:
        return [self.seed + i for i in range(self.n_seeds)]


# --------------------------------------------------------------------------- metrics
def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = np.asarray(pred), np.asarray(target)
    if len(target) == 0:
        raise ValueError("empty evaluation set")
    return float((pred == target).mean())


def robustness_gap(id_acc: float, shifted_acc: float) -> float:
    """In-distribution accuracy minus shifted accuracy (positive = loses accuracy under shift)."""
    return float(id_acc - shifted_acc)


def confusion_matrix(pred: np.ndarray, target: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(target), np.asarray(pred)), 1)  # rows = true, cols = predicted
    return cm


def per_class_accuracy(cm: np.ndarray) -> List[float]:
    totals = cm.sum(axis=1)
    return [float(cm[i, i] / totals[i]) if totals[i] else float("nan") for i in range(len(cm))]


def wilson_interval(correct: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion (test-set sampling uncertainty only)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


# --------------------------------------------------------------------------- data
def build_eval_sets(cfg: ExperimentConfig, seed: int) -> Dict[str, SyntheticDataset]:
    """The three fixed test sets for a seed. Same objects in each; only backgrounds differ."""
    s = seed * 10_000 + 3
    n, sz = cfg.n_test, cfg.image_size
    return {
        "id": make_dataset(n, s, "correlated", cfg.correlation, sz),
        "shifted": make_dataset(n, s, "reversed", 0.0, sz),
        "neutral": make_dataset(n, s, "random", 0.0, sz),
    }


def build_train_val(cond: str, cfg: ExperimentConfig, seed: int):
    """Train / validation sets for a condition. Identical object geometry across conditions."""
    tr_seed, va_seed = seed * 10_000 + 1, seed * 10_000 + 2
    sz = cfg.image_size
    if cond in ("A_baseline", "C_color_aug", "D_hue_aug"):
        train = make_dataset(cfg.n_train, tr_seed, "correlated", cfg.correlation, sz)
        val = make_dataset(cfg.n_val, va_seed, "correlated", cfg.correlation, sz)
    elif cond == "B_bg_random":
        train = make_dataset(cfg.n_train, tr_seed, "random", 0.0, sz)
        val = make_dataset(cfg.n_val, va_seed, "random", 0.0, sz)
    else:
        raise ValueError(f"unknown condition {cond!r}")
    return train, val


def color_augment(x: torch.Tensor, cfg: ExperimentConfig, gen: torch.Generator) -> torch.Tensor:
    """Per-sample colour augmentation (condition C). Operates on a (N,3,H,W) batch in [0,1]."""
    n = x.shape[0]
    out = x.clone()
    # random channel permutation
    perm_mask = torch.rand(n, generator=gen) < cfg.aug_perm_p
    if perm_mask.any():
        perms = torch.stack([torch.randperm(3, generator=gen) for _ in range(n)])
        permuted = torch.gather(x, 1, perms[:, :, None, None].expand_as(x))
        out = torch.where(perm_mask[:, None, None, None], permuted, out)
    # random grayscale
    gray_mask = torch.rand(n, generator=gen) < cfg.aug_gray_p
    if gray_mask.any():
        g = (0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]).expand_as(out)
        out = torch.where(gray_mask[:, None, None, None], g, out)
    # brightness jitter
    scale = 1 + (torch.rand(n, 1, 1, 1, generator=gen) * 2 - 1) * cfg.aug_brightness
    return (out * scale).clamp(0, 1)


# YIQ transform: Y = luminance, (I, Q) = chroma plane. Rotating (I, Q) rotates hue and leaves luminance unchanged.
_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


def hue_rotation_matrices(theta: torch.Tensor) -> torch.Tensor:
    """Per-sample 3x3 RGB->RGB matrices rotating hue by ``theta`` radians (shape (N,) -> (N,3,3))."""
    c, s = torch.cos(theta), torch.sin(theta)
    rot = torch.zeros(len(theta), 3, 3, dtype=_RGB2YIQ.dtype)
    rot[:, 0, 0] = 1.0
    rot[:, 1, 1], rot[:, 1, 2] = c, -s
    rot[:, 2, 1], rot[:, 2, 2] = s, c
    return torch.linalg.inv(_RGB2YIQ) @ rot @ _RGB2YIQ


def hue_augment(x: torch.Tensor, cfg: ExperimentConfig, gen: torch.Generator) -> torch.Tensor:
    """Condition D: rotate every image's hue by a uniformly random angle (with prob ``aug_hue_p``).

    Unlike channel permutation (condition C) this can produce *any* background hue, so the three palette
    colours stop being a reliable cue. Luminance is preserved; achromatic objects are unaffected.
    """
    n = x.shape[0]
    theta = torch.rand(n, generator=gen) * (2 * math.pi)
    theta = torch.where(torch.rand(n, generator=gen) < cfg.aug_hue_p, theta, torch.zeros_like(theta))
    out = torch.einsum("nij,njhw->nihw", hue_rotation_matrices(theta), x)
    return out.clamp(0, 1)


# --------------------------------------------------------------------------- train / eval
@torch.no_grad()
def predict(model: SmallCNN, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(x), batch_size):
        logits = model(torch.from_numpy(x[i : i + batch_size]))
        preds.append(logits.argmax(1).numpy())
    return np.concatenate(preds)


def evaluate_model(model: SmallCNN, eval_sets: Dict[str, SyntheticDataset]) -> Dict[str, dict]:
    out = {}
    for name, ds in eval_sets.items():
        pred = predict(model, ds.x)
        cm = confusion_matrix(pred, ds.y)
        correct = int((pred == ds.y).sum())
        lo, hi = wilson_interval(correct, len(ds.y))
        out[name] = {
            "accuracy": accuracy(pred, ds.y),
            "ci95": [lo, hi],
            "n": int(len(ds.y)),
            "per_class": per_class_accuracy(cm),
            "confusion": cm.tolist(),
            # share of predictions equal to the class the background implies
            "bg_follow_rate": float((pred == ds.bg).mean()),
        }
    return out


def train_condition(cond: str, cfg: ExperimentConfig, seed: int, on_epoch: Optional[Callable] = None):
    """Train one model. Test data is never touched here."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    train, val = build_train_val(cond, cfg, seed)
    model = SmallCNN(N_CLASSES)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    steps_per_epoch = math.ceil(len(train.y) / cfg.batch_size)
    # cosine LR decay for every condition (added after a pilot showed constant-LR instability in C)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * steps_per_epoch)
    xtr, ytr = torch.from_numpy(train.x), torch.from_numpy(train.y)
    gen = torch.Generator().manual_seed(seed * 7 + 11)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        order = torch.randperm(len(ytr), generator=gen)
        tot_loss, tot_correct = 0.0, 0
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i : i + cfg.batch_size]
            xb, yb = xtr[idx], ytr[idx]
            if cond == "C_color_aug":
                xb = color_augment(xb, cfg, gen)
            elif cond == "D_hue_aug":
                xb = hue_augment(xb, cfg, gen)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += int((logits.argmax(1) == yb).sum())
        val_acc = accuracy(predict(model, val.x), val.y)  # validation = condition's own training distribution
        rec = {
            "epoch": epoch,
            "train_loss": tot_loss / len(ytr),
            "train_acc": tot_correct / len(ytr),
            "val_acc": val_acc,
        }
        history.append(rec)
        if on_epoch:
            on_epoch(rec)
    return model, history


# --------------------------------------------------------------------------- orchestration
def summarize(runs: List[dict]) -> Dict[str, dict]:
    """Aggregate per-seed metrics per condition (mean, sample std, summed confusion matrices)."""
    summary: Dict[str, dict] = {}
    for cond in dict.fromkeys(r["condition"] for r in runs):
        rs = [r for r in runs if r["condition"] == cond]

        def ms(vals):
            vals = np.asarray(vals, dtype=float)
            return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        entry = {"n_seeds": len(rs), "seeds": [r["seed"] for r in rs]}
        for key in EVAL_SETS:
            if key in rs[0]["metrics"]:
                entry[f"{key}_acc_mean"], entry[f"{key}_acc_std"] = ms([r["metrics"][key]["accuracy"] for r in rs])
                entry[f"{key}_confusion"] = np.sum([r["metrics"][key]["confusion"] for r in rs], axis=0).tolist()
        entry["gap_mean"], entry["gap_std"] = ms([r["robustness_gap"] for r in rs])
        entry["bg_follow_mean"], entry["bg_follow_std"] = ms([r["metrics"]["shifted"]["bg_follow_rate"] for r in rs])
        summary[cond] = entry
    return summary


def run_experiment(
    cfg: ExperimentConfig,
    progress: Optional[Callable[[float, str], None]] = None,
    results_root: Optional[Path] = None,
    save_models: bool = True,
    run_name: Optional[str] = None,
) -> dict:
    """Run all selected conditions x seeds, save everything under results/<run_id>/."""
    cfg.validate()
    root = Path(results_root) if results_root else RESULTS_ROOT
    run_id = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = root / run_id
    n = 1
    while run_dir.exists():  # never overwrite
        run_dir = root / f"{run_id}_{n}"
        n += 1
    run_id = run_dir.name
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    report = progress or (lambda frac, msg: None)

    results = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "config": asdict(cfg),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "model_parameters": count_parameters(SmallCNN(N_CLASSES)),
        },
        "classes": CLASSES,
        "runs": [],
        "error": None,
    }
    jobs = [(c, s) for s in cfg.seeds for c in cfg.conditions]
    t_start = time.time()
    try:
        eval_cache: Dict[int, Dict[str, SyntheticDataset]] = {}
        for j, (cond, seed) in enumerate(jobs):
            base = j / len(jobs)

            def on_epoch(rec, j=j, cond=cond, seed=seed):
                frac = (j + rec["epoch"] / cfg.epochs) / len(jobs)
                report(frac, f"{CONDITIONS[cond]} | seed {seed} | epoch {rec['epoch']}/{cfg.epochs} "
                             f"| train acc {rec['train_acc']:.3f} | val acc {rec['val_acc']:.3f}")

            report(base, f"Starting {CONDITIONS[cond]} (seed {seed})")
            t0 = time.time()
            model, history = train_condition(cond, cfg, seed, on_epoch)
            train_s = time.time() - t0
            if seed not in eval_cache:
                eval_cache[seed] = build_eval_sets(cfg, seed)
            metrics = evaluate_model(model, eval_cache[seed])
            model_path = None
            if save_models:
                model_path = f"models/{cond}_seed{seed}.pt"
                torch.save(model.state_dict(), run_dir / model_path)
            results["runs"].append(
                {
                    "condition": cond,
                    "seed": seed,
                    "history": history,
                    "metrics": metrics,
                    "robustness_gap": robustness_gap(metrics["id"]["accuracy"], metrics["shifted"]["accuracy"]),
                    "train_seconds": round(train_s, 2),
                    "model_path": model_path,
                }
            )
            (run_dir / "results.json").write_text(json.dumps({**results, "status": "partial"}, indent=2))
        results["summary"] = summarize(results["runs"])
        results["status"] = "complete"
        results["total_seconds"] = round(time.time() - t_start, 2)
        report(1.0, "Done")
    except Exception as exc:  # keep partial results; never fabricate
        results["status"] = "failed"
        results["error"] = f"{type(exc).__name__}: {exc}"
        results["traceback"] = traceback.format_exc()
        if results["runs"]:
            results["summary"] = summarize(results["runs"])
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    (run_dir / "config.json").write_text(json.dumps(results["config"], indent=2))
    return results


# --------------------------------------------------------------------------- persistence helpers
def list_runs(root: Optional[Path] = None) -> List[str]:
    root = Path(root) if root else RESULTS_ROOT
    if not root.exists():
        return []
    return sorted((p.name for p in root.iterdir() if (p / "results.json").exists()), reverse=True)


def load_results(run_id: str, root: Optional[Path] = None) -> dict:
    root = Path(root) if root else RESULTS_ROOT
    return json.loads((root / run_id / "results.json").read_text())


def load_model(run_id: str, model_path: str, root: Optional[Path] = None) -> SmallCNN:
    root = Path(root) if root else RESULTS_ROOT
    model = SmallCNN(N_CLASSES)
    model.load_state_dict(torch.load(root / run_id / model_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def config_from_dict(d: dict) -> ExperimentConfig:
    known = {k: v for k, v in d.items() if k in ExperimentConfig.__dataclass_fields__}
    return ExperimentConfig(**known)


# --------------------------------------------------------------------------- CLI
def _cli() -> None:
    p = argparse.ArgumentParser(description="Run a LatentLens experiment from the command line")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=1)
    p.add_argument("--correlation", type=float, default=0.9)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n-train", type=int, default=1800)
    p.add_argument("--aug-perm-p", type=float, default=0.8, help="condition C channel-permutation probability")
    p.add_argument("--aug-gray-p", type=float, default=0.2, help="condition C grayscale probability")
    p.add_argument("--aug-hue-p", type=float, default=1.0, help="condition D hue-rotation probability")
    p.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS), choices=list(CONDITIONS),
                   help="conditions to run (D_hue_aug is opt-in)")
    p.add_argument("--name", default=None, help="run directory name (default: timestamp)")
    p.add_argument("--quick", action="store_true", help="tiny run for smoke-testing")
    a = p.parse_args()
    cfg = ExperimentConfig(seed=a.seed, n_seeds=a.n_seeds, correlation=a.correlation, epochs=a.epochs, n_train=a.n_train,
                           aug_perm_p=a.aug_perm_p, aug_gray_p=a.aug_gray_p, aug_hue_p=a.aug_hue_p, conditions=a.conditions)
    if a.quick:
        cfg.n_train, cfg.n_val, cfg.n_test, cfg.epochs = 300, 90, 150, 2
    res = run_experiment(cfg, progress=lambda f, m: print(f"[{f * 100:5.1f}%] {m}", flush=True), run_name=a.name)
    print("STATUS:", res["status"], res.get("error") or "")
    for cond, s in res.get("summary", {}).items():
        print(f"{CONDITIONS[cond]:<42} id={s['id_acc_mean']:.3f} shifted={s['shifted_acc_mean']:.3f} "
              f"gap={s['gap_mean']:+.3f} bg-follow={s['bg_follow_mean']:.3f}")
    print("saved to", RESULTS_ROOT / res["run_id"])


if __name__ == "__main__":
    _cli()
