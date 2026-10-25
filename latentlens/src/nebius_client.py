"""Nemotron research assistant via Nebius Token Factory (OpenAI-compatible REST API).

Verified against public Nebius material (see README "Nebius / Nemotron verification"):
  * Base URL : https://api.tokenfactory.nebius.com/v1
  * Auth     : ``Authorization: Bearer <NEBIUS_API_KEY>``
  * Chat     : POST {base}/chat/completions   (OpenAI schema)
  * Models   : GET  {base}/models
  * Default model id : ``nvidia/Nemotron-3_5-Lightning`` (listed in the Nebius Token Factory
    model catalogue). Override with NEBIUS_MODEL, e.g. ``nvidia/nemotron-3-super-120b-a12b``.

The API key is only read from the environment (or a local, git-ignored ``.env``) and is never
logged, returned in error messages, or rendered in the UI. The numerical evaluation never
depends on this module.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"
DEFAULT_MODEL = "nvidia/Nemotron-3_5-Lightning"
TIMEOUT_S = 90

SYSTEM_PROMPT = (
    "You are the research assistant inside LatentLens, a lab that studies shortcut learning "
    "and distribution shift in a small CNN trained on synthetic circle/square/triangle images "
    "whose background colour can be correlated with the class.\n"
    "RULES:\n"
    "1. Use ONLY the numbers in the EXPERIMENT RESULTS JSON provided. Never invent, round up, "
    "or estimate metrics that are not present. If a number is missing, say so.\n"
    "2. You did not run any experiment. PyTorch code in the app computed the metrics; you only "
    "interpret them. Never claim to have run, trained, or tested anything.\n"
    "3. Distinguish measured results from hypotheses; label hypotheses explicitly.\n"
    "4. Do not claim causal proof from accuracy changes or feature maps. Mention uncertainty "
    "(few seeds, finite test sets, synthetic data, small model) and possible confounders.\n"
    "5. When asked for next experiments, give concrete, cheap, testable proposals.\n"
    "6. Be concise: short paragraphs or a few bullets."
)


class NebiusError(Exception):
    """Any failure talking to the API. Messages never contain the API key."""


@dataclass
class NebiusConfig:
    api_key: Optional[str]
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def redacted(self) -> dict:
        return {"base_url": self.base_url, "model": self.model, "api_key_set": self.configured}


def load_dotenv(path: Optional[Path] = None) -> None:
    """Tiny .env loader (KEY=VALUE lines). Existing environment variables win."""
    path = path or Path(__file__).resolve().parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if v:
            os.environ.setdefault(k.strip(), v)


def get_config() -> NebiusConfig:
    load_dotenv()
    return NebiusConfig(
        api_key=os.environ.get("NEBIUS_API_KEY"),
        base_url=(os.environ.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL).rstrip("/"),
        model=os.environ.get("NEBIUS_MODEL") or DEFAULT_MODEL,
    )


def _headers(cfg: NebiusConfig) -> Dict[str, str]:
    return {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}


def _scrub(text: str, cfg: NebiusConfig) -> str:
    if cfg.api_key:
        text = text.replace(cfg.api_key, "***")
    return text


def compact_results_for_llm(results: dict) -> dict:
    """Reduce a results dict to the numbers the assistant may cite (no raw histories/matrices)."""
    if not results:
        return {}
    cfg = results.get("config", {})
    out = {
        "run_id": results.get("run_id"),
        "status": results.get("status"),
        "config": {k: cfg.get(k) for k in (
            "seed", "n_seeds", "correlation", "n_train", "n_val", "n_test", "epochs",
            "batch_size", "lr", "conditions", "aug_perm_p", "aug_gray_p", "aug_brightness", "aug_hue_p")},
        "metric_definitions": {
            "id": "accuracy on test set with backgrounds correlated at config.correlation (the baseline's training distribution)",
            "shifted": "accuracy on test set whose background ALWAYS points to a wrong class",
            "neutral": "accuracy on test set whose background is independent of class",
            "gap": "id accuracy minus shifted accuracy",
            "bg_follow_rate": "fraction of shifted-set predictions equal to the class the background implies (shortcut following)",
            "chance_level": 1 / 3,
        },
        "conditions": {},
    }
    for cond, s in (results.get("summary") or {}).items():
        out["conditions"][cond] = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in s.items() if not k.endswith("_confusion")
        }
    if results.get("error"):
        out["error"] = results["error"]
    return out


def build_messages(results: dict, question: str, history: Optional[List[dict]] = None) -> List[dict]:
    context = json.dumps(compact_results_for_llm(results), indent=1)
    msgs: List[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": "EXPERIMENT RESULTS (measured by PyTorch code in this app):\n" + context},
    ]
    for m in (history or [])[-6:]:
        if m.get("role") in ("user", "assistant") and m.get("content"):
            msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": question})
    return msgs


def _strip_reasoning(text: str) -> str:
    """Remove chain-of-thought. Handles closed <think>..</think> blocks and the common reasoning-model
    case where only the closing tag is emitted (opening tag lives in the chat template)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.replace("<think>", "").strip()


def pick_model(available: List[str], preferred: str) -> Optional[str]:
    """Return ``preferred`` if served, else the best available Nemotron id (None if there is none).

    Nebius retires models on a schedule, so a stale default should degrade gracefully, not fail.
    Preference order: Lightning/Nano (cheap, fast) > Super > Ultra > anything else with 'nemotron'.
    """
    if preferred in available:
        return preferred
    nemo = [m for m in available if "nemotron" in m.lower()]
    for key in ("lightning", "nano", "super", "ultra"):
        for m in nemo:
            if key in m.lower():
                return m
    return nemo[0] if nemo else None


def chat(messages: List[dict], cfg: Optional[NebiusConfig] = None, max_tokens: int = 1500,
         temperature: float = 0.3, session=None) -> str:
    """Send a chat completion. Raises NebiusError with a safe, human-readable message."""
    cfg = cfg or get_config()
    if not cfg.configured:
        raise NebiusError("NEBIUS_API_KEY is not set. Add it to your environment or a local .env file "
                          "(see .env.example). The rest of LatentLens works without it.")
    http = session or requests
    payload = {"model": cfg.model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    try:
        r = http.post(f"{cfg.base_url}/chat/completions", headers=_headers(cfg), json=payload, timeout=TIMEOUT_S)
    except requests.Timeout:
        raise NebiusError("The Nebius request timed out. Try again in a moment.") from None
    except requests.RequestException as exc:
        raise NebiusError(_scrub(f"Network error contacting Nebius: {type(exc).__name__}", cfg)) from None
    if r.status_code == 401 or r.status_code == 403:
        raise NebiusError("Nebius rejected the credentials (HTTP %d). Check NEBIUS_API_KEY." % r.status_code)
    if r.status_code == 404:
        raise NebiusError(f"Model or endpoint not found (HTTP 404). Check NEBIUS_MODEL (currently '{cfg.model}') "
                          "against the Token Factory model catalogue.")
    if r.status_code == 429:
        raise NebiusError("Nebius rate limit reached (HTTP 429). Wait a bit and retry.")
    if r.status_code >= 400:
        detail = _scrub((r.text or "")[:300], cfg)
        raise NebiusError(f"Nebius returned HTTP {r.status_code}: {detail}")
    try:
        data = r.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        raise NebiusError("Unexpected response format from Nebius (no choices[0].message.content).") from None
    content = _strip_reasoning(content or "")
    if not content:
        finish = (data["choices"][0].get("finish_reason") or "") if isinstance(data["choices"][0], dict) else ""
        if finish == "length":
            raise NebiusError("The model ran out of tokens while reasoning and returned no answer. "
                              "Try again, or choose a less verbose model via NEBIUS_MODEL.")
        raise NebiusError("Nebius returned an empty answer.")
    return content


def list_models(cfg: Optional[NebiusConfig] = None, session=None) -> List[str]:
    """GET /models -> list of model ids (used for the connection test)."""
    cfg = cfg or get_config()
    if not cfg.configured:
        raise NebiusError("NEBIUS_API_KEY is not set.")
    http = session or requests
    try:
        r = http.get(f"{cfg.base_url}/models", headers=_headers(cfg), timeout=30)
    except requests.RequestException as exc:
        raise NebiusError(_scrub(f"Network error contacting Nebius: {type(exc).__name__}", cfg)) from None
    if r.status_code in (401, 403):
        raise NebiusError("Nebius rejected the credentials. Check NEBIUS_API_KEY.")
    if r.status_code >= 400:
        raise NebiusError(f"Nebius returned HTTP {r.status_code} for /models.")
    try:
        return [m["id"] for m in r.json()["data"]]
    except (ValueError, KeyError, TypeError):
        raise NebiusError("Unexpected /models response format.") from None


def ask(results: dict, question: str, history: Optional[List[dict]] = None,
        cfg: Optional[NebiusConfig] = None, session=None) -> str:
    return chat(build_messages(results, question, history), cfg=cfg, session=session)


QUICK_PROMPTS = {
    "Explain the configuration": "Explain this experiment's configuration in plain language: what each condition does and what the test sets mean.",
    "Summarize measured results": "Summarize the measured results. Cite the exact numbers from the JSON.",
    "Compare baseline vs interventions": "Compare the baseline with each intervention using the measured numbers, including the robustness gap and bg_follow_rate. State which differences may be within noise.",
    "Suggest the next experiment": "Suggest the single most informative next experiment, with a concrete setting to change and what outcome would support or refute the hypothesis.",
    "Limitations & confounders": "List the main limitations and possible confounders of this experiment given its configuration (seeds, data size, synthetic data, model size).",
}
