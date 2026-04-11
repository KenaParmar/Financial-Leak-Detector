"""
inference.py — LLM agent for FinancialLeakEnv.

Mandatory environment variables
────────────────────────────────
  HF_TOKEN       required — used as LLM API key
  API_BASE_URL   optional — LLM endpoint  (default: https://api.openai.com/v1)
  MODEL_NAME     optional — model name    (default: gpt-4.1-mini)
  ENV_BASE_URL   optional — where your FastAPI server is running
                            (default: http://localhost:7860)

STDOUT FORMAT (exact spec)
──────────────────────────
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ── Environment variables ─────────────────────────────────────────────────

HF_TOKEN:     str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")

BENCHMARK             = "financial_leak_env"
MAX_STEPS             = 5
TEMPERATURE           = 0.2
MAX_TOKENS            = 512
SUCCESS_SCORE_THRESHOLD = 0.5

# ── HTTP client (requests — stdlib urllib fallback) ───────────────────────

def _http_post(url: str, payload: dict) -> dict:
    """POST JSON to url, return parsed response dict. Raises on HTTP error."""
    try:
        import requests
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except ImportError:
        pass
    # stdlib fallback
    import urllib.request, urllib.error
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def _http_get(url: str) -> dict:
    """GET url, return parsed response dict."""
    try:
        import requests
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except ImportError:
        pass
    import urllib.request
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read().decode())


# ── Wait for the env server to be ready ──────────────────────────────────

def _wait_for_server(max_wait: int = 60) -> bool:
    """Poll GET / until the server responds or timeout."""
    for _ in range(max_wait):
        try:
            _http_get(f"{ENV_BASE_URL}/")
            return True
        except Exception:
            time.sleep(1)
    return False


# ── Env HTTP wrappers ─────────────────────────────────────────────────────

def env_reset(task_id: str) -> Tuple[Optional[Dict], Optional[str]]:
    """POST /reset?task_id=<id> → (observation_dict, error_or_None)"""
    try:
        obs = _http_post(f"{ENV_BASE_URL}/reset?task_id={task_id}", {})
        return obs, None
    except Exception as exc:
        return None, str(exc)


def env_step(action: dict) -> Tuple[Optional[Dict], float, bool, Dict, Optional[str]]:
    """POST /step with action → (obs, reward, done, info, error_or_None)"""
    try:
        result = _http_post(f"{ENV_BASE_URL}/step", action)
        obs    = result.get("observation", {})
        reward = float(result.get("reward", 0.0))
        done   = bool(result.get("done", False))
        info   = result.get("info", {})
        return obs, reward, done, info, None
    except Exception as exc:
        return None, 0.0, False, {}, str(exc)


# ── LLM client (lazy) ─────────────────────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client
    if not HF_TOKEN:
        print("[WARN] HF_TOKEN not set — using fallback action.", flush=True)
        return None
    try:
        from openai import OpenAI
        _client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        return _client
    except Exception as exc:
        print(f"[WARN] OpenAI client failed: {exc}", flush=True)
        return None


# ── Prompts ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a personal-finance assistant.
    You receive a JSON observation of a user's transactions, subscriptions,
    spending summary, and savings goal.

    Decide:
    - Which subscriptions to cancel (only unused/redundant ones)
    - Which categories to reduce and by how much (0.0–1.0 fraction)
    - Key insights about spending patterns

    Rules:
    - Only cancel subscriptions that appear in the observation's subscriptions list.
    - reduce_categories values must be 0.0–1.0.
    - Insights must be specific and actionable.
    - Always include a forward savings plan in at least one insight.

    Respond ONLY with valid JSON — no markdown, no explanation:
    {
      "cancel_subscriptions": ["<name>", ...],
      "reduce_categories":    {"<category>": <0.0-1.0>, ...},
      "insights":             ["<insight>", ...]
    }
""").strip()


def _build_user_prompt(step: int, obs: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-3:]) if history else "None"
    return textwrap.dedent(f"""
        Step {step} of {MAX_STEPS}

        Observation:
        {json.dumps(obs, indent=2)}

        Recent history:
        {history_block}

        Provide your action as a JSON object.
    """).strip()


# ── Fallback action ───────────────────────────────────────────────────────

FALLBACK_ACTION = {
    "cancel_subscriptions": ["Gym", "MagazineX"],
    "reduce_categories":    {"food": 0.20, "entertainment": 0.25, "shopping": 0.20},
    "insights": [
        "Late-night food and entertainment spending is unusually high — set a cut-off time.",
        "Unused subscriptions (Gym, MagazineX) are silently leaking money each month.",
        "Plan: redirect cancelled subscription savings into an emergency fund.",
    ],
}


def get_action(step: int, obs: dict, history: List[str]) -> Tuple[dict, Optional[str]]:
    """Call LLM, parse response into action dict. Never raises."""
    client = _get_client()
    if client is None:
        return FALLBACK_ACTION, "no_client"
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(step, obs, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        # Validate keys exist
        action = {
            "cancel_subscriptions": parsed.get("cancel_subscriptions", []),
            "reduce_categories":    parsed.get("reduce_categories", {}),
            "insights":             parsed.get("insights", []),
        }
        return action, None
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return FALLBACK_ACTION, str(exc)


# ── Logging ───────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Episode runner ────────────────────────────────────────────────────────

def run_task(task_id: str) -> dict:
    """Run one full episode via HTTP. Never raises."""
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    history:     List[str]   = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── reset ──────────────────────────────────────────────────────
        obs, reset_err = env_reset(task_id)
        if obs is None:
            raise RuntimeError(f"reset failed: {reset_err}")

        for step_num in range(1, MAX_STEPS + 1):
            # ── get action from LLM ────────────────────────────────────
            action, llm_error = get_action(step_num, obs, history)

            # ── step the env ───────────────────────────────────────────
            obs, reward, done, info, step_error = env_step(action)
            error = llm_error or step_error

            if obs is None:
                # Server error — use zero reward and stop
                print(f"[DEBUG] step {step_num} server error: {step_error}", flush=True)
                done = True

            rewards.append(reward)
            steps_taken = step_num

            action_str = json.dumps(action, separators=(",", ":"))
            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step_num}: cancelled={action['cancel_subscriptions']} "
                f"reduced={list(action['reduce_categories'].keys())} reward={reward:.2f}"
            )

            if done:
                break

        score   = float(info.get("task_score", 0.0)) if info else 0.0
        score   = max(0.01, min(score, 0.99))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] run_task error task={task_id}: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "success": success,
        "steps":   steps_taken,
        "score":   score,
        "rewards": rewards,
    }


# ── Entry point ───────────────────────────────────────────────────────────

ALL_TASKS = [
    "leak_detection_easy",
    "duplicate_charge_easy",
    "micro_spending_easy",
    "behavior_analysis_medium",
    "weekend_spending_medium",
    "subscription_overlap_medium",
    "goal_optimization_hard",
    "tight_budget_hard",
    "multi_goal_hard",
]

if __name__ == "__main__":
    # Wait for the env server (Docker container may still be starting)
    print(f"[INFO] Waiting for env server at {ENV_BASE_URL} ...", flush=True)
    if not _wait_for_server(max_wait=60):
        print(f"[ERROR] Env server did not become ready at {ENV_BASE_URL}", flush=True)
        sys.exit(1)
    print("[INFO] Env server ready.", flush=True)

    results = [run_task(t) for t in ALL_TASKS]
    passed  = sum(1 for r in results if r["success"])
    avg     = sum(r["score"] for r in results) / len(results)
    print(
        f"\n[SUMMARY] {passed}/{len(results)} tasks passed  avg_score={avg:.3f}",
        flush=True,
    )
    sys.exit(0 if passed == len(results) else 1)