

import json
import os
import sys
import textwrap
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from env.financial_env import FinancialLeakEnv
from env.models import Action

load_dotenv()

# ── Environment variables ─────────────────────────────────────────────────

HF_TOKEN: str = os.getenv("HF_TOKEN") or ""
if not HF_TOKEN:
    raise EnvironmentError(
        "HF_TOKEN is not set. "
        "Add it in HF Space → Settings → Repository secrets."
    )

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
API_KEY:      str = HF_TOKEN   

BENCHMARK             = "financial_leak_env"
MAX_STEPS             = 5
TEMPERATURE           = 0.2
MAX_TOKENS            = 512
SUCCESS_SCORE_THRESHOLD = 0.5

# ── OpenAI client ─────────────────────────────────────────────────────────

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

# ── Prompts ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a personal-finance assistant.
    You will receive a JSON observation of a user's transactions, subscriptions,
    spending summary, and savings goal.

    Your job: decide which subscriptions to cancel, which categories to reduce,
    and write clear insights about the spending patterns.

    Rules:
    - Only cancel subscriptions that appear in the observation's subscriptions list.
    - reduce_categories values must be between 0.0 and 1.0 (fraction to cut).
    - Insights should be specific, actionable, and mention patterns you see
      (e.g. night spending, duplicates, micro-transactions, weekend spikes).
    - Always mention a forward savings plan in at least one insight.

    Respond ONLY with a valid JSON object — no markdown, no explanation:
    {
      "cancel_subscriptions": ["<name>", ...],
      "reduce_categories":    {"<category>": <0.0-1.0>, ...},
      "insights":             ["<insight>", ...]
    }
""").strip()


def build_user_prompt(step: int, obs_dict: dict, history: List[str]) -> str:
    history_block = "\n".join(history[-3:]) if history else "None"
    return textwrap.dedent(f"""
        Step {step} of {MAX_STEPS}

        Observation:
        {json.dumps(obs_dict, indent=2)}

        Recent history:
        {history_block}

        Provide your action as a JSON object.
    """).strip()


# ── LLM call ──────────────────────────────────────────────────────────────

_FALLBACK_ACTION = Action(
    cancel_subscriptions=["Gym", "MagazineX"],
    reduce_categories={"food": 0.20, "entertainment": 0.25, "shopping": 0.20},
    insights=[
        "Late-night spending on food and entertainment is unusually high — set a cut-off time.",
        "Unused subscriptions (Gym, MagazineX) are leaking money each month.",
        "Plan: redirect cancelled subscription savings into an emergency fund.",
    ],
)


def get_action(step: int, obs_dict: dict, history: List[str]) -> Tuple[Action, Optional[str]]:
    """Call the LLM; return (Action, error_or_None)."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(step, obs_dict, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        return Action(**parsed), None
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return _FALLBACK_ACTION, str(exc)


# ── Logging (exact spec format) ───────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ── Episode runner ────────────────────────────────────────────────────────

def run_task(task_id: str) -> dict:
    env = FinancialLeakEnv(task_id, max_steps=MAX_STEPS)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    info:        dict        = {}
    history:     List[str]   = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()

        for step_num in range(1, MAX_STEPS + 1):
            action, error = get_action(step_num, obs.model_dump(), history)

            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            steps_taken = step_num

            action_str = json.dumps(action.model_dump(), separators=(",", ":"))

            log_step(step=step_num, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step_num}: cancelled={action.cancel_subscriptions} "
                f"reduced={list(action.reduce_categories.keys())} reward={reward:.2f}"
            )

            if done:
                break

        score = float(info.get("task_score", 0.0))
        score = max(0.01, min(score, 0.99))
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] run_task error task={task_id}: {exc}", flush=True)

    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)
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
    results = [run_task(t) for t in ALL_TASKS]
    passed  = sum(1 for r in results if r["success"])
    avg     = sum(r["score"] for r in results) / len(results)
    print(
        f"\n[SUMMARY] {passed}/{len(results)} tasks passed  avg_score={avg:.3f}",
        flush=True,
    )
    sys.exit(0 if passed == len(results) else 1)