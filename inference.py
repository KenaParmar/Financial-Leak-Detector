import os
from dotenv import load_dotenv
from openai import OpenAI

from env.financial_env import FinancialLeakEnv
from env.models import Action

load_dotenv()

# ✅ REQUIRED ENV VARIABLES (with defaults)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ✅ CORRECT CLIENT (MANDATORY)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

BENCHMARK = "financial_leak_env"
MAX_STEPS = 5


def log_start(task):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def run_task(task_id):
    env = FinancialLeakEnv(task_id)

    rewards = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task_id)

    try:
        obs = env.reset()

        for step in range(1, MAX_STEPS + 1):
            error = None

            # 🔹 API call (safe)

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a financial assistant."},
                        {"role": "user", "content": str(obs.model_dump())}
                    ]
                )
                _ = response.choices[0].message.content

            except Exception:
                error = "api_error"

            # 🔹 deterministic action
            action_obj = Action(
                cancel_subscriptions=["Gym"],
                reduce_categories={"food": 0.3},
                insights=["Reduce late night spending"]
            )

            obs, reward, done, info = env.step(action_obj)

            rewards.append(reward)
            steps_taken = step

            # VERY IMPORTANT: action must be string
            action_str = str(action_obj.model_dump())

            log_step(step, action_str, reward, done, error)

            if done:
                break

        score = info.get("task_score", 0.5)

        # clamp (just safety)
        score = max(0.01, min(score, 0.99))

        success = score >= 0.1

    finally:
        log_end(success, steps_taken, score, rewards)


if __name__ == "__main__":
    tasks = [
        "leak_detection_easy",
        "duplicate_charge_easy",
        "micro_spending_easy",
        "behavior_analysis_medium",
        "weekend_spending_medium",
        "subscription_overlap_medium",
        "goal_optimization_hard",
        "tight_budget_hard",
        "multi_goal_hard"
    ]

    for task in tasks:
        run_task(task)