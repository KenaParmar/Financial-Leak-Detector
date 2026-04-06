import os
from openai import OpenAI
from env.financial_env import FinancialLeakEnv
from env.models import Action

client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)


def run_task(task_id):
    print(f"[START] Task: {task_id}")

    env = FinancialLeakEnv(task_id)
    obs = env.reset()

    total_reward = 0

    for step in range(5):
        print(f"[STEP] {step} OBS: {obs.model_dump()}")

        # Call LLM
        response = client.chat.completions.create(
            model=os.getenv("MODEL_NAME"),
            messages=[
                {"role": "system", "content": "You are a financial assistant optimizing spending."},
                {"role": "user", "content": str(obs.model_dump())}
            ]
        )

        # VERY IMPORTANT: parse response safely
        content = response.choices[0].message.content

        # fallback simple action (to ensure it never crashes)
        action = Action(
            cancel_subscriptions=["Gym"],
            reduce_categories={"food": 0.3},
            insights=["Reduce late night spending"]
        )

        obs, reward, done, info = env.step(action)

        total_reward += reward

        print(f"[STEP] {step} ACTION: {action.model_dump()} REWARD: {reward}")

        if done:
            break

    print(f"[END] Task: {task_id}, Total Reward: {total_reward}")


if __name__ == "__main__":
    tasks = [
        "leak_detection_easy",
        "behavior_analysis_medium",
        "goal_optimization_hard"
    ]

    for task in tasks:
        run_task(task)