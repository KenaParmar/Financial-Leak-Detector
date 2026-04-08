from fastapi import FastAPI
from typing import Optional
from env.financial_env import FinancialLeakEnv
from env.models import Action

app = FastAPI()

# Global environment instance
env_instance = None


@app.get("/")
def home():
    return {"message": "Financial Leak Detection Environment is running"}


# ✅ RESET ENDPOINT (SAFE)
@app.post("/reset")
def reset(task_id: Optional[str] = None):
    global env_instance

    # default task if not provided
    if task_id is None:
        task_id = "leak_detection_easy"

    try:
        env_instance = FinancialLeakEnv(task_id)
        obs = env_instance.reset()
        return obs

    except Exception as e:
        return {
            "error": "reset_failed",
            "message": str(e)
        }


# ✅ STEP ENDPOINT (SAFE)
@app.post("/step")
def step(action: Optional[Action] = None):
    global env_instance

    # if reset not called yet → auto reset
    if env_instance is None:
        env_instance = FinancialLeakEnv("leak_detection_easy")
        env_instance.reset()

    # default action if none provided
    if action is None:
        action = Action(
            cancel_subscriptions=[],
            reduce_categories={},
            insights=[]
        )

    try:
        obs, reward, done, info = env_instance.step(action)

        return {
            "observation": obs,
            "reward": reward,
            "done": done,
            "info": info
        }

    except Exception as e:
        return {
            "error": "step_failed",
            "message": str(e)
        }


# ✅ STATE ENDPOINT (OPTIONAL BUT GOOD)
@app.get("/state")
def state():
    global env_instance

    if env_instance is None:
        return {"state": None}

    try:
        return env_instance.state()
    except Exception as e:
        return {
            "error": "state_failed",
            "message": str(e)
        }