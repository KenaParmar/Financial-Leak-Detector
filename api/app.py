from fastapi import FastAPI
from env.financial_env import FinancialLeakEnv
from env.models import Action
from env.models import Observation

app = FastAPI()

env = None

@app.post("/reset", response_model=Observation)
def reset(task_id: str):
    global env

    task_id = task_id.strip().replace('"', '')
    env = FinancialLeakEnv(task_id)

    return env.reset()

from typing import Dict, Any

@app.post("/step")
def step(action: Action) -> Dict[str, Any]:
    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state()