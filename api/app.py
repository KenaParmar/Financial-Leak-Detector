"""


Endpoints
─────────
  GET  /        — health check (validator pings this + /reset)
  POST /reset   — start new episode, returns Observation
  POST /step    — send Action, receive StepResult
  GET  /state   — inspect raw task state (debug)
"""

from fastapi import FastAPI, HTTPException, Query
from typing import Optional

from env.financial_env import (
    FinancialLeakEnv,
    TaskNotFoundError,
    EnvironmentNotResetError,
)
from env.models import Action, Observation, StepResult

app = FastAPI(
    title="Financial Leak Detection Environment",
    description=(
        "OpenEnv-compliant API for evaluating LLM-based personal finance agents. "
        "Detects subscription leaks, duplicate charges, and spending pattern issues."
    ),
    version="1.0.0",
)

_env: Optional[FinancialLeakEnv] = None
_DEFAULT_TASK = "leak_detection_easy"


# ── Health check ──────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
def health():
   
    return {"status": "ok", "env": "financial_leak_env", "version": "1.0.0"}


# ── Reset ─────────────────────────────────────────────────────────────────

@app.post("/reset", response_model=Observation, tags=["env"])
def reset(task_id: str = Query(default=_DEFAULT_TASK)):
    
    global _env
    if _env is not None:
        _env.close()
    try:
        _env = FinancialLeakEnv(task_id)
        return _env.reset()
    except TaskNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"reset_failed: {exc}")


# ── Step ──────────────────────────────────────────────────────────────────

@app.post("/step", response_model=StepResult, tags=["env"])
def step(action: Optional[Action] = None):
    
    global _env
    if _env is None:
        _env = FinancialLeakEnv(_DEFAULT_TASK)
        _env.reset()
    if action is None:
        action = Action()
    try:
        obs, reward, done, info = _env.step(action)
        return StepResult(observation=obs, reward=reward, done=done, info=info)
    except EnvironmentNotResetError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"step_failed: {exc}")


# ── State ─────────────────────────────────────────────────────────────────

@app.get("/state", tags=["env"])
def state():
   
    if _env is None:
        return {"state": None, "hint": "Call /reset first."}
    try:
        return {"state": _env.state()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"state_failed: {exc}")