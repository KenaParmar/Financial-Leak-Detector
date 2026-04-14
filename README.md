# 💸 Financial Leak Detection Environment

An **OpenEnv-compliant** RL environment for evaluating LLM-based personal finance agents.

An agent analyses a user's monthly transactions and subscriptions to detect money leaks, identify behavioural patterns, and optimise toward savings goals — with dense per-step rewards and 9 tasks across three difficulty levels.

---

## Project Structure

```
financial-leak-env/
├── api/
│   ├── __init__.py
│   └── app.py              # FastAPI server (OpenEnv REST Api)
├── env/
│   ├── __init__.py
│   ├── financial_env.py    # Core environment (reset/step/state/close)
│   ├── models.py           # Observation, Action, StepResult (Pydantic)
│   ├── reward.py           # Dense reward function
│   └── graders.py          # Per-task deterministic graders (9 tasks)
├── data/
│   └── tasks.json          # Task definitions with realistic financial data
├── inference.py            # Baseline LLM agent
├── openenv.yaml            # OpenEnv spec metadata
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Start the API server
uvicorn api.app:app --host 0.0.0.0 --port 7860

# Run the baseline agent
HF_TOKEN=your_token python inference.py
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Health check — returns `{"status": "ok"}` |
| `POST` | `/reset?task_id=<id>` | Start new episode, returns `Observation` |
| `POST` | `/step` | Send `Action`, receive `StepResult` |
| `GET`  | `/state` | Inspect raw task state (debug) |

---

## Tasks (9 total)

| Task ID | Difficulty | Objective |
|---------|-----------|-----------|
| `leak_detection_easy` | Easy | Cancel unused subscriptions |
| `duplicate_charge_easy` | Easy | Detect duplicate charges |
| `micro_spending_easy` | Easy | Identify micro-transaction drain |
| `behavior_analysis_medium` | Medium | Surface night/impulse spending patterns |
| `weekend_spending_medium` | Medium | Detect weekend vs weekday spending spikes |
| `subscription_overlap_medium` | Medium | Consolidate overlapping subscriptions |
| `goal_optimization_hard` | Hard | Hit £500 savings target across categories |
| `tight_budget_hard` | Hard | Save £200 without violating constraints |
| `multi_goal_hard` | Hard | Optimise for 3 simultaneous financial goals |

---

## Reward Design (Dense — signal every step)

| Component | Value | Condition |
|-----------|-------|-----------|
| Leak detection | +0.30 per sub | Cancel an unused subscription |
| Behaviour insight | +0.20 | Mention night/impulsive spending |
| Reduction effort | +0.20 (max) | Proportional to category cuts |
| Planning bonus | +0.10 | Forward savings plan in insights |
| Wrong cancellation | −0.20 per sub | Cancel a subscription that is used |
| Invalid fraction | −0.10 | reduction value > 1.0 |

---

## Action Space

```json
{
  "cancel_subscriptions": ["Gym", "MagazineX"],
  "reduce_categories":    {"food": 0.20, "entertainment": 0.25},
  "insights":             ["Late-night food spend is 40% above average — set a cut-off time."]
}
```

---

## Docker

```bash
docker build -t financial-leak-env .
docker run -p 7860:7860 financial-leak-env
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | Hugging Face / API key |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM endpoint |
| `MODEL_NAME` | No | `gpt-4.1-mini` | Model identifier |

---

## Example Output

```
[START] task=leak_detection_easy env=financial_leak_env model=gpt-4.1-mini
[STEP] step=1 action={"cancel_subscriptions":["Gym","MagazineX"],...} reward=0.80 done=false error=null
[STEP] step=2 action={...} reward=0.80 done=false error=null
[END] success=true steps=5 score=0.990 rewards=0.80,0.80,0.80,0.80,0.80
```
