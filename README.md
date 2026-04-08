# Intelligent Financial Leak Detection Environment (OpenEnv)

## Problem

Most people don’t truly understand where their money goes.

* Hidden subscriptions silently drain money
* Impulse spending (late-night orders, weekend spikes)
* No visibility into behavioral patterns
* No actionable plan to achieve savings goals

Existing finance apps only **track expenses** — they don’t **analyze, predict, or guide decisions**.

---

## Our Solution

We built an **AI-powered Financial Decision Environment** using OpenEnv that:

* Detects financial leaks
* Understands user behavior
* Predicts future spending
* Guides users toward savings goals

This is not a static tool — it is a **multi-step learning environment** where an AI agent continuously improves financial decisions over time.

---

## Environment Overview

### Interaction Loop

1. Agent observes financial data
2. Agent takes actions (optimize spending)
3. Environment updates state
4. Reward + task score returned
5. Loop continues

---

## Observation Space

Each step provides structured financial data:

* **Transactions** (amount, category, time)
* **Subscriptions** (usage, cost)
* **Spending Summary**
* **User Goal**
* **Time progression (month)**

---

## Action Space

The agent can:

* Cancel unnecessary subscriptions
* Reduce spending in categories
* Generate behavioral insights

Example:

```json
{
  "cancel_subscriptions": ["Gym"],
  "reduce_categories": {"food": 0.3},
  "insights": ["Reduce late night spending"]
}
```

---

## Reward Function

The reward is **dense and meaningful**, capturing real-world financial reasoning:

* ✅ Reward for detecting unused subscriptions
* ✅ Reward for identifying behavior patterns
* ✅ Reward for reducing spending
* ❌ Penalty for removing useful services

Encourages **progressive improvement**, not just final success.

---

## Task Design (9 Tasks)

We designed **multi-difficulty tasks** to simulate real-world complexity:

### Easy (Basic Leak Detection)

* `leak_detection_easy`
* `duplicate_charge_easy`
* `micro_spending_easy`

### Medium (Behavior Understanding)

* `behavior_analysis_medium`
* `weekend_spending_medium`
* `subscription_overlap_medium`

### Hard (Optimization & Planning)

* `goal_optimization_hard`
* `tight_budget_hard`
* `multi_goal_hard`

---

## Graders (Strict Scoring)

Each task has a **deterministic grader**:

* Score range: **(0, 1)** (strictly enforced)
* Evaluates:

  * correctness of actions
  * quality of insights
  * optimization effectiveness

Ensures fair and consistent evaluation.

---

## Baseline Agent

We provide a **deterministic baseline agent**:

* Uses OpenAI client (as required)
* Falls back safely if API fails
* Produces reproducible results
* Logs in strict evaluation format

---

## Example Output

```
[START] task=leak_detection_easy env=financial_leak_env model=gpt-4o-mini
[STEP] step=1 action={...} reward=0.80 done=false error=api_error
[STEP] step=2 action={...} reward=0.80 done=false error=api_error
[END] success=true steps=5 score=0.990 rewards=0.80,0.80,0.80,0.80,0.80
```

---

## Running Locally

### 1️⃣ Build Docker Image

```bash
docker build -t financial-env .
```

### 2️⃣ Run Container

```bash
docker run -p 7860:7860 financial-env
```

### 3️⃣ Open API Docs

```
http://localhost:7860/docs
```

---

## Run Inference

```bash
python inference.py
```

---

## Deployment

Deployed using **Hugging Face Spaces (Docker)**

* Fully containerized
* OpenEnv compliant
* Ready for automated evaluation

---

## OpenEnv Compliance

* ✔ Typed Observation / Action models
* ✔ `step()`, `reset()`, `state()` implemented
* ✔ `openenv.yaml` defined
* ✔ 3+ tasks with graders
* ✔ Dense reward function
* ✔ Baseline inference script
* ✔ Dockerized + deployable

---

## Why This Project Stands Out

* Real-world financial intelligence (not a toy problem)
* Multi-step learning environment
* Behavior-aware decision making
* Explainable outputs (insights + plans)
* Strong reward shaping and grading system

---

## Future Improvements

* Personalized RL-based optimization
* Dynamic pricing suggestions
* Real-time transaction integration
* Multi-user collaborative insights

---

## Author

Built as part of an OpenEnv challenge to design realistic AI environments.

---
