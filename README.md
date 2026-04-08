---
title: AI Employee Work Simulator
emoji: 💼
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

# AI Employee Work Simulator

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where an AI agent acts as an employee completing real-world office tasks under deadline pressure. Built for the OpenEnv Round 1 Hackathon by team **WooshiWooshi**.

## Environment Description

The agent must complete **3 tasks** of increasing difficulty by selecting tasks, submitting formatted answers, and managing a limited step budget. Each task has a deterministic grader scoring 0.0–1.0.

**Validated:** 6/6 OpenEnv compliance checks pass. Live score: **3.45 / 3.45 max** (100%).

## Tasks

| Task ID | Difficulty | Description | Max Score |
|---|---|---|---|
| `email_categorizer` | Easy | Classify 10 emails as `urgent`, `normal`, or `spam` | 1.0 |
| `bug_prioritizer` | Medium | Rank 5 bug reports by priority (highest → lowest) using severity + users affected | 1.0 |
| `sprint_planner` | Hard | Order 8 work items respecting dependencies, within a 30h budget, minimising time to T4 | 1.0 |

## Action Space

```json
{
  "action_type": "select_task | submit_answer | request_info | skip_task",
  "task_id": "<task id — only for select_task>",
  "content": "<JSON answer string — only for submit_answer>"
}
```

| Action | Effect | Reward |
|---|---|---|
| `select_task` | Activate a task | 0.0 |
| `submit_answer` | Grade the answer | 0.0–1.15 (score + first-attempt bonus) |
| `request_info` | Get a hint | −0.05 |
| `skip_task` | Abandon task | −0.40 |

## Observation Space

```json
{
  "step": 3,
  "steps_remaining": 22,
  "tasks": [{"task_id": "...", "difficulty": "...", "status": "...", "score": 0.0, "attempts": 0}],
  "current_task_id": "email_categorizer",
  "last_action_result": "Task completed! Score: 1.0000.",
  "total_reward": 1.15,
  "instructions": "..."
}
```

## Reward Function

- **Dense**: partial credit on every submission (`score × 1.0`)
- **First-attempt bonus**: +0.15 for scoring ≥ 0.9 on first try
- **Step budget**: 25 steps per episode across all 3 tasks
- **Max possible**: ~3.45 across all 3 tasks

## Graders

| Task | Method |
|---|---|
| `email_categorizer` | Exact match per email (e5 accepts `urgent` or `normal`) |
| `bug_prioritizer` | Kendall's tau — partial credit for near-correct orderings |
| `sprint_planner` | 3-component score: dependency order (40%) + budget (30%) + T4 speed (30%) |

## Quick Start

```python
from ai_employee_env import AiEmployeeAction, AiEmployeeEnv

async with AiEmployeeEnv.from_env("piroBeastie/ai-employee-env") as env:
    await env.reset()
    await env.step(AiEmployeeAction(action_type="select_task", task_id="email_categorizer"))
    result = await env.step(AiEmployeeAction(
        action_type="submit_answer",
        content='{"e1":"urgent","e2":"normal","e3":"spam","e4":"urgent","e5":"normal","e6":"spam","e7":"urgent","e8":"normal","e9":"urgent","e10":"normal"}'
    ))
    print(result.reward)  # 1.15 (score 1.0 + first-attempt bonus 0.15)
```

## Local Setup

### pip
```bash
pip install openenv-core fastapi uvicorn pydantic
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
docker build -t ai-employee-env -f server/Dockerfile .
docker run -p 8000:8000 ai-employee-env
```

## Test & Validate

```bash
curl http://localhost:8000/health
openenv validate --url http://localhost:8000
```

## Run Inference

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Live Space

`https://huggingface.co/spaces/piroBeastie/ai-employee-env`
