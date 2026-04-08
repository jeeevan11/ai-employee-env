---
title: AI Employee Work Simulator
emoji: 💼
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
---


TEST

# AI Employee Work Simulator

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment where an agent acts as an AI employee completing real-world office tasks under deadline pressure.

## Environment Description

The agent must complete **3 tasks** of increasing difficulty by selecting tasks, submitting formatted answers, and managing limited steps. Each task has a deterministic grader scoring 0.0–1.0.

## Tasks

| Task ID | Difficulty | Description | Max Score |
|---|---|---|---|
| `email_categorizer` | Easy | Classify 10 emails as `urgent`, `normal`, or `spam` | 1.0 |
| `bug_prioritizer` | Medium | Rank 5 bug reports by priority (highest → lowest) | 1.0 |
| `sprint_planner` | Hard | Order 8 work items respecting dependencies, within budget | 1.0 |

## Action Space

```json
{
  "action_type": "select_task | submit_answer | request_info | skip_task",
  "task_id": "<task id — only for select_task>",
  "content": "<JSON answer — only for submit_answer>"
}
```

| Action | Effect | Reward |
|---|---|---|
| `select_task` | Activate a task | 0.0 |
| `submit_answer` | Grade the answer | 0.0–1.15 (score + first-attempt bonus) |
| `request_info` | Get a hint | -0.05 |
| `skip_task` | Abandon task | -0.40 |

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

- **Dense**: partial credit on every submission (`score × 0.4`)
- **First-attempt bonus**: +0.15 for scoring ≥ 0.9 on first try
- **Max possible**: ~3.45 across all 3 tasks

## Setup

### Local (uv — recommended)

```bash
cd ai_employee_env
uv run server
```

### Local (pip)

```bash
pip install openenv-core fastapi uvicorn pydantic
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t ai-employee-env -f server/Dockerfile .
docker run -p 8000:8000 ai-employee-env
```

## Test Endpoints

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
curl http://localhost:8000/state
openenv validate --url http://localhost:8000
```

## Run Inference

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Deploy to Hugging Face Spaces

```bash
huggingface-cli login
openenv push --repo-id your-username/ai-employee-env
```

## Validate

```bash
openenv validate                              # local file check
openenv validate --url http://localhost:8000  # live server check
```
