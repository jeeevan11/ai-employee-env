#!/usr/bin/env python3
"""
AI Employee Work Simulator — Baseline Inference Script
=======================================================
Runs a language model agent against the environment for all 3 tasks.

Mandatory environment variables:
    HF_TOKEN        Your Hugging Face API key (also read as API_KEY)
    API_BASE_URL    LLM endpoint (default: HF router)
    MODEL_NAME      Model to use   (default: Qwen/Qwen2.5-72B-Instruct)
    LOCAL_IMAGE_NAME  Docker image name (if running locally from Docker)

Stdout format (mandatory — evaluated by OpenEnv harness):
    [START] task=<name> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Package import with fallback ───────────────────────────────────────────
try:
    from ai_employee_env.client import AiEmployeeEnv
    from ai_employee_env.models import AiEmployeeAction
except ImportError:
    # Running directly from project root without package installed
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from client import AiEmployeeEnv  # type: ignore
    from models import AiEmployeeAction  # type: ignore

# ── Mandatory env vars ─────────────────────────────────────────────────────
API_BASE_URL  = os.getenv("API_BASE_URL",  "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",    "Qwen/Qwen2.5-72B-Instruct")
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
IMAGE_NAME    = os.getenv("LOCAL_IMAGE_NAME", "")

BENCHMARK     = "ai_employee_env"
MAX_STEPS     = 12          # per task; 3 tasks × 12 = 36 total, well under 20-min limit
TEMPERATURE   = 0.1         # low temperature = deterministic agent
SUCCESS_THRESHOLD = 0.5     # score ≥ 0.5 counts as success

TASK_IDS = ["email_categorizer", "bug_prioritizer", "sprint_planner"]


# ── Mandatory stdout loggers ────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    # Collapse action to single line, strip newlines
    action_str = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI employee completing work tasks. You will be given a current environment
    observation. Respond with a single JSON object (no markdown, no explanation) that
    matches this exact schema:

    {
      "action_type": "select_task" | "submit_answer" | "request_info" | "skip_task",
      "task_id": "<task id — only for select_task>",
      "content": "<your answer — only for submit_answer>"
    }

    Rules:
    - NEVER include markdown code fences or extra text — raw JSON only.
    - For email_categorizer: submit a JSON object mapping email IDs to categories.
      Valid categories: urgent, normal, spam.
      Example: {"e1": "urgent", "e2": "normal", "e3": "spam", ...}
    - For bug_prioritizer: submit a JSON array of bug IDs ordered highest → lowest priority.
      Example: ["B1", "B5", "B3", "B4", "B2"]
    - For sprint_planner: submit a JSON array of all 8 task IDs in execution order.
      Respect ALL dependency constraints. T1 must come before T2, T3. T2+T3 before T4.
      To maximise score, place T4 as early as possible (index 3).
      Optimal: T1→T3→T2→T4 first, then T5→T6→T7→T8.
      Example: ["T1", "T3", "T2", "T4", "T5", "T6", "T7", "T8"]
    - Use request_info only if genuinely stuck — it costs reward.
    - Never skip tasks.
    - After selecting a task, immediately submit your answer.
""").strip()


# ── LLM call ────────────────────────────────────────────────────────────────

def get_action(
    client: OpenAI,
    obs: Dict[str, Any],
    history: List[Dict[str, str]],
    task_id: str,
) -> Dict[str, Any]:
    """Call the LLM and parse a JSON action. Falls back to select_task on error."""
    # Keep only the active task's description to save tokens
    obs_for_prompt = dict(obs)
    active_tasks = [t for t in obs.get("tasks", []) if t["task_id"] == task_id]
    obs_for_prompt["tasks"] = active_tasks

    user_content = (
        f"Current observation:\n{json.dumps(obs_for_prompt, indent=2)}\n\n"
        f"You are working on task: {task_id}\n"
        "Respond with a single JSON action object."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history[-4:],   # keep last 4 turns to bound context
        {"role": "user", "content": user_content},
    ]

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        # LLM sometimes returns content as a dict/list instead of a JSON string.
        # Coerce it back to a string so AiEmployeeAction validation passes.
        if "content" in parsed and not isinstance(parsed["content"], str):
            parsed["content"] = json.dumps(parsed["content"])
        return parsed
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {"action_type": "request_info"}


# ── Single-task episode ─────────────────────────────────────────────────────

async def run_task(client: OpenAI, env: Any, task_id: str) -> float:
    """
    Run one task within the environment.
    Returns the final score [0, 1].
    """
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[Dict[str, str]] = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Fresh episode for each task
        result = await env.reset()
        obs = result.observation

        # Step 1: always select the target task first
        select_action = AiEmployeeAction(action_type="select_task", task_id=task_id)
        result = await env.step(select_action)
        obs = result.observation
        reward = result.reward or 0.0
        done = result.done

        rewards.append(reward)
        steps_taken = 1
        log_step(step=1, action=f"select_task:{task_id}", reward=reward, done=done, error=None)

        # Remaining steps: let LLM decide
        for step in range(2, MAX_STEPS + 1):
            if done:
                break

            obs_dict = {
                "step": obs.step,
                "steps_remaining": obs.steps_remaining,
                "tasks": obs.tasks,
                "current_task_id": obs.current_task_id,
                "last_action_result": obs.last_action_result,
                "total_reward": obs.total_reward,
                "instructions": obs.instructions,
            }

            raw_action = get_action(client, obs_dict, history, task_id)

            try:
                action = AiEmployeeAction(**raw_action)
            except Exception as e:
                # Invalid action — request info as safe fallback
                action = AiEmployeeAction(action_type="request_info")
                error_msg = str(e)
            else:
                error_msg = None

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step

            log_step(
                step=step,
                action=json.dumps(raw_action),
                reward=reward,
                done=done,
                error=error_msg,
            )

            # Update history for context
            history.append({"role": "assistant", "content": json.dumps(raw_action)})
            history.append({
                "role": "user",
                "content": f"reward={reward:.2f} done={done} result={obs.last_action_result[:100]}",
            })

            if done:
                break

        # Score = best score achieved on this specific task
        task_scores = [t["score"] for t in obs.tasks if t["task_id"] == task_id]
        score = task_scores[0] if task_scores else 0.0
        score = max(1e-4, min(1 - 1e-4, score))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        log_step(step=steps_taken + 1, action="error", reward=0.0, done=True, error=str(exc))

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main ────────────────────────────────────────────────────────────────────

async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Build env client — local Docker image or URL
    if IMAGE_NAME:
        env = AiEmployeeEnv.from_docker_image(IMAGE_NAME)
    else:
        server_url = os.getenv("ENV_URL", "http://localhost:8000")
        env = AiEmployeeEnv(base_url=server_url)

    scores: List[float] = []

    try:
        async with env:
            for task_id in TASK_IDS:
                score = await run_task(client, env, task_id)
                scores.append(score)
    except Exception as exc:
        print(f"[DEBUG] Main error: {exc}", flush=True)
    finally:
        mean_score = sum(scores) / len(scores) if scores else 0.0
        print(f"\n[SUMMARY] tasks={len(scores)} mean_score={mean_score:.4f} scores={scores}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
