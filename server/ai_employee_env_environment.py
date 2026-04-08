"""
AI Employee Work Simulator — Core Environment
=============================================
Implements the OpenEnv Environment interface.

Three tasks:
  email_categorizer  (easy)    deadline: 8 steps
  bug_prioritizer    (medium)  deadline: 12 steps
  sprint_planner     (hard)    deadline: 20 steps

Episode length: 25 steps max.
"""

import copy
from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AiEmployeeAction, AiEmployeeObservation
except ImportError:
    from models import AiEmployeeAction, AiEmployeeObservation

try:
    from .tasks import EmailCategorizerTask, BugPrioritizerTask, SprintPlannerTask
except ImportError:
    from server.tasks import EmailCategorizerTask, BugPrioritizerTask, SprintPlannerTask

# ── Constants ──────────────────────────────────────────────────────────────
MAX_STEPS = 25
MAX_ATTEMPTS = 3          # attempts before a task auto-fails
FIRST_TRY_BONUS = 0.15   # bonus for scoring ≥ 0.9 on first attempt
PARTIAL_FACTOR = 0.40    # scaling for sub-threshold submissions

_TASK_REGISTRY = {
    EmailCategorizerTask.task_id: EmailCategorizerTask,
    BugPrioritizerTask.task_id:   BugPrioritizerTask,
    SprintPlannerTask.task_id:    SprintPlannerTask,
}


class AiEmployeeEnvironment(Environment):
    """AI Employee Work Simulator environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._tasks = {tid: cls() for tid, cls in _TASK_REGISTRY.items()}
        self._episode_id: str = str(uuid4())
        # Auto-init so HTTP /step doesn't crash on fresh instances
        self._state: Dict[str, Any] = self._initial_state()

    # ── Internal state factory ─────────────────────────────────────────────

    def _initial_state(self) -> Dict[str, Any]:
        return {
            "step": 0,
            "current_task_id": None,
            "tasks": {
                tid: copy.deepcopy(task.initial_state())
                for tid, task in self._tasks.items()
            },
            "history": [],
            "total_reward": 0.0,
        }

    # ── Public interface ───────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> AiEmployeeObservation:
        self._episode_id = episode_id or str(uuid4())
        self._state = self._initial_state()
        return self._build_obs("Episode started. Use select_task to begin.")

    def step(self, action: AiEmployeeAction, **kwargs) -> AiEmployeeObservation:

        self._state["step"] += 1
        reward = 0.0
        msg = ""

        try:
            at = action.action_type

            if at == "select_task":
                reward, msg = self._handle_select(action)
            elif at == "submit_answer":
                reward, msg = self._handle_submit(action)
            elif at == "request_info":
                reward, msg = self._handle_info()
            elif at == "skip_task":
                reward, msg = self._handle_skip()
            else:
                reward, msg = -0.05, f"Unknown action_type: {at}"

        except Exception as exc:
            reward, msg = -0.05, f"Action error: {exc}"

        self._state["total_reward"] += reward
        self._state["history"].append({
            "step": self._state["step"],
            "action": action.action_type,
            "task_id": action.task_id,
            "reward": round(reward, 4),
        })

        done = self._is_done()
        obs = self._build_obs(msg)
        obs.done = done
        obs.reward = round(reward, 4)
        return obs

    @property
    def state(self) -> State:
        return State(
            episode_id=self._episode_id,
            step_count=self._state["step"],
            tasks=self._state["tasks"],
            current_task_id=self._state["current_task_id"],
            total_reward=round(self._state["total_reward"], 4),
            history=self._state["history"],
        )

    # ── Action handlers ────────────────────────────────────────────────────

    def _handle_select(self, action: AiEmployeeAction):
        tid = action.task_id
        if tid not in self._tasks:
            return -0.05, f"Unknown task_id '{tid}'. Valid: {list(self._tasks.keys())}"

        ts = self._state["tasks"][tid]
        if ts["status"] in ("completed", "failed", "skipped"):
            return -0.05, f"Task '{tid}' is already {ts['status']}. Pick another."

        ts["status"] = "active"
        self._state["current_task_id"] = tid
        return 0.0, f"Task '{tid}' selected. Read the description and submit your answer."

    def _handle_submit(self, action: AiEmployeeAction):
        tid = self._state["current_task_id"]
        if tid is None:
            return -0.05, "No task selected. Use select_task first."

        ts = self._state["tasks"][tid]
        if ts["status"] in ("completed", "failed", "skipped"):
            return -0.05, f"Task '{tid}' is already {ts['status']}."

        content = action.content or ""
        task = self._tasks[tid]
        score = task.grade(content, ts)

        ts["attempts"] += 1
        ts["score"] = max(ts["score"], score)   # best-of-N

        if score >= 0.9:
            ts["status"] = "completed"
            bonus = FIRST_TRY_BONUS if ts["attempts"] == 1 else 0.0
            reward = round(score + bonus, 4)
            msg = f"Task '{tid}' completed! Score: {score:.4f}."
            if bonus:
                msg += f" First-attempt bonus: +{bonus:.2f}."
        elif ts["attempts"] >= MAX_ATTEMPTS:
            ts["status"] = "failed"
            reward = round(score * PARTIAL_FACTOR, 4)
            msg = (
                f"Task '{tid}' failed — max attempts ({MAX_ATTEMPTS}) reached. "
                f"Best score: {ts['score']:.4f}."
            )
        else:
            remaining = MAX_ATTEMPTS - ts["attempts"]
            reward = round(score * PARTIAL_FACTOR, 4)
            msg = (
                f"Partial score: {score:.4f} (reward: {reward:.4f}). "
                f"{remaining} attempt(s) remaining."
            )

        return reward, msg

    def _handle_info(self):
        tid = self._state["current_task_id"]
        if tid is None:
            return -0.05, "No task selected. Use select_task before requesting info."
        hint = self._tasks[tid].get_hint()
        return -0.05, f"Hint for '{tid}': {hint}"

    def _handle_skip(self):
        tid = self._state["current_task_id"]
        if tid is None:
            return -0.05, "No task is active — nothing to skip."
        self._state["tasks"][tid]["status"] = "skipped"
        self._state["current_task_id"] = None
        return -0.40, f"Task '{tid}' skipped. Heavy penalty applied."

    # ── Helpers ────────────────────────────────────────────────────────────

    def _is_done(self) -> bool:
        if self._state["step"] >= MAX_STEPS:
            return True
        terminal = {"completed", "failed", "skipped"}
        return all(
            ts["status"] in terminal
            for ts in self._state["tasks"].values()
        )

    def _build_obs(self, msg: str) -> AiEmployeeObservation:
        s = self._state
        tasks_out = []
        for tid, task in self._tasks.items():
            ts = s["tasks"][tid]
            tasks_out.append({
                "task_id": tid,
                "difficulty": task.difficulty,
                "description": task.description,
                "status": ts["status"],
                "score": ts["score"],
                "attempts": ts["attempts"],
                "deadline_steps": ts["deadline_steps"],
            })

        return AiEmployeeObservation(
            step=s["step"],
            steps_remaining=MAX_STEPS - s["step"],
            tasks=tasks_out,
            current_task_id=s["current_task_id"],
            last_action_result=msg,
            total_reward=round(s["total_reward"], 4),
            done=False,
            reward=None,
        )
