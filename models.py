"""
Data models for the AI Employee Work Simulator environment.

Three tasks:
  - email_categorizer  (easy)   classify 10 emails as urgent/normal/spam
  - bug_prioritizer    (medium) rank 5 bugs by priority
  - sprint_planner     (hard)   order 8 work items respecting dependencies
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field

#Testing
# ── Action ─────────────────────────────────────────────────────────────────

class AiEmployeeAction(Action):
    """
    Single action type for the AI Employee environment.

    action_type options:
        select_task    - choose a task to work on (requires task_id)
        submit_answer  - submit answer for current task (requires content)
        request_info   - ask for a hint about the current task (costs -0.05)
        skip_task      - abandon the current task (-0.40 penalty)
    """

    action_type: Literal["select_task", "submit_answer", "request_info", "skip_task"] = Field(
        ..., description="Type of action to perform"
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Task to select. Required for select_task. One of: email_categorizer, bug_prioritizer, sprint_planner",
    )
    content: Optional[str] = Field(
        default=None,
        description="Answer content for submit_answer actions. Must be valid JSON string.",
    )


# ── Task summary (embedded in Observation — NO ground truth) ───────────────

# ── Observation ────────────────────────────────────────────────────────────

class AiEmployeeObservation(Observation):
    """
    Observation returned after every reset() and step().

    Contains:
    - current step count and remaining steps
    - summary of all 3 tasks (id, description, status, score, attempts)
    - which task is currently active
    - feedback from the last action
    - cumulative reward so far
    """

    step: int = Field(default=0, description="Current step number")
    steps_remaining: int = Field(default=25, description="Steps left in episode")

    tasks: List[Dict] = Field(
        default_factory=list,
        description=(
            "List of task summaries. Each contains: "
            "task_id, difficulty, description, status (pending/active/completed/failed/skipped), "
            "score (best so far), attempts."
        ),
    )

    current_task_id: Optional[str] = Field(
        default=None,
        description="ID of the currently active task, or null if none selected.",
    )

    last_action_result: str = Field(
        default="",
        description="Human-readable result/feedback from the last action.",
    )

    total_reward: float = Field(
        default=0.0,
        description="Cumulative reward collected so far this episode.",
    )

    instructions: str = Field(
        default=(
            "You are an AI employee. Complete all 3 tasks to maximise your score.\n"
            "Actions: select_task (pick a task), submit_answer (JSON answer), "
            "request_info (get a hint, costs -0.05), skip_task (abandon, costs -0.40).\n"
            "Available task IDs: email_categorizer, bug_prioritizer, sprint_planner."
        ),
        description="Standing instructions for the agent.",
    )
