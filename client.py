"""
AI Employee Work Simulator — Environment Client.

Wraps the openenv-core EnvClient to connect to the running server
via WebSocket. Used by inference.py.

Usage:
    # Connect to a running server
    async with AiEmployeeEnv(base_url="http://localhost:8000") as env:
        result = await env.reset()
        result = await env.step(AiEmployeeAction(action_type="select_task", task_id="email_categorizer"))

    # Start from Docker image (used by inference.py)
    env = AiEmployeeEnv.from_docker_image("ai_employee_env-env:latest")
"""

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AiEmployeeAction, AiEmployeeObservation


class AiEmployeeEnv(EnvClient[AiEmployeeAction, AiEmployeeObservation, State]):
    """WebSocket client for the AI Employee Work Simulator environment."""

    def _step_payload(self, action: AiEmployeeAction) -> Dict[str, Any]:
        """Serialize action to JSON payload for WebSocket step message."""
        payload: Dict[str, Any] = {
            "action_type": action.action_type,
            "metadata": action.metadata,
        }
        if action.task_id is not None:
            payload["task_id"] = action.task_id
        if action.content is not None:
            payload["content"] = action.content
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AiEmployeeObservation]:
        """Parse server WebSocket response into StepResult."""
        obs_data = payload.get("observation", payload)

        observation = AiEmployeeObservation(
            step=obs_data.get("step", 0),
            steps_remaining=obs_data.get("steps_remaining", 0),
            tasks=obs_data.get("tasks", []),
            current_task_id=obs_data.get("current_task_id"),
            last_action_result=obs_data.get("last_action_result", ""),
            total_reward=obs_data.get("total_reward", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            instructions=obs_data.get("instructions", ""),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        """Parse server state response."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
