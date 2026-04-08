"""
FastAPI application for the AI Employee Work Simulator.

Exposes the environment over HTTP + WebSocket endpoints via openenv-core's
create_app() factory. The framework handles:
  POST /reset   → env.reset()
  POST /step    → env.step(action)
  GET  /state   → env.state
  GET  /schema  → action/observation JSON schemas
  WS   /ws      → persistent WebSocket session

Usage (local):
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

Usage (Docker / HF Spaces):
    CMD in Dockerfile runs uvicorn on port 8000.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

from fastapi.middleware.cors import CORSMiddleware

try:
    from ..models import AiEmployeeAction, AiEmployeeObservation
    from .ai_employee_env_environment import AiEmployeeEnvironment
except ImportError:
    from models import AiEmployeeAction, AiEmployeeObservation
    from server.ai_employee_env_environment import AiEmployeeEnvironment

app = create_app(
    AiEmployeeEnvironment,
    AiEmployeeAction,
    AiEmployeeObservation,
    env_name="ai_employee_env",
    max_concurrent_envs=4,
)

# Allow browser requests from file:// and any origin (needed for the frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
