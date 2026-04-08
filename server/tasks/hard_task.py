"""
HARD TASK — Sprint Planner
===========================
Agent receives 8 work items with dependencies and hour estimates.
Must output a valid execution order that:
  1. Respects all dependency constraints (DAG — no dep before its parent)
  2. Stays within a 30-hour budget
  3. Minimises time-to-user-notification (T4 should appear as early as valid)

Scoring breakdown:
  40%  dependency correctness  (0 violations = 1.0, each violation reduces score)
  30%  budget compliance       (total ≤ 30h = 1.0, else penalised)
  30%  T4 earliness            (T4 at position 4/8 = 0.5, earlier = higher)

Total hours for all 8 items: 26h (within budget).
Critical path to T4: T1 → T2 → T4 (12h) or T1 → T3 → T4 (9h).
Optimal T4 position: index 3 out of 8 (after T1, T3, and either T2 or T5).

Expected trajectory (3–5 steps):
  1. select_task  task_id=sprint_planner
  2. [optional] request_info
  3. submit_answer content=["T1","T3","T5","T2","T4","T6","T7","T8"]
"""

import json
from typing import Any, Dict, List

WORK_ITEMS: Dict[str, Dict[str, Any]] = {
    "T1": {"name": "Investigate root cause of auth failure",
           "estimate_h": 3, "deps": []},
    "T2": {"name": "Fix database connection pool",
           "estimate_h": 5, "deps": ["T1"]},
    "T3": {"name": "Patch API gateway auth middleware",
           "estimate_h": 2, "deps": ["T1"]},
    "T4": {"name": "Notify affected users via email",
           "estimate_h": 2, "deps": ["T2", "T3"]},
    "T5": {"name": "Restore backup for corrupted records",
           "estimate_h": 6, "deps": []},
    "T6": {"name": "Write incident post-mortem draft",
           "estimate_h": 3, "deps": ["T5"]},
    "T7": {"name": "Deploy hotfix to production",
           "estimate_h": 4, "deps": ["T4", "T6"]},
    "T8": {"name": "Close incident ticket and archive logs",
           "estimate_h": 1, "deps": ["T7"]},
}

BUDGET_HOURS: int = 30
VALID_IDS = set(WORK_ITEMS.keys())
TOTAL_HOURS = sum(v["estimate_h"] for v in WORK_ITEMS.values())  # = 26

TASK_DESCRIPTION = (
    "TASK: Sprint Planner (hard)\n\n"
    "You are the incident manager. Plan the execution order for 8 work items.\n\n"
    "Requirements:\n"
    "  1. Respect ALL dependency constraints (a task's deps must come before it).\n"
    "  2. Total estimated hours must be ≤ 30h.\n"
    "  3. Minimise time to reach T4 ('Notify affected users') — get it done as early as valid.\n\n"
    "Work items:\n"
    + "\n".join(
        f"  {tid}: {info['name']} [{info['estimate_h']}h]"
        + (f" — depends on: {', '.join(info['deps'])}" if info["deps"] else " — no dependencies")
        for tid, info in WORK_ITEMS.items()
    )
    + f"\n\nTotal hours: {TOTAL_HOURS}h  |  Budget: {BUDGET_HOURS}h\n\n"
    "Submit a JSON array of all 8 task IDs in execution order.\n"
    "Example: [\"T1\", \"T5\", \"T3\", \"T2\", \"T4\", \"T6\", \"T7\", \"T8\"]"
)


class SprintPlannerTask:
    task_id = "sprint_planner"
    difficulty = "hard"
    deadline_steps = 20

    def initial_state(self) -> Dict[str, Any]:
        return {
            "status": "pending",
            "score": 0.0,
            "attempts": 0,
            "deadline_steps": self.deadline_steps,
        }

    @property
    def description(self) -> str:
        return TASK_DESCRIPTION

    def get_hint(self) -> str:
        return (
            "Hint: T1 must come first — everything depends on it. "
            "T2 and T3 both need T1. T4 needs T2 AND T3 — so T4 can only start after both. "
            "T5 is independent and can run in parallel with T1–T3. "
            "To get T4 early: T1 → T3 (2h) → T2 (5h) → T4, with T5 alongside. "
            "T7 needs both T4 and T6. T8 is the final step."
        )

    def grade(self, submission: str, _state: Dict[str, Any]) -> float:
        """
        Three-component score:
          dep_score    (40%) — 1.0 if zero dep violations
          budget_score (30%) — 1.0 if ≤ 30h, penalised above
          t4_score     (30%) — 1.0 if T4 is as early as valid (position 3 of 7 = index 3)
        """
        try:
            order: List[str] = json.loads(submission)
        except (json.JSONDecodeError, TypeError):
            return 0.0

        if not isinstance(order, list):
            return 0.0

        # All 8 tasks must be present exactly once
        if set(order) != VALID_IDS or len(order) != len(VALID_IDS):
            return 0.0

        # ── Dependency check (40%) ──────────────────────────────────────
        seen: set = set()
        violations = 0
        for tid in order:
            for dep in WORK_ITEMS[tid]["deps"]:
                if dep not in seen:
                    violations += 1
            seen.add(tid)

        dep_score = 0.40 * max(0.0, 1.0 - violations / max(1, len(order)))

        # ── Budget check (30%) ─────────────────────────────────────────
        total_h = sum(WORK_ITEMS[t]["estimate_h"] for t in order)
        if total_h <= BUDGET_HOURS:
            budget_score = 0.30
        else:
            over = total_h - BUDGET_HOURS
            budget_score = 0.30 * max(0.0, 1.0 - over / 10.0)

        # ── T4 earliness (30%) ─────────────────────────────────────────
        # Earliest valid position for T4 is index 3 (after T1, T2/T3, T3/T2).
        # Score = 1.0 at index 3, decreases linearly toward 0.0 at index 7.
        if "T4" in order:
            t4_idx = order.index("T4")
            earliest_possible = 3
            latest_possible = len(order) - 1
            span = latest_possible - earliest_possible
            t4_score = 0.30 * max(0.0, 1.0 - (t4_idx - earliest_possible) / span)
        else:
            t4_score = 0.0

        return round(dep_score + budget_score + t4_score, 4)
