"""
MEDIUM TASK — Bug Prioritizer
==============================
Agent receives 5 bug reports and must rank them highest → lowest priority.

Graded by Kendall's tau: measures pairwise ordering agreement vs ground truth.
Score range: 0.0 (completely reversed) → 1.0 (perfect order).

Ground truth: B1 > B5 > B3 > B4 > B2
Reasoning:
  B1  critical severity, 500 users  → highest
  B5  critical severity, 200 users  → second (same severity, fewer users)
  B3  high severity,    100 users   → third
  B4  medium severity,   50 users   → fourth
  B2  low severity,       2 users   → lowest

Expected trajectory (2–3 steps):
  1. select_task  task_id=bug_prioritizer
  2. submit_answer content=["B1","B5","B3","B4","B2"]
"""

import json
from typing import Any, Dict, List

BUGS: List[Dict[str, Any]] = [
    {"id": "B1", "title": "Authentication service completely down",
     "severity": "critical", "users_affected": 500, "component": "auth",
     "reported_at": "2024-01-15T08:00:00Z"},
    {"id": "B2", "title": "Dark mode toggle misaligned on settings page",
     "severity": "low",      "users_affected": 2,   "component": "ui",
     "reported_at": "2024-01-15T11:30:00Z"},
    {"id": "B3", "title": "Payment processing fails for non-US cards",
     "severity": "high",     "users_affected": 100, "component": "payments",
     "reported_at": "2024-01-15T09:15:00Z"},
    {"id": "B4", "title": "API rate-limit headers missing from responses",
     "severity": "medium",   "users_affected": 50,  "component": "api",
     "reported_at": "2024-01-15T10:00:00Z"},
    {"id": "B5", "title": "Database connection pool exhaustion under load",
     "severity": "critical", "users_affected": 200, "component": "database",
     "reported_at": "2024-01-15T08:45:00Z"},
]

GROUND_TRUTH_ORDER: List[str] = ["B1", "B5", "B3", "B4", "B2"]
VALID_IDS = {b["id"] for b in BUGS}

TASK_DESCRIPTION = (
    "TASK: Bug Prioritizer (medium)\n\n"
    "Rank the following 5 bugs from HIGHEST to LOWEST priority.\n"
    "Consider: severity (critical > high > medium > low) and number of users affected.\n\n"
    "Bugs:\n"
    + "\n".join(
        f"  {b['id']}: [{b['severity'].upper()}] {b['title']} "
        f"(affects {b['users_affected']} users, component: {b['component']})"
        for b in BUGS
    )
    + "\n\n"
    "Submit a JSON array of bug IDs in priority order (highest first).\n"
    "Example: [\"B1\", \"B3\", \"B5\", \"B2\", \"B4\"]\n"
    "All 5 bug IDs must appear exactly once."
)


class BugPrioritizerTask:
    task_id = "bug_prioritizer"
    difficulty = "medium"
    deadline_steps = 12

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
            "Hint: Sort primarily by severity (critical > high > medium > low). "
            "When severity is equal, prefer the bug affecting more users. "
            "Ground truth prioritises: auth failure > DB exhaustion > payment failure > API issue > UI glitch."
        )

    def grade(self, submission: str, _state: Dict[str, Any]) -> float:
        """
        Kendall's tau correlation between submitted order and ground truth.
        Returns 0.0–1.0.
        - 0.0 if can't parse or wrong IDs
        - 0.1 if all IDs present but none of the logic applies
        - Tau formula: concordant_pairs / total_pairs
        """
        try:
            order: List[str] = json.loads(submission)
        except (json.JSONDecodeError, TypeError):
            return 0.0

        if not isinstance(order, list):
            return 0.0

        # Must contain exactly the right IDs
        if set(order) != VALID_IDS:
            return 0.0

        n = len(order)
        total_pairs = n * (n - 1) / 2
        gt_rank = {v: i for i, v in enumerate(GROUND_TRUTH_ORDER)}
        concordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                pred_before = order.index(order[i]) < order.index(order[j])
                true_before = gt_rank[order[i]] < gt_rank[order[j]]
                if pred_before == true_before:
                    concordant += 1

        tau = concordant / total_pairs          # raw tau in [0, 1]
        return round(tau, 4)
