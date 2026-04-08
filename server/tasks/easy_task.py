"""
EASY TASK — Email Categorizer
==============================
Agent receives 10 emails and must classify each as: urgent | normal | spam

Ground truth is fixed (seed=42 determinism — no randomness).
One email (e5) is deliberately ambiguous: both "urgent" and "normal" earn full credit.

Expected trajectory (2 steps):
  1. select_task  task_id=email_categorizer
  2. submit_answer content=<JSON map>
"""

import json
from typing import Any, Dict

# Fixed email dataset — never changes
EMAILS: Dict[str, Dict[str, str]] = {
    "e1":  {"subject": "URGENT: Production server is DOWN — 500 users affected",
            "from_": "ops@company.com",        "gt": "urgent"},
    "e2":  {"subject": "Team lunch tomorrow at noon?",
            "from_": "sarah@company.com",      "gt": "normal"},
    "e3":  {"subject": "Congratulations! You've WON a free iPhone!!!",
            "from_": "noreply@prizewin.net",   "gt": "spam"},
    "e4":  {"subject": "Q3 financial report due by Friday — please review",
            "from_": "manager@company.com",    "gt": "urgent"},
    "e5":  {"subject": "Meeting with the client has been rescheduled",
            "from_": "hr@company.com",         "gt": "normal"},   # ambiguous — urgent also accepted
    "e6":  {"subject": "CLAIM your $500 gift card NOW — limited time offer",
            "from_": "offers@deals.biz",       "gt": "spam"},
    "e7":  {"subject": "Payment gateway failure — transactions declining",
            "from_": "payments@company.com",   "gt": "urgent"},
    "e8":  {"subject": "How do I export my account data to CSV?",
            "from_": "user4821@gmail.com",     "gt": "normal"},
    "e9":  {"subject": "Security alert: unusual login detected on your account",
            "from_": "security@company.com",   "gt": "urgent"},
    "e10": {"subject": "Friendly reminder: submit your weekly status update",
            "from_": "no-reply@company.com",   "gt": "normal"},
}

AMBIGUOUS_IDS = {"e5"}          # both urgent and normal accepted
VALID_CATEGORIES = {"urgent", "normal", "spam"}

TASK_DESCRIPTION = (
    "TASK: Email Categorizer (easy)\n\n"
    "You have received 10 emails. Classify each one as exactly one of: urgent, normal, or spam.\n\n"
    "Emails:\n"
    + "\n".join(
        f"  {eid}: [{info['from_']}] {info['subject']}"
        for eid, info in EMAILS.items()
    )
    + "\n\n"
    "Submit a JSON object mapping each email ID to its category.\n"
    "Example: {\"e1\": \"urgent\", \"e2\": \"normal\", \"e3\": \"spam\", ...}\n"
    "You must classify all 10 emails."
)


class EmailCategorizerTask:
    task_id = "email_categorizer"
    difficulty = "easy"
    deadline_steps = 8

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
            "Hint: 'urgent' = affects users NOW or is a security risk. "
            "'spam' = promotional/phishing. "
            "'normal' = routine questions and updates. "
            "e5 is ambiguous — either urgent or normal is accepted."
        )

    def grade(self, submission: str, _state: Dict[str, Any]) -> float:
        """
        Score 0.0–1.0.
        - 0.0  invalid JSON or not a dict
        - per-email: 1.0 exact match, 0.5 ambiguous email with acceptable answer
        - final score = correct / total
        """
        try:
            parsed = json.loads(submission)
        except (json.JSONDecodeError, TypeError):
            return 0.0

        if not isinstance(parsed, dict):
            return 0.0

        correct = 0.0
        for eid, info in EMAILS.items():
            answer = str(parsed.get(eid, "")).strip().lower()
            expected = info["gt"]

            if answer == expected:
                correct += 1.0
            elif eid in AMBIGUOUS_IDS and answer in VALID_CATEGORIES:
                # Any valid category for the ambiguous email gets full credit
                correct += 1.0

        return round(correct / len(EMAILS), 4)
