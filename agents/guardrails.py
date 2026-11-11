"""Response validation and safety guardrails for the support workflow."""

from typing import Any, Dict, List


_UNSAFE_MARKERS = (
    "ignore previous instructions",
    "system prompt",
    "developer message",
    "<|system|>",
)


def validate_response(query: str, response: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a specialist response and attach transparent guardrail metadata."""
    checked = dict(response)
    flags: List[str] = []
    answer = str(checked.get("answer", "")).strip()

    if not answer:
        flags.append("empty_answer")
    if not checked.get("agent"):
        flags.append("missing_agent")
    if any(marker in answer.lower() for marker in _UNSAFE_MARKERS):
        flags.append("prompt_content_detected")

    if flags:
        checked["answer"] = (
            "I could not safely complete that request automatically. "
            "Your query has been sent to human support."
        )
        checked["escalated"] = True
        checked["should_escalate"] = True

    checked["guardrail_passed"] = not flags
    checked["guardrail_flags"] = flags
    checked["query"] = query
    return checked


# Descriptive alias for callers that prefer an imperative name.
check_response = validate_response
