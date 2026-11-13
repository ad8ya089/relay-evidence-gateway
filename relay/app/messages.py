"""Safe user-facing control-plane messages; no guessed provider reset times."""

MESSAGES = {
    "queue_full": "The research assistant is busy. Please try again shortly.",
    "projected_wait": (
        "I can't finish a verified answer within the available waiting time. "
        "Please try again later."
    ),
    "daily_budget_exhausted": (
        "I can't provide a verified answer right now because the research assistant's daily "
        "model-service budget has been exhausted. I haven't generated or sent an unverified "
        "response. Please try again after the provider's quota resets."
    ),
    "breaker_open": "The model service is temporarily unavailable. Please try again later.",
    "provider_429_minute": (
        "The model service's short-term capacity is unavailable. Please try again later."
    ),
    "provider_429_day": (
        "The model service reports a daily quota limit. Please try again after its quota resets."
    ),
    "provider_429_unknown": (
        "The model service reports a quota limit whose recovery time is unknown. "
        "Please try again later."
    ),
    "provider_unavailable": (
        "I couldn't obtain a verified answer from the model service. Please try again later."
    ),
    "deadline_infeasible": (
        "There isn't enough time to finish a verified answer. Please try again later."
    ),
    "request_in_progress": (
        "This request identifier is already in use. Please use a unique identifier."
    ),
    "invalid_setting": "Choose retrieval setting A, B or C.",
}


def message(reason: str) -> str:
    return MESSAGES.get(reason, MESSAGES["provider_unavailable"])
