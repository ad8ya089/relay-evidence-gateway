import pytest

from relay.app.session import SessionBudget
from relay.providers.base import LLMRequest, Message, ProviderError


def request(bound: int) -> LLMRequest:
    return LLMRequest(
        messages=[Message(role="user", content="FAKE")],
        max_completion_tokens=10,
        token_reservation=bound,
    )


def test_estimated_bounds_do_not_refund_after_small_actual_usage() -> None:
    budget = SessionBudget(token_cap=5000, request_cap=4)
    budget.reserve(request(3200))
    budget.reserve(request(1500))
    with pytest.raises(ProviderError):
        budget.reserve(request(301))
    assert budget.calls == 2 and budget.estimated_tokens == 4700


def test_stop_after_429_or_call_limit() -> None:
    budget = SessionBudget(token_cap=5000, request_cap=1)
    budget.reserve(request(100))
    with pytest.raises(ProviderError):
        budget.reserve(request(100))
    budget = SessionBudget(token_cap=5000, request_cap=4, stopped=True)
    with pytest.raises(ProviderError):
        budget.reserve(request(100))
