"""Additional run-wide bounds; every dispatch still passes the durable provider guard."""

from pydantic import BaseModel, Field

from relay.providers.base import LLMRequest, ProviderError, ProviderFailure


class SessionBudget(BaseModel):
    token_cap: int = Field(gt=0)
    request_cap: int = Field(gt=0)
    estimated_tokens: int = 0
    calls: int = 0
    stopped: bool = False

    def reserve(self, request: LLMRequest) -> None:
        bound = request.token_reservation
        if (
            self.stopped
            or bound is None
            or self.calls >= self.request_cap
            or self.estimated_tokens + bound > self.token_cap
        ):
            raise ProviderError(
                ProviderFailure(
                    kind="budget_exceeded",
                    safe_context={"dispatched": False, "reason": "run-wide estimate/request cap"},
                )
            )
        self.estimated_tokens += bound
        self.calls += 1
