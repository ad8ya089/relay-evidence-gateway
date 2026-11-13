"""FAKE scripted provider; real protocol integration uses P2 against P3 separately."""

import json
from collections.abc import Callable

from relay.providers.base import DeltaCallback, HeaderSnapshot, LLMRequest, LLMResult, Usage


class ScriptedProvider:
    def __init__(self, responses: list[str | Exception | Callable[[LLMRequest], str]]) -> None:
        self.responses = responses
        self.requests: list[LLMRequest] = []

    async def complete(
        self, request: LLMRequest, *, on_delta: DeltaCallback | None = None
    ) -> LLMResult:
        self.requests.append(request)
        response = self.responses[len(self.requests) - 1]
        if isinstance(response, Exception):
            raise response
        text = response(request) if callable(response) else response
        return LLMResult(
            text=text,
            model="FAKE",
            status_code=200,
            usage=Usage(tokens_in=10, tokens_out=5, tokens_cached=2, total_tokens=15),
            usage_complete=True,
            header_snapshot=HeaderSnapshot(observed_ns=0),
            total_ms=0,
        )


def verdicts(*values: str) -> str:
    return json.dumps(
        {"verdicts": [{"i": i, "v": v, "why": "FAKE verdict"} for i, v in enumerate(values)]}
    )
