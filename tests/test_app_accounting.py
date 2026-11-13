import json
from pathlib import Path

import httpx
import pytest

from relay.app.models import State
from relay.app.orchestrator import Orchestrator
from relay.app.settings import pipeline_settings
from relay.control.ledger import Ledger
from relay.providers.base import LLMRequest, LLMResult
from relay.providers.groq import GroqCompatibleProvider
from relay.tools.executor import ToolExecutor
from tests.fixtures.app_provider import verdicts
from tests.fixtures.backends import fixture_engine
from tests.standin_helpers import clock
from tests.test_app_validation import example


@pytest.mark.parametrize(
    "mode,state,calls",
    [
        ("missing", State.RELEASED, 2),
        ("server", State.FAILED, 1),
        ("draft_schema", State.RELEASED, 3),
        ("verify_schema", State.ABSTAINED, 4),
        ("ledger_mismatch", State.FAILED, 1),
    ],
)
async def test_actual_p2_ledger_is_the_usage_source_for_all_attempts(
    tmp_path: Path, mode: str, state: State, calls: int
) -> None:
    draft, _ = await example()
    sent = []
    usage = {
        "prompt_tokens": 23,
        "completion_tokens": 7,
        "total_tokens": 30,
        "prompt_tokens_details": {"cached_tokens": 3},
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"]["type"] == "json_schema"
        sent.append(payload)
        if mode == "server":
            return httpx.Response(500, json={"error": {"message": "FAKE server"}, "usage": usage})
        verify = "RELAY_VERIFY_V1" in payload["messages"][0]["content"]
        content = verdicts("supported") if verify else draft.model_dump_json()
        if mode == "draft_schema" and len(sent) == 1:
            content = "FAKE malformed JSON"
        if mode == "verify_schema" and verify:
            content = "{}"
        body = {"choices": [{"message": {"content": content}}]}
        if not (mode == "missing" and len(sent) == 1):
            body["usage"] = usage
        return httpx.Response(200, json=body)

    c = clock()
    config = pipeline_settings().model_copy(update={"draft_stream": False, "verify_stream": False})
    with Ledger(tmp_path / "calls.sqlite3", clock=c) as ledger:
        async with GroqCompatibleProvider(
            "http://127.0.0.1/openai/v1",
            None,
            "FAKE-model",
            clock=c,
            ledger=ledger,
            transport=httpx.MockTransport(handler),
        ) as actual:

            class Mismatch:
                async def complete(self, request: LLMRequest) -> LLMResult:
                    result = await actual.complete(request)
                    return result.model_copy(
                        update={"usage": result.usage.model_copy(update={"tokens_in": 999})}
                    )

            app = Orchestrator(
                Mismatch() if mode == "ledger_mismatch" else actual,
                ToolExecutor(fixture_engine().search_papers, clock=c, config=config),
                model="FAKE-model",
                clock=c,
                ledger=ledger,
                config=config,
            )
            outcome = await app.handle("alpha", setting="A", request_id="same-request")
        assert outcome.state == state and len(sent) == outcome.provider_calls == calls
        assert outcome.retries == 0
        records = [ledger.get_call(call.call_id) for call in outcome.calls]
        assert len({r.id for r in records}) == calls and all(
            r.request_id == "same-request" for r in records
        )
        if mode == "missing":
            assert outcome.tokens_in is None and outcome.tokens_out is None
            assert outcome.counted_tokens is None and not outcome.usage.complete
            assert outcome.usage.known_tokens_in == 23 and outcome.usage.unknown_usage_calls == 1
            assert records[0].tokens_in is None
        else:
            assert outcome.tokens_in == sum(r.tokens_in for r in records) == 23 * calls
            assert outcome.counted_tokens == sum(r.counted_tokens for r in records) == 27 * calls
            assert outcome.usage.complete
        if mode == "ledger_mismatch":
            assert outcome.failure_reason == "provider_persistence"
            assert "ledger_usage_mismatch" in outcome.calls[0].diagnostics
        if mode in {"draft_schema", "verify_schema"}:
            assert outcome.repair_calls == 1
            assert any(r.status == "error" and r.tokens_out == 7 for r in records)
