import json
from pathlib import Path

import pytest

from relay.app.delivery import iter_validated_answer
from relay.app.models import Draft, State
from relay.app.orchestrator import Orchestrator
from relay.app.settings import pipeline_settings
from relay.control.ledger import Ledger
from relay.obs.events import EventWriter
from relay.providers.base import LLMRequest, LLMResult
from relay.providers.groq import GroqCompatibleProvider
from relay.retrieval.models import RetrievalHit, RetrievalResult
from relay.retrieval.search import RetrievalEngine
from relay.tools.executor import ToolExecutor
from standin.service import create_app
from tests.fixtures.app_integration import integrated
from tests.fixtures.backends import fake_chunk
from tests.standin_helpers import profile
from tests.standin_server import running


@pytest.mark.parametrize("reject", [False, True])
async def test_actual_p2_p3_pipeline_and_ledger_usage(tmp_path: Path, reject: bool) -> None:
    async with integrated(tmp_path, reject=reject) as (app, ledger, path):
        outcome = await app.handle("alpha", setting="A", request_id="logical-request")
        assert outcome.state == (State.ABSTAINED if reject else State.RELEASED)
        expected = 4 if reject else 2
        assert outcome.provider_calls == expected and outcome.repair_calls == int(reject)
        assert all(call.ledger_reconciled for call in outcome.calls)
        rows = [ledger.get_call(call.call_id) for call in outcome.calls]
        assert outcome.tokens_in == sum(row.tokens_in for row in rows)
        assert outcome.tokens_out == sum(row.tokens_out for row in rows)
        assert outcome.tokens_cached == sum(row.tokens_cached for row in rows)
        assert outcome.counted_tokens == sum(row.counted_tokens for row in rows)
        assert all(not row.real and row.request_id == "logical-request" for row in rows)
        assert outcome.usage.complete and outcome.retries == 0
        assert outcome.first_token_ms is None
        output = "".join(
            [
                part
                async for part in iter_validated_answer(outcome, clock=app.clock, events=app.events)
            ]
        )
        if reject:
            assert (
                output == "I couldn't verify a reliable answer from the available paper passages."
            )
        else:
            assert "v7" in output and outcome.citations[0].chunk_id in outcome.supplied_chunk_ids
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert all(e["rid"] == "logical-request" for e in events)
        names = [e["ev"] for e in events]
        assert (
            names.index("first_delta") < names.index("validated") < names.index("first_user_token")
        )
        assert "FAKE alpha evidence" not in path.read_text()
        if reject:
            assert outcome.tokens_cached > 0  # Repair reuses the exact draft prefix.


async def test_real_corpus_answer_with_actual_p3_service(tmp_path: Path) -> None:
    if not Path("data/chunks/A.jsonl").exists():
        pytest.skip(
            "Pinned corpus not installed; local pipeline smoke exercises it after ingestion"
        )
    async with integrated(tmp_path, engine=RetrievalEngine()) as (app, _, _):
        outcome = await app.handle("Explain encoder-decoder attention", setting="A")
    assert outcome.state == State.RELEASED, outcome.failure_reason
    assert outcome.citations and all(
        c.arxiv_version == 7 and c.arxiv_id == "1706.03762" for c in outcome.citations
    )


@pytest.mark.parametrize("fault", ["chunk_id", "quote", "version", "section", "page", "numeric"])
async def test_corrupted_evidence_is_never_released_through_actual_adapter(
    tmp_path: Path, fault: str
) -> None:
    source = "The theoretical latency is 160 ms, 200 ms in practice."
    chunk = fake_chunk(0, source, "moshi")
    hit = RetrievalHit(**chunk.model_dump(), score=1, rank=1)
    if fault == "version":
        hit = hit.model_copy(update={"arxiv_version": 999})
    elif fault == "section":
        hit = hit.model_copy(update={"section": None})
    elif fault == "page":
        hit = hit.model_copy(update={"page_start": 0})

    async def search(query: str, *args: object, **kwargs: object) -> RetrievalResult:
        return RetrievalResult(
            status="ok", hits=[hit], setting="A", query_used=query, method="bm25"
        )

    config = pipeline_settings().model_copy(update={"repair_max": 0})
    async with integrated(tmp_path, config=config) as (app, _, _):
        original = app.runner.provider

        class CorruptResponse:
            async def complete(self, request: LLMRequest) -> LLMResult:
                result = await original.complete(request)
                draft = Draft.model_validate_json(result.text)
                if fault == "chunk_id":
                    draft.claims[0].evidence[0].chunk_id = "forged"
                elif fault == "quote":
                    draft.claims[0].evidence[0].quote = "This quotation does not exist."
                elif fault == "numeric":
                    draft.claims[0].text = "The practical latency is 160 ms."
                return result.model_copy(update={"text": draft.model_dump_json()})

        app.runner.provider = CorruptResponse()
        app.executor = ToolExecutor(search, clock=app.clock, events=app.events, config=config)
        outcome = await app.handle("Moshi latency", setting="A")
    assert outcome.state != State.RELEASED and not outcome.citations
    assert "practical latency is 160" not in outcome.text


async def test_pipeline_over_ephemeral_local_http(tmp_path: Path, loopback_ports: set[int]) -> None:
    p = profile()
    app = create_app(profile=p)
    async with running(app, loopback_ports) as url:
        with Ledger(tmp_path / "usage.sqlite3") as ledger:
            events = EventWriter(tmp_path / "wire.jsonl")
            async with GroqCompatibleProvider(
                url, None, p.model, ledger=ledger, events=events
            ) as provider:
                from tests.fixtures.backends import fixture_engine

                pipeline = Orchestrator(
                    provider,
                    ToolExecutor(fixture_engine().search_papers, events=events),
                    model=p.model,
                    ledger=ledger,
                    events=events,
                    run_id="FAKE-wire-integration",
                )
                outcome = await pipeline.handle("alpha", setting="A")
    assert outcome.state == State.RELEASED and outcome.provider_ttft_ms is not None
    assert outcome.provider_ttft_ms < outcome.provider_ms and outcome.first_token_ms is None
