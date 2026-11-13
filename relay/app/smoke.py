"""Local-only end-to-end QA diagnostic; no real-provider option or credential lookup."""

import argparse
import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from relay.app.delivery import iter_validated_answer
from relay.app.local import local_standin
from relay.app.orchestrator import Orchestrator
from relay.app.settings import pipeline_settings
from relay.control.ledger import Ledger
from relay.obs.clock import SystemClock
from relay.obs.events import EventWriter, Sanitizer
from relay.providers.groq import GroqCompatibleProvider, is_loopback_url
from relay.retrieval.search import RetrievalEngine
from relay.tools.executor import ToolExecutor
from standin.profile import load_profile


@asynccontextmanager
async def endpoint(base_url: str | None) -> AsyncIterator[str]:
    if base_url is not None:
        if not is_loopback_url(base_url):
            raise ValueError("Pipeline smoke is restricted to loopback stand-ins")
        yield base_url
    else:
        async with local_standin(pipeline_settings()) as url:
            yield url


async def run(args: argparse.Namespace) -> int:
    clock = SystemClock()
    config = pipeline_settings()
    model = load_profile("development").model
    async with endpoint(args.base_url) as url:
        with Ledger(args.ledger, clock=clock) as ledger:
            events = EventWriter(args.events, clock=clock)
            engine = RetrievalEngine(clock=clock)
            async with GroqCompatibleProvider(
                url, None, model, ledger=ledger, events=events, clock=clock
            ) as provider:
                app = Orchestrator(
                    provider,
                    ToolExecutor(engine.search_papers, clock=clock, events=events, config=config),
                    model=model,
                    ledger=ledger,
                    events=events,
                    clock=clock,
                    run_id="p4-smoke-" + uuid4().hex,
                    config=config,
                )
                outcome = await app.handle(
                    args.question, setting=args.setting, deadline_ms=args.deadline_ms
                )
                text = (
                    "".join(
                        [
                            part
                            async for part in iter_validated_answer(
                                outcome, clock=clock, events=events
                            )
                        ]
                    )
                    if outcome.releasable
                    else ""
                )
                summary = {
                    "source": "stand-in",
                    "state": outcome.state.value,
                    "status": outcome.status,
                    "text": text,
                    "citations": [c.model_dump() for c in outcome.citations],
                    "provider_calls": outcome.provider_calls,
                    "repair_calls": outcome.repair_calls,
                    "usage": outcome.usage.model_dump(),
                    "total_ms": outcome.total_ms,
                    "provider_ttft_ms": outcome.provider_ttft_ms,
                    "first_token_ms": outcome.first_token_ms,
                    "failure_reason": outcome.failure_reason,
                    "setting": outcome.setting,
                }
    print(json.dumps(Sanitizer().value(summary), ensure_ascii=True, indent=2))
    return 0 if outcome.status == "ok" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--setting", choices=["A", "B", "C"], default="A")
    parser.add_argument("--deadline-ms", type=int)
    parser.add_argument("--base-url")
    parser.add_argument("--ledger", type=Path, default=Path("results/p4_smoke.sqlite3"))
    parser.add_argument("--events", type=Path, default=Path("results/p4_events.jsonl"))
    try:
        return asyncio.run(run(parser.parse_args()))
    except (ValueError, OSError, RuntimeError):
        print("Pipeline smoke failed: invalid local configuration or unavailable service/corpus")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
