"""A sealed, validated-only delivery boundary; internal provider text never enters it."""

from collections.abc import AsyncIterator

from relay.app.models import Outcome, State
from relay.app.render import refusal_text
from relay.app.settings import pipeline_settings
from relay.app.text import digest
from relay.obs.clock import Clock, SystemClock
from relay.obs.events import EventContext, EventWriter


class DeliveryError(ValueError):
    """Outcome was not authorized for delivery, changed, or expired."""


def _fingerprint(outcome: Outcome) -> str:
    return digest(
        {
            "outcome": outcome.model_dump(exclude={"first_user_token_at", "first_token_ms"}),
            "arrival_ns": outcome.arrival_ns,
            "deadline_ns": outcome.deadline_ns,
        }
    )


def _authorize(outcome: Outcome) -> None:
    if not outcome.releasable or not outcome.text:
        raise DeliveryError("No releasable response")
    if outcome.state == State.RELEASED:
        if not (
            outcome.validation
            and outcome.validation.status == "pass"
            and outcome.validation.verifier_passed
            and outcome.validation.deterministic_passed
            and outcome.citations
        ):
            raise DeliveryError("Answer lacks completed validation")
    elif outcome.state == State.REFUSED_LOCAL:
        if not outcome.validated_refusal or outcome.text != refusal_text("local"):
            raise DeliveryError("Invalid local refusal")
    elif outcome.state == State.REFUSED_MODEL:
        if not outcome.validated_refusal or outcome.text != refusal_text("model"):
            raise DeliveryError("Invalid model refusal")
    elif outcome.state == State.ABSTAINED:
        if outcome.text != refusal_text("validation"):
            raise DeliveryError("Invalid validation abstention")
    else:
        raise DeliveryError("Pipeline has not reached a releasable state")
    outcome._delivery_seal = _fingerprint(outcome)


async def iter_validated_answer(
    outcome: Outcome, *, clock: Clock | None = None, events: EventWriter | None = None
) -> AsyncIterator[str]:
    clock = clock or SystemClock()
    size = pipeline_settings().delivery_chunk_chars
    for start in range(0, len(outcome.text) or 1, size):
        if outcome._delivery_seal is None or outcome._delivery_seal != _fingerprint(outcome):
            raise DeliveryError("Unapproved or modified outcome")
        now = clock.monotonic_ns()
        if now >= outcome.deadline_ns:
            raise DeliveryError("Logical request deadline expired before delivery")
        if outcome.first_user_token_at is None:
            outcome.first_user_token_at = now
            outcome.first_token_ms = (now - outcome.arrival_ns) / 1e6
            if events is not None:
                events.write(
                    EventContext(
                        rid=outcome.request_id, run=outcome.run_id, arrival_ns=outcome.arrival_ns
                    ),
                    "first_user_token",
                    state=outcome.state.value,
                )
        yield outcome.text[start : start + size]
    if events is not None:
        events.write(
            EventContext(rid=outcome.request_id, run=outcome.run_id, arrival_ns=outcome.arrival_ns),
            "done",
            state=outcome.state.value,
            status=outcome.status,
            failure_reason=outcome.failure_reason,
            total_ms=(clock.monotonic_ns() - outcome.arrival_ns) / 1e6,
            provider_calls=outcome.provider_calls,
            usage_complete=outcome.usage.complete,
        )
