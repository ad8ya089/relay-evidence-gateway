"""Compact LLM verification requires every claim index exactly once."""

from relay.app.calls import CallRunner, Invocation, RequestCalls
from relay.app.models import Draft, ValidationResult, Verification
from relay.app.prompts import Prompts


class VerificationCallError(RuntimeError):
    def __init__(self, invocation: Invocation) -> None:
        self.invocation = invocation
        super().__init__("Verifier provider operation failed")


def invalid_verifier(reason: str, deterministic: ValidationResult) -> ValidationResult:
    return deterministic.model_copy(
        update={
            "status": "verify_invalid",
            "verifier_passed": False,
            "repairable": True,
            "failure_reason": reason,
            "verifier_diagnostics": [reason],
        }
    )


async def verify(
    draft: Draft,
    deterministic: ValidationResult,
    runner: CallRunner,
    book: RequestCalls,
    prompts: Prompts,
) -> ValidationResult:
    if not deterministic.deterministic_passed:
        return deterministic
    invocation = await runner.invoke(book, "verify", prompts.verify(draft.claims), Verification)
    if invocation.failure is not None:
        if invocation.failure.kind == "invalid_response":
            return invalid_verifier("verifier_schema_invalid", deterministic)
        raise VerificationCallError(invocation)
    assert invocation.result is not None
    try:
        response = Verification.model_validate_json(invocation.result.text)
    except ValueError:
        return invalid_verifier("verifier_schema_invalid", deterministic)
    indices = [v.i for v in response.verdicts]
    if len(indices) != len(draft.claims) or set(indices) != set(range(len(draft.claims))):
        return invalid_verifier("verifier_indices_invalid", deterministic)
    supported = {v.i: v.v == "supported" for v in response.verdicts}
    passed = all(supported.values())
    checks = [
        c.model_copy(
            update={
                "passed": supported[c.i],
                "checks": c.checks | {"llm_supported": supported[c.i]},
                "reasons": c.reasons + ([] if supported[c.i] else ["verifier_rejected"]),
            }
        )
        for c in deterministic.claims
    ]
    return ValidationResult(
        status="pass" if passed else "fail_repairable",
        claims=checks,
        deterministic_passed=True,
        verifier_passed=passed,
        verdicts=response.verdicts,
        repairable=not passed,
        failure_reason=None if passed else "verifier_rejected",
    )
