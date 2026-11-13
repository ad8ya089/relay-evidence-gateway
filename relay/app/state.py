"""Explicit P4 transitions; terminal states never resume model work."""

from relay.app.models import State

TERMINAL = {
    State.RELEASED,
    State.REFUSED_LOCAL,
    State.REFUSED_MODEL,
    State.ABSTAINED,
    State.TIMED_OUT,
    State.FAILED,
}
TRANSITIONS = {
    State.ARRIVED: {State.INTERPRETED},
    State.INTERPRETED: {State.RETRIEVED},
    State.RETRIEVED: {State.GATE},
    State.GATE: {State.BROADEN, State.DRAFTING, State.REFUSED_LOCAL},
    State.BROADEN: {State.RETRIEVED},
    State.DRAFTING: {State.VALIDATING, State.REFUSED_MODEL},
    State.VALIDATING: {State.REPAIRING, State.RELEASED, State.ABSTAINED},
    State.REPAIRING: {State.VALIDATING, State.REFUSED_MODEL},
}


class InvalidTransition(RuntimeError):
    """A pipeline invariant was violated."""


class StateMachine:
    def __init__(self) -> None:
        self.state = State.ARRIVED
        self.history = [self.state]

    def validate(self, target: State) -> None:
        if self.state in TERMINAL or (
            target not in TRANSITIONS.get(self.state, set())
            and target not in {State.TIMED_OUT, State.FAILED}
        ):
            raise InvalidTransition(f"Invalid pipeline transition: {self.state} -> {target}")

    def move(self, target: State) -> None:
        self.validate(target)
        self.state = target
        self.history.append(target)
