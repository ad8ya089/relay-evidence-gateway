"""Human workload validation, with a separately named fixture-only path."""

from pathlib import Path

from eval.retrieval_offline import Question
from loadgen.settings import LoadSettings


class PrerequisiteError(ValueError):
    pass


def questions(
    path: Path, config: LoadSettings, *, fixture_only: bool = False
) -> dict[str, Question]:
    rows = [
        Question.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({q.qid for q in rows}) != len(rows) or any(
        not q.qid.strip() or not q.question.strip() for q in rows
    ):
        raise PrerequisiteError("Question IDs/text must be nonempty and unique")
    if not fixture_only and (
        len(rows) != config.expected_questions
        or sum(q.answerable for q in rows) != config.expected_answerable
    ):
        raise PrerequisiteError(
            f"Official workload requires {config.expected_questions} human-authored questions "
            f"({config.expected_answerable} answerable); found {len(rows)}. "
            "Use development_smoke explicitly for fixtures."
        )
    if not rows:
        raise PrerequisiteError("Question workload is empty")
    return {q.qid: q for q in rows}
