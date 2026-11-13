import pytest

from relay.app.interpret import interpret


@pytest.mark.parametrize(
    "text,paper",
    [
        ("Explain inner monologue", "moshi"),
        ("How does H-Net learn boundaries?", "dynamic_chunking"),
        ("Explain transformer attention", "transformer"),
    ],
)
def test_ambiguous_concepts_are_hints_not_filters(text: str, paper: str) -> None:
    result = interpret(text)
    assert result.paper_hint == paper and result.explicit_paper is None


def test_explicit_scope_and_multi_paper_comparison() -> None:
    assert interpret("In the Moshi paper, what is latency?").explicit_paper == "moshi"
    both = interpret("Compare Attention Is All You Need with Moshi latency")
    assert both.qtype == "compare" and both.numeric
    assert both.explicit_paper is None and both.paper_hint is None
    assert interpret("Compare them", paper_scope="transformer").explicit_paper == "transformer"


@pytest.mark.parametrize(
    "text",
    [
        "What is latency?",
        "What is the complexity?",
        "How many layers?",
        "Which model dimensions and sizes?",
        "What percentage is retained?",
    ],
)
def test_numeric_detection(text: str) -> None:
    assert interpret(text).numeric


@pytest.mark.parametrize(
    "text",
    [
        "What are Groq current rate limits?",
        "Does Vaani use Moshi in production?",
        "What is my account quota?",
        "Recommend a chunk size for RAG",
    ],
)
def test_absent_is_advisory(text: str) -> None:
    result = interpret(text)
    assert result.looks_out_of_scope and result.qtype == "absent"
    assert result.keywords and result.diagnostics
