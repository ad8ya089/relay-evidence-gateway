import pytest

from relay.control.settings import load_policy
from relay.providers.factory import selection
from relay.providers.guard import RealCallDenied


@pytest.mark.parametrize(
    "provider,allow,real",
    [
        ("standin", "0", True),
        ("groq", "0", True),
        ("groq", "1", False),
    ],
)
def test_real_selection_requires_all_permissions(
    monkeypatch: pytest.MonkeyPatch, provider: str, allow: str, real: bool
) -> None:
    monkeypatch.setenv("RELAY_PROVIDER", provider)
    monkeypatch.setenv("RELAY_ALLOW_REAL", allow)
    with pytest.raises(RealCallDenied):
        selection(load_policy(), real=real, base_url=None)


def test_real_credentials_origin_and_no_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RELAY_PROVIDER", "groq")
    monkeypatch.setenv("RELAY_ALLOW_REAL", "1")
    with pytest.raises(RealCallDenied, match="credential"):
        selection(load_policy(), real=True, base_url=None)
    monkeypatch.setenv("GROQ_API_KEY", "FAKE-test-only")
    with pytest.raises(RealCallDenied, match="origin"):
        selection(load_policy(), real=True, base_url="http://127.0.0.1:8081")
    url, key = selection(load_policy(), real=True, base_url=None)
    assert url == "https://api.groq.com/openai/v1" and key == "FAKE-test-only"


def test_standin_default_never_reads_real_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RELAY_PROVIDER", raising=False)
    assert selection(load_policy(), real=False, base_url=None)[1] is None


async def test_real_application_lifecycle_proves_guard_without_dispatch(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    from relay.app.api import create_local_app

    if not Path("data/corpus_manifest.json").exists():
        pytest.skip("Local pinned corpus is not installed")
    monkeypatch.setenv("RELAY_PROVIDER", "groq")
    monkeypatch.setenv("RELAY_ALLOW_REAL", "1")
    monkeypatch.setenv("GROQ_API_KEY", "FAKE-lifecycle-only")
    path = Path(str(tmp_path))
    app = create_local_app(
        real=True,
        ledger_path=path / "ledger",
        cache_path=path / "cache",
        events_path=path / "events",
    )
    async with app.router.lifespan_context(app):
        plane = app.state.control
        state = await plane.state()
        assert state["provider_source"] == "real"
        assert state["real_guard"]["enabled"] and state["real_guard"]["durable"]
        assert plane.ledger.daily_usage(plane.model).attempts == 0
