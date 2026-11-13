from argparse import Namespace
from pathlib import Path

import pytest

from eval import runner


async def test_root_url_is_rejected_before_application_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(**kwargs: object) -> None:
        raise AssertionError("Application created before validating the provider API path")

    monkeypatch.setattr(runner, "create_local_app", unexpected)
    args = Namespace(
        real=False,
        settings=["A"],
        limit=1,
        base_url="http://127.0.0.1:8081",
        output=tmp_path,
        quota_evidence=None,
    )
    with pytest.raises(ValueError, match="openai/v1"):
        await runner.run(args)
