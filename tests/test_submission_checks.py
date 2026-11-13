from pathlib import Path

import fitz
import pytest

from scripts.check_submission import member, pdf_issues, secret_findings, validate


def test_missing_artifacts_and_secret_fixture_fail_without_values(tmp_path: Path) -> None:
    assert validate(tmp_path)["status"] == "failed"
    key = "gsk_" + "X" * 32
    (tmp_path / "transcript.txt").write_text(key)
    issues = secret_findings(tmp_path)
    assert issues and all(key not in issue for issue in issues)
    with pytest.raises(ValueError, match="escapes"):
        member(tmp_path, "../outside")


def test_pdf_page_limit_and_missing_headings_are_reported(tmp_path: Path) -> None:
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    path = tmp_path / "bad.pdf"
    doc.save(path)
    issues = pdf_issues(path)
    assert any("two pages" in issue for issue in issues)
    assert any("Missing" in issue for issue in issues)
