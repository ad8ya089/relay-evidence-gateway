from scripts.report import headings, html_table


def test_report_preserves_exact_required_heading_order() -> None:
    values = headings()
    assert len(values) == 12
    assert values[0] == "1 What I built and why"
    assert values[5] == "Retrieval: quality versus cost"
    assert values[-1] == "How I worked with the assistant"
    assert "&lt;" in html_table("Table", {"headers": ["h"], "rows": [["<value>"]]})
