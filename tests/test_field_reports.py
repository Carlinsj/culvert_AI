from pathlib import Path

from culvert_ai.field_reports import (
    _culvert_ids,
    _normalize_route,
    _records_from_text,
    _report_date,
)


def test_report_date_parses_team_two_filename_formats():
    assert _report_date(Path("Team 2 field report - 5.21.26.pdf")) == "2026-05-21"
    assert _report_date(Path("Team 2 - Field Report 6_04_2026.docx.pdf")) == "2026-06-04"
    assert _report_date(Path("Team 2 Report updated Jun 1, 2026.pdf")) == "2026-06-01"
    assert _report_date(Path("Team 2 Field Report June 17, 2026.docx.pdf")) == "2026-06-17"


def test_records_from_text_handles_zero_width_pdf_spacing():
    text = "\u200bR-8\u200b NY-9G (CID: 149980) \u200b73.898 W\u200b \u200b42.056 N\u200b"

    records = _records_from_text(
        Path("Team 2 Field Report June 3, 2026.docx - Google Docs.pdf"),
        text,
    )

    assert len(records) == 1
    assert records[0].report_date == "2026-06-03"
    assert records[0].nysdot_region == "8"
    assert records[0].route == "NY9G"
    assert records[0].latitude == 42.056
    assert records[0].longitude == -73.898


def test_route_and_culvert_ids_are_normalized():
    assert _normalize_route("NY-9G") == "NY9G"
    assert _normalize_route("State Route 212") == "NY212"
    assert _culvert_ids("NY-9G (CID: 149980) SC150237 CID: Not Assigned") == [
        "CID149980",
        "SC150237",
        "CID-NOTASSIGNED",
    ]
