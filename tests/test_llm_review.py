import json

import geopandas as gpd

from culvert_ai.io import read_vector
from culvert_ai.llm_review import import_llm_reviewed_labels


def test_import_llm_reviewed_labels_accepts_corrected_coordinates(tmp_path):
    review_path = tmp_path / "reviewed.jsonl"
    output_path = tmp_path / "reviewed.gpkg"
    csv_path = tmp_path / "reviewed.csv"
    review_row = {
        "review_id": "abc123",
        "source": {
            "source_file": "report.pdf",
            "report_date": "2026-06-03",
            "context_text": "R-8 NY-9G 73.898 W 42.056 N",
            "raw_coordinate_text": "73.898 W 42.056 N",
        },
        "extracted": {
            "latitude": 42.056,
            "longitude": -73.898,
            "route": "NY9G",
            "culvert_id": "CID149980",
            "label_confidence": 0.85,
        },
        "review": {
            "accepted": True,
            "latitude": 42.0561,
            "longitude": -73.8981,
            "route": "NY9G",
            "culvert_id": "CID149980",
            "label_confidence": 0.95,
            "reason": "coordinate appears in the site-location table",
        },
    }
    review_path.write_text(json.dumps(review_row) + "\n", encoding="utf-8")

    result = import_llm_reviewed_labels(review_path, output_path, csv_output=csv_path)
    reviewed = read_vector(output_path)

    assert result["rows"] == 1
    assert isinstance(reviewed, gpd.GeoDataFrame)
    assert reviewed.iloc[0]["label"] == "llm_reviewed_field_observed_culvert"
    assert reviewed.iloc[0]["label_confidence"] == 0.95
    assert reviewed.iloc[0]["latitude"] == 42.0561
    assert csv_path.exists()


def test_import_llm_reviewed_labels_rejects_invalid_rows(tmp_path):
    review_path = tmp_path / "reviewed.jsonl"
    output_path = tmp_path / "reviewed.gpkg"
    review_path.write_text(
        json.dumps(
            {
                "source": {"source_file": "report.pdf"},
                "extracted": {"latitude": 42.0, "longitude": -74.0},
                "review": {"accepted": False, "reason": "not a culvert row"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        import_llm_reviewed_labels(review_path, output_path)
    except ValueError as exc:
        assert "No accepted" in str(exc)
    else:
        raise AssertionError("Expected rejected review rows to fail import")
