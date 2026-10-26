"""Recorded-versus-predicted workflow invariants."""

from staging.comparison_service import compare, options
from scripts.serve_feature_staging import benchmark_payload


def test_comparison_options_keep_candidate_populations_and_inventory_counts_explicit():
    result = options()
    assert result["actual_total"] == 959
    assert result["supported_search_records"] == 847
    assert result["outside_supported_searches"] == 112
    models = {item["id"]: item for item in result["models"]}
    assert models["baseline_rf"]["sites"] == 13065
    assert models["v3"]["sites"] == 68302


def test_all_inventory_reproduces_frozen_baseline_top_1000():
    result = compare({"selection": "all", "model": "baseline_rf", "budget": 1000, "tolerance_m": 20})
    assert result["active_actual"] == 847
    assert result["shown_predictions"] == 1000
    assert result["matched"] == 123
    assert result["missed"] == 724
    assert result["budget_scope"] == "Top K by saved global rank after geographic selection"


def test_rock_ledge_selection_loads_expected_records_and_marks_each_actual_once():
    result = compare({"selection": "rock", "model": "baseline_rf", "budget": 1000, "tolerance_m": 20})
    assert result["active_actual"] == 7
    actuals = result["actuals"]["features"]
    assert len(actuals) == 7
    assert {item["properties"]["match_status"] for item in actuals} <= {"matched", "missed"}


def test_v3_equal_budget_is_not_mislabeled_as_full_population_recall():
    result = compare({"selection": "all", "model": "v3", "budget": 13065, "tolerance_m": 20})
    assert result["model"]["full_population"] == 68302
    assert result["shown_predictions"] == 13065
    assert result["matched"] == 236


def test_v4_challenger_is_available_but_not_the_default():
    models = options()["models"]
    assert models[0]["id"] == "baseline_rf"
    v4 = next(item for item in models if item["id"] == "v4")
    assert v4["sites"] == 13065
    result = compare({"selection": "all", "model": "v4", "budget": 1000, "tolerance_m": 20})
    assert result["matched"] == 23


def test_v5_baseline_preserving_challenger_is_selectable_but_not_default():
    models = options()["models"]
    assert models[0]["id"] == "baseline_rf"
    v5 = next(item for item in models if item["id"] == "v5")
    assert v5["sites"] == 13065
    result = compare({"selection": "all", "model": "v5", "budget": 1000, "tolerance_m": 20})
    assert result["matched"] == 113


def test_v6_compact_physical_queue_is_selectable_but_not_default():
    models = options()["models"]
    assert models[0]["id"] == "baseline_rf"
    v6 = next(item for item in models if item["id"] == "v6")
    assert v6["sites"] == 50826
    result = compare({"selection": "all", "model": "v6", "budget": 1000, "tolerance_m": 20})
    assert result["matched"] == 123
    assert result["model"]["full_population"] == 50826


def test_v7_surveyed_route_comparison_uses_sealed_31_record_denominator():
    models = options()["models"]
    v7 = next(item for item in models if item["id"] == "v7")
    assert v7["sites"] == 142
    result = compare(
        {"selection": "surveyed", "model": "v7", "budget": 119, "tolerance_m": 20}
    )
    assert result["actual_total"] == result["active_actual"] == 31
    assert result["shown_predictions"] == 119
    assert result["matched"] == 22
    assert result["model"]["full_population"] == 142


def test_performance_payload_separates_equal_budget_from_full_population():
    result = benchmark_payload()
    assert result["budgets"]["1000"]["existing_baseline_rf"]["matches"] == 123
    assert result["budgets"]["1000"]["v4_physical_ranker"]["matches"] == 23
    assert result["budgets"]["1000"]["v5_baseline_preserving"]["matches"] == 113
    assert result["budgets"]["1000"]["v6_compact_physical"]["matches"] == 123
    full = {item["id"]: item for item in result["full_population"]}
    assert full["existing_baseline_rf"]["matches"] == 347
    assert full["v3_physical_feature_model"] == {
        "id": "v3_physical_feature_model",
        "label": "v3 RF",
        "sites": 68302,
        "matches": 672,
        "recall": 672 / 847,
    }
    assert full["v6_compact_physical"] == {
        "id": "v6_compact_physical",
        "label": "v6 compact physical",
        "sites": 50826,
        "matches": 638,
        "recall": 638 / 847,
    }
