import pandas as pd

from culvert_ai.consensus_ranking import add_consensus_precision_features


def test_consensus_ranking_is_label_free_and_rewards_agreement():
    base = {
        "route": ["1", "1", "1"],
        "x_m": [0.0, 200.0, 400.0],
        "y_m": [0.0, 0.0, 0.0],
        "flow_c_score": [0.9, 0.8, 0.7],
        "route_z_score": [1.5, 1.0, 0.5],
        "minimum_any_crossing_distance_m": [2.0, 200.0, 300.0],
        "distance_to_nearest_mapped_crossing_m": [3.0, 200.0, 300.0],
        "minimum_dem_crossing_distance_m": [4.0, 200.0, 300.0],
        "mapped_dem_crossing_distance_agreement_m": [5.0, 200.0, 300.0],
        "dem_crossing_threshold_support_count_50m": [4, 0, 0],
        "local_road_stream_angle_degrees": [90.0, 0.0, 0.0],
        "route_low_point_score": [1.0, 0.0, 0.0],
        "mapped_dem_both_within_50m": [1, 0, 0],
    }
    frame = pd.DataFrame({"candidate_id": ["a", "b", "c"], **base})
    ranked = add_consensus_precision_features(frame)
    assert ranked.loc[0, "precision_score"] > ranked.loc[1, "precision_score"]
    assert ranked.loc[0, "supporting_family_count"] >= 3
    assert not any("official" in column or "ground_truth" in column for column in ranked.columns)

