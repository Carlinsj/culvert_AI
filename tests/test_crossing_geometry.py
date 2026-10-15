from __future__ import annotations

import math

import pytest
from shapely.geometry import LineString, Point

from culvert_ai.crossing_geometry import (
    acute_angle_degrees,
    bilateral_continuity,
    local_line_tangent,
)


def test_acute_angle_is_undirected_and_bounded() -> None:
    assert acute_angle_degrees((1, 0), (0, 1)) == pytest.approx(90.0)
    assert acute_angle_degrees((1, 0), (-1, 0)) == pytest.approx(0.0)
    assert acute_angle_degrees((1, 0), (1, 1)) == pytest.approx(45.0)
    assert math.isnan(acute_angle_degrees((0, 0), (1, 0)))


def test_local_tangent_and_bilateral_continuity() -> None:
    road = LineString([(-100, 0), (100, 0)])
    crossing = LineString([(0, -50), (0, 50)])
    parallel = LineString([(-50, 10), (50, 10)])
    origin, tangent = local_line_tangent(road, Point(2, 3))
    assert origin.distance(Point(2, 0)) < 1e-9
    assert tangent == pytest.approx((1.0, 0.0))
    assert bilateral_continuity(crossing, origin, tangent, 40) == 1.0
    assert bilateral_continuity(parallel, origin, tangent, 40) == 0.0

