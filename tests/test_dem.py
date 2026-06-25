from culvert_ai.dem import dem_tiles_for_bounds, usgs_3dep_tile_url


def test_dem_tiles_for_ulster_bounds():
    bounds = (-74.75, 41.58, -73.92, 42.17)

    assert dem_tiles_for_bounds(bounds) == [
        "n41w075",
        "n41w074",
        "n42w075",
        "n42w074",
    ]


def test_dem_tiles_do_not_add_next_tile_on_exact_max_boundary():
    bounds = (-75.0, 41.0, -74.0, 42.0)

    assert dem_tiles_for_bounds(bounds) == ["n41w075"]


def test_usgs_3dep_tile_url():
    assert (
        usgs_3dep_tile_url("n41w075")
        == "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1/TIFF/current/n41w075/USGS_1_n41w075.tif"
    )
