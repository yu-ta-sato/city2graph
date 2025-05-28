import pytest, zipfile, os
import pandas as pd, geopandas as gpd, numpy as np
from shapely.geometry import Point, LineString

from city2graph.transportation import (
    load_gtfs,
    get_od_pairs,
    travel_summary_network,
    _create_timestamp,
    _time_to_seconds,
    _vectorized_time_to_seconds,
)


@pytest.fixture
def minimal_gtfs_zip(tmp_path):
    # create minimal GTFS zip
    stops = pd.DataFrame(
        {
            "stop_id": ["a", "b"],
            "stop_lat": ["0", "1"],
            "stop_lon": ["0", "1"],
        }
    )
    routes = pd.DataFrame({"route_id": ["r1"], "route_type": ["3"]})
    shapes = pd.DataFrame(
        {
            "shape_id": ["s1", "s1"],
            "shape_pt_lat": ["0", "1"],
            "shape_pt_lon": ["0", "1"],
            "shape_pt_sequence": ["1", "2"],
        }
    )
    trips = pd.DataFrame(
        {
            "route_id": ["r1"],
            "service_id": ["sv1"],
            "trip_id": ["t1"],
            "shape_id": ["s1"],
        }
    )
    stop_times = pd.DataFrame(
        {
            "trip_id": ["t1", "t1"],
            "stop_id": ["a", "b"],
            "stop_sequence": ["1", "2"],
            "departure_time": ["00:00:00", "00:05:00"],
            "arrival_time": ["00:00:00", "00:05:00"],
        }
    )
    calendar = pd.DataFrame(
        {
            "service_id": ["sv1"],
            "start_date": ["20210101"],
            "end_date": ["20210102"],
            "monday": [1],
            "tuesday": [1],
            "wednesday": [1],
            "thursday": [1],
            "friday": [1],
            "saturday": [0],
            "sunday": [0],
        }
    )
    calendar_dates = pd.DataFrame(columns=["service_id", "date", "exception_type"])
    zip_path = tmp_path / "gtfs.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, df in [
            ("stops", stops),
            ("routes", routes),
            ("shapes", shapes),
            ("trips", trips),
            ("stop_times", stop_times),
            ("calendar", calendar),
            ("calendar_dates", calendar_dates),
        ]:
            data = df.to_csv(index=False).encode("utf-8")
            zf.writestr(f"{name}.txt", data)
    return zip_path


def test_load_gtfs(minimal_gtfs_zip):
    gtfs = load_gtfs(str(minimal_gtfs_zip))
    assert isinstance(gtfs, dict)
    assert "stops" in gtfs and isinstance(gtfs["stops"], gpd.GeoDataFrame)
    assert "shapes" in gtfs and isinstance(gtfs["shapes"], gpd.GeoDataFrame)
    assert "routes" in gtfs and isinstance(gtfs["routes"], pd.DataFrame)


def test_get_od_pairs(minimal_gtfs_zip):
    gtfs = load_gtfs(str(minimal_gtfs_zip))
    od = get_od_pairs(gtfs, include_geometry=False)
    assert isinstance(od, pd.DataFrame)
    assert not od.empty
    expected = [
        "trip_id",
        "orig_stop_id",
        "dest_stop_id",
        "departure_timestamp",
        "arrival_timestamp",
        "service_id",
        "orig_stop_sequence",
        "dest_stop_sequence",
    ]
    for col in expected:
        assert col in od.columns
    gen = get_od_pairs(gtfs, include_geometry=False, as_generator=True, chunk_size=1)
    chunks = list(gen)
    assert all(isinstance(c, pd.DataFrame) for c in chunks)
    assert sum(len(c) for c in chunks) == len(od)


def test_travel_summary_network(minimal_gtfs_zip):
    gtfs = load_gtfs(str(minimal_gtfs_zip))
    summary_gdf = travel_summary_network(gtfs, as_gdf=True)
    assert hasattr(summary_gdf, "geometry")
    summary_dict = travel_summary_network(gtfs, as_gdf=False)
    assert isinstance(summary_dict, dict)
    assert next(iter(summary_dict)) in summary_dict


def test_timestamp_and_time_conversions():
    from datetime import datetime

    ts = _create_timestamp("25:00:00", datetime(2021, 1, 1))
    assert ts.hour == 1 and ts.day == 2
    assert _create_timestamp(None, datetime(2021, 1, 1)) is None
    assert _time_to_seconds("01:02:03") == 3723
    assert np.isnan(_time_to_seconds(None))
    s = pd.Series(["00:00:10", "00:01:00", None])
    sec = _vectorized_time_to_seconds(s)
    assert sec.iloc[0] == 10 and sec.iloc[1] == 60 and np.isnan(sec.iloc[2])


def test_create_timestamp_invalid():
    from datetime import datetime

    assert _create_timestamp("ab:cd:ef", datetime(2021, 1, 1)) is None


def test_get_od_pairs_with_geometry(minimal_gtfs_zip):
    gtfs = load_gtfs(str(minimal_gtfs_zip))
    od_gdf = get_od_pairs(gtfs, include_geometry=True)
    assert hasattr(od_gdf, "geometry")
    assert len(od_gdf) > 0


def test_get_od_pairs_generator_with_geometry(minimal_gtfs_zip):
    gtfs = load_gtfs(str(minimal_gtfs_zip))
    gen = get_od_pairs(gtfs, include_geometry=True, as_generator=True, chunk_size=1)
    chunks = list(gen)
    assert all(hasattr(c, "geometry") for c in chunks)


def test_load_gtfs_invalid_path():
    with pytest.raises(KeyError):
        load_gtfs("nonexistent.zip")
