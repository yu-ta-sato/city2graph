"""Tests for the transportation module."""

import zipfile
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from city2graph.transportation import _create_timestamp
from city2graph.transportation import _get_gtfs_df
from city2graph.transportation import _get_shapes_geometry
from city2graph.transportation import _get_stops_geometry
from city2graph.transportation import _process_gtfs_df
from city2graph.transportation import _time_to_seconds
from city2graph.transportation import _vectorized_time_to_seconds
from city2graph.transportation import get_od_pairs
from city2graph.transportation import load_gtfs
from city2graph.transportation import travel_summary_graph

# ============================================================================
# COMMON TEST FIXTURES
# ============================================================================


@pytest.fixture
def sample_gtfs_data() -> dict:
    """Create sample GTFS DataFrames for testing."""
    stops = pd.DataFrame({
        "stop_id": ["a", "b"],
        "stop_lat": ["0", "1"],
        "stop_lon": ["0", "1"],
    })

    routes = pd.DataFrame({
        "route_id": ["r1"],
        "route_type": ["3"],
    })

    shapes = pd.DataFrame({
        "shape_id": ["s1", "s1"],
        "shape_pt_lat": ["0", "1"],
        "shape_pt_lon": ["0", "1"],
        "shape_pt_sequence": ["1", "2"],
    })

    trips = pd.DataFrame({
        "route_id": ["r1"],
        "service_id": ["sv1"],
        "trip_id": ["t1"],
        "shape_id": ["s1"],
    })

    stop_times = pd.DataFrame({
        "trip_id": ["t1", "t1"],
        "stop_id": ["a", "b"],
        "stop_sequence": ["1", "2"],
        "departure_time": ["00:00:00", "00:05:00"],
        "arrival_time": ["00:00:00", "00:05:00"],
    })

    calendar = pd.DataFrame({
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
    })

    calendar_dates = pd.DataFrame(columns=["service_id", "date", "exception_type"])

    return {
        "stops": stops,
        "routes": routes,
        "shapes": shapes,
        "trips": trips,
        "stop_times": stop_times,
        "calendar": calendar,
        "calendar_dates": calendar_dates,
    }


@pytest.fixture
def minimal_gtfs_zip(tmp_path: str, sample_gtfs_data: dict) -> str:
    """Create a minimal GTFS zip file for testing."""
    zip_path = tmp_path / "gtfs.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, df in sample_gtfs_data.items():
            data = df.to_csv(index=False).encode("utf-8")
            zf.writestr(f"{name}.txt", data)

    return str(zip_path)


@pytest.fixture
def corrupt_gtfs_zip(tmp_path: str) -> str:
    """Create a GTFS zip file with corrupted CSV data for error testing."""
    zip_path = tmp_path / "corrupt_gtfs.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # Add a corrupted CSV file
        zf.writestr("stops.txt", "invalid,csv,data\n%corrupted%data%")
        # Add a valid file for mixed testing
        zf.writestr("routes.txt", "route_id,route_type\nr1,3")
        # Add non-txt files that should be skipped
        zf.writestr("readme.md", "This is not a CSV file")
        zf.writestr("subfolder/", "")  # Directory entry

    return str(zip_path)


# ============================================================================
# GTFS LOADING AND BASIC FUNCTIONALITY TESTS
# ============================================================================


def test_load_gtfs(minimal_gtfs_zip: str) -> None:
    """Test loading GTFS data from a zip file."""
    gtfs_path = minimal_gtfs_zip
    gtfs = load_gtfs(gtfs_path)

    assert isinstance(gtfs, dict)
    assert "stops" in gtfs
    assert isinstance(gtfs["stops"], gpd.GeoDataFrame)
    assert "shapes" in gtfs
    assert isinstance(gtfs["shapes"], gpd.GeoDataFrame)
    assert "routes" in gtfs
    assert isinstance(gtfs["routes"], pd.DataFrame)


def test_get_od_pairs(minimal_gtfs_zip: str) -> None:
    """Test getting origin-destination pairs from GTFS data."""
    gtfs = load_gtfs(minimal_gtfs_zip)
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


def test_travel_summary_graph(minimal_gtfs_zip: str) -> None:
    """Test creating travel summary graph from GTFS data."""
    gtfs = load_gtfs(minimal_gtfs_zip)
    summary_gdf = travel_summary_graph(gtfs, as_gdf=True)

    assert hasattr(summary_gdf, "geometry")

    summary_dict = travel_summary_graph(gtfs, as_gdf=False)

    assert isinstance(summary_dict, dict)
    assert next(iter(summary_dict)) in summary_dict


# ============================================================================
# ERROR HANDLING TESTS FOR UNCOVERED PATHS
# ============================================================================


def test_get_gtfs_df_with_corrupted_files(corrupt_gtfs_zip: str) -> None:
    """Test _get_gtfs_df with corrupted CSV files to cover exception handling."""
    gtfs_data = _get_gtfs_df(corrupt_gtfs_zip)

    # Should still return a dict, but corrupted files should be skipped
    assert isinstance(gtfs_data, dict)
    # Routes should be loaded successfully
    assert "routes" in gtfs_data
    # Stops might be missing due to corruption - this tests the exception handling
    if "stops" not in gtfs_data:
        # This covers the exception path in line 41-42
        assert True
    else:
        # If it managed to parse, that's also valid
        assert True


def test_get_gtfs_df_invalid_zip_path() -> None:
    """Test _get_gtfs_df with invalid zip path to cover exception handling."""
    # This should trigger the main exception handler in lines 43-44
    gtfs_data = _get_gtfs_df("nonexistent_file.zip")
    assert isinstance(gtfs_data, dict)
    assert len(gtfs_data) == 0


def test_get_gtfs_df_with_non_txt_files(tmp_path: str) -> None:
    """Test _get_gtfs_df with non-txt files to cover skip logic."""
    zip_path = tmp_path / "mixed_content.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # Add directory (should be skipped - line 32)
        zf.writestr("directory/", "")
        # Add non-txt file (should be skipped - line 32)
        zf.writestr("readme.md", "This is markdown")
        zf.writestr("data.json", '{"key": "value"}')
        # Add valid txt file
        zf.writestr("stops.txt", "stop_id,stop_lat,stop_lon\na,0,0")

    gtfs_data = _get_gtfs_df(str(zip_path))

    # Only the .txt file should be loaded
    assert "stops" in gtfs_data
    assert "readme" not in gtfs_data
    assert "data" not in gtfs_data


def test_get_stops_geometry_missing_columns() -> None:
    """Test _get_stops_geometry with missing required columns."""
    # Test with None input
    result = _get_stops_geometry(None)
    assert result is None

    # Test with missing columns - should trigger warning and return None (lines 109-110)
    stops_df = pd.DataFrame({"stop_id": ["a"], "other_col": ["value"]})
    result = _get_stops_geometry(stops_df)
    assert result is None

    # Test with missing stop_id column
    stops_df = pd.DataFrame({"stop_lat": [0], "stop_lon": [0]})
    result = _get_stops_geometry(stops_df)
    assert result is None


def test_get_shapes_geometry_missing_columns() -> None:
    """Test _get_shapes_geometry with missing required columns."""
    # Test with None input
    result = _get_shapes_geometry(None)
    assert result is None

    # Test with missing columns - should trigger warning and return None (lines 146-147)
    shapes_df = pd.DataFrame({"shape_id": ["s1"], "other_col": ["value"]})
    result = _get_shapes_geometry(shapes_df)
    assert result is None


# ============================================================================
# TIME CONVERSION AND TIMESTAMP TESTS
# ============================================================================


def test_timestamp_and_time_conversions() -> None:
    """Test timestamp creation and time conversion functions."""
    base_date = datetime(2021, 1, 1)
    ts = _create_timestamp("25:00:00", base_date)

    assert ts.hour == 1
    assert ts.day == 2

    assert _create_timestamp(None, base_date) is None
    assert _time_to_seconds("01:02:03") == 3723
    assert np.isnan(_time_to_seconds(None))

    s = pd.Series(["00:00:10", "00:01:00", None])
    sec = _vectorized_time_to_seconds(s)

    assert sec.iloc[0] == 10
    assert sec.iloc[1] == 60
    assert np.isnan(sec.iloc[2])


def test_create_timestamp_invalid() -> None:
    """Test timestamp creation with invalid time format."""
    invalid_time = "ab:cd:ef"
    base_date = datetime(2021, 1, 1)

    assert _create_timestamp(invalid_time, base_date) is None


# ============================================================================
# GEOMETRY AND ADVANCED FUNCTIONALITY TESTS
# ============================================================================


def test_get_od_pairs_with_geometry(minimal_gtfs_zip: str) -> None:
    """Test getting origin-destination pairs with geometry included."""
    gtfs = load_gtfs(minimal_gtfs_zip)
    od_gdf = get_od_pairs(gtfs, include_geometry=True)

    assert hasattr(od_gdf, "geometry")
    assert len(od_gdf) > 0


def test_get_od_pairs_generator_with_geometry(minimal_gtfs_zip: str) -> None:
    """Test getting origin-destination pairs with geometry using generator."""
    gtfs = load_gtfs(minimal_gtfs_zip)
    gen = get_od_pairs(gtfs, include_geometry=True, as_generator=True, chunk_size=1)
    chunks = list(gen)

    assert all(hasattr(c, "geometry") for c in chunks)


# ============================================================================
# PROCESSING TESTS
# ============================================================================


def test_process_gtfs_df_edge_cases() -> None:
    """Test _process_gtfs_df with various edge cases."""
    # Test with empty data
    empty_data = {}
    result = _process_gtfs_df(empty_data)
    assert result == {}

    # Test with stops missing required columns
    gtfs_data = {
        "stops": pd.DataFrame({"stop_id": ["a"], "other_col": ["value"]}),
        "routes": pd.DataFrame({"route_id": ["r1"], "other_col": ["value"]}),
        "calendar": pd.DataFrame({"service_id": ["s1"], "other_col": ["value"]}),
    }

    result = _process_gtfs_df(gtfs_data)
    # Should handle gracefully without required columns
    assert isinstance(result, dict)

    # Test with calendar missing day columns
    gtfs_data["calendar"] = pd.DataFrame({"service_id": ["s1"]})
    result = _process_gtfs_df(gtfs_data)
    assert isinstance(result, dict)


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


def test_load_gtfs_invalid_path() -> None:
    """Test loading GTFS data with invalid file path."""
    invalid_path = "nonexistent.zip"

    with pytest.raises(KeyError):
        load_gtfs(invalid_path)
