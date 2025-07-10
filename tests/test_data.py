"""Tests for the data module - refactored version focusing on public API only."""

from __future__ import annotations

import importlib.util
import subprocess
from typing import TYPE_CHECKING

# Import directly from data module to avoid torch import issues
from unittest.mock import Mock
from unittest.mock import patch

import geopandas as gpd
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point

if TYPE_CHECKING:
    from shapely.geometry import Polygon

spec = importlib.util.spec_from_file_location("data_module", "city2graph/data.py")
assert spec is not None
data_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(data_module)
VALID_OVERTURE_TYPES = data_module.VALID_OVERTURE_TYPES
WGS84_CRS = data_module.WGS84_CRS
load_overture_data = data_module.load_overture_data
process_overture_segments = data_module.process_overture_segments


# Tests for constants and basic functionality
def test_valid_overture_types_constant() -> None:
    """Test that VALID_OVERTURE_TYPES contains expected types."""
    expected_types = {
        "address",
        "bathymetry",
        "building",
        "building_part",
        "division",
        "division_area",
        "division_boundary",
        "place",
        "segment",
        "connector",
        "infrastructure",
        "land",
        "land_cover",
        "land_use",
        "water",
    }
    assert expected_types == VALID_OVERTURE_TYPES


def test_wgs84_crs_constant() -> None:
    """Test that WGS84_CRS is correctly defined."""
    assert WGS84_CRS == "EPSG:4326"


# Tests for load_overture_data function
@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
@patch("city2graph.data.Path.mkdir")
def test_load_overture_data_with_bbox_list(
    mock_mkdir: Mock,
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_bbox: list[float],
) -> None:
    """Test load_overture_data with bounding box as list."""
    # Setup
    types = ["building", "segment"]

    # Mock GeoDataFrame
    mock_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_gdf.empty = False
    mock_read_file.return_value = mock_gdf

    # Execute
    result = load_overture_data(test_bbox, types=types, output_dir="test_dir")

    # Verify
    assert len(result) == 2
    assert "building" in result
    assert "segment" in result
    mock_mkdir.assert_called_once()
    assert mock_subprocess.call_count == 2


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
@patch("city2graph.data.Path.mkdir")
def test_load_overture_data_with_polygon(
    mock_mkdir: Mock,
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_polygon: Polygon,
) -> None:
    """Test load_overture_data with Polygon area."""
    # Setup
    mock_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_gdf.empty = False
    mock_read_file.return_value = mock_gdf

    # Execute
    result = load_overture_data(test_polygon, types=["building"])

    # Verify
    assert "building" in result
    mock_subprocess.assert_called_once()
    mock_mkdir.assert_called()  # Verify directory creation


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
@patch("city2graph.data.gpd.clip")
@patch("city2graph.data.Path.exists")
def test_load_overture_data_with_polygon_clipping(
    mock_exists: Mock,
    mock_clip: Mock,
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_polygon: Polygon,
) -> None:
    """Test that polygon areas are properly clipped."""
    # Setup
    mock_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_gdf.empty = False
    mock_gdf.crs = "EPSG:3857"
    mock_read_file.return_value = mock_gdf
    mock_exists.return_value = True

    mock_clipped_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_clip.return_value = mock_clipped_gdf

    # Execute
    result = load_overture_data(test_polygon, types=["building"])

    # Verify
    mock_clip.assert_called_once()
    mock_subprocess.assert_called()  # Verify subprocess was called
    assert result["building"] == mock_clipped_gdf


def test_load_overture_data_invalid_types(test_bbox: list[float]) -> None:
    """Test that invalid types raise ValueError."""
    invalid_types = ["building", "invalid_type", "another_invalid"]

    with pytest.raises(ValueError, match="Invalid types: \\['invalid_type', 'another_invalid'\\]"):
        load_overture_data(test_bbox, types=invalid_types)


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
def test_load_overture_data_default_types(
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_bbox: list[float],
) -> None:
    """Test that all valid types are used when types=None."""
    mock_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_gdf.empty = False
    mock_read_file.return_value = mock_gdf

    result = load_overture_data(test_bbox, types=None, save_to_file=False)

    # Should have all valid types
    assert len(result) == len(VALID_OVERTURE_TYPES)
    for data_type in VALID_OVERTURE_TYPES:
        assert data_type in result

    # Verify subprocess was called for each type
    assert mock_subprocess.call_count == len(VALID_OVERTURE_TYPES)


@patch("city2graph.data.subprocess.run")
def test_load_overture_data_save_to_file_false(
    mock_subprocess: Mock,
    test_bbox: list[float],
) -> None:
    """Test load_overture_data with save_to_file=False."""
    result = load_overture_data(
        test_bbox,
        types=["building"],
        save_to_file=False,
        return_data=False,
    )

    # Should return empty dict when return_data=False
    assert result == {}

    # Subprocess should be called without output file
    mock_subprocess.assert_called_once()
    args = mock_subprocess.call_args[0][0]
    assert "-o" not in args


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
def test_load_overture_data_return_data_false(
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_bbox: list[float],
) -> None:
    """Test load_overture_data with return_data=False."""
    result = load_overture_data(test_bbox, types=["building"], return_data=False)

    assert result == {}
    mock_read_file.assert_not_called()
    mock_subprocess.assert_called()  # Should still call subprocess to generate files


@patch("city2graph.data.subprocess.run")
def test_load_overture_data_subprocess_error(mock_subprocess: Mock, test_bbox: list[float]) -> None:
    """Test that subprocess errors are propagated."""
    mock_subprocess.side_effect = subprocess.CalledProcessError(1, "overturemaps")

    with pytest.raises(subprocess.CalledProcessError):
        load_overture_data(test_bbox, types=["building"])


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
def test_load_overture_data_with_prefix(
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_bbox: list[float],
) -> None:
    """Test load_overture_data with filename prefix."""
    prefix = "test_prefix_"

    mock_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_gdf.empty = False
    mock_read_file.return_value = mock_gdf

    load_overture_data(test_bbox, types=["building"], prefix=prefix)

    # Check that the output path includes the prefix
    args = mock_subprocess.call_args[0][0]
    output_index = args.index("-o") + 1
    output_path = args[output_index]
    assert prefix in output_path


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
@patch("city2graph.data.Path.exists")
def test_load_overture_data_file_not_exists(
    mock_exists: Mock,
    mock_read_file: Mock,
    mock_subprocess: Mock,
    test_bbox: list[float],
) -> None:
    """Test behavior when output file doesn't exist."""
    mock_exists.return_value = False

    result = load_overture_data(test_bbox, types=["building"])

    # Should return empty GeoDataFrame when file doesn't exist
    mock_read_file.assert_not_called()
    assert "building" in result
    mock_subprocess.assert_called()  # Should still attempt to generate files


# Tests for process_overture_segments function
def test_process_overture_segments_empty_input(data_empty_gdf: gpd.GeoDataFrame) -> None:
    """Test process_overture_segments with empty GeoDataFrame."""
    result = process_overture_segments(data_empty_gdf)

    assert result.empty
    assert result.crs == WGS84_CRS


def test_process_overture_segments_basic(data_sample_segments_gdf: gpd.GeoDataFrame) -> None:
    """Test basic functionality of process_overture_segments."""
    result = process_overture_segments(data_sample_segments_gdf, get_barriers=False)

    # Should have length column
    assert "length" in result.columns
    assert all(result["length"] > 0)

    # Should preserve original data
    assert len(result) >= len(data_sample_segments_gdf)
    assert "id" in result.columns


def test_process_overture_segments_with_connectors(
    data_sample_segments_gdf: gpd.GeoDataFrame,
    data_sample_connectors_gdf: gpd.GeoDataFrame,
) -> None:
    """Test process_overture_segments with connectors."""
    result = process_overture_segments(
        data_sample_segments_gdf,
        connectors_gdf=data_sample_connectors_gdf,
        get_barriers=False,
    )

    # Should have split columns for segments that were split
    split_segments = result[result["id"].str.contains("_")]
    if not split_segments.empty:
        assert "split_from" in result.columns
        assert "split_to" in result.columns


def test_process_overture_segments_with_barriers(
    data_sample_segments_gdf: gpd.GeoDataFrame,
) -> None:
    """Test process_overture_segments with barrier generation."""
    result = process_overture_segments(data_sample_segments_gdf, get_barriers=True)

    # Should have barrier_geometry column
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_missing_level_rules() -> None:
    """Test process_overture_segments with missing level_rules column."""
    geometries = [LineString([(0, 0), (1, 1)])]
    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    # This should work - the function should handle missing level_rules gracefully
    result = process_overture_segments(segments_gdf)
    assert "level_rules" in result.columns
    assert result["level_rules"].iloc[0] == ""


def test_process_overture_segments_with_threshold(
    data_sample_segments_gdf: gpd.GeoDataFrame,
    data_sample_connectors_gdf: gpd.GeoDataFrame,
) -> None:
    """Test process_overture_segments with custom threshold."""
    result = process_overture_segments(
        data_sample_segments_gdf,
        connectors_gdf=data_sample_connectors_gdf,
        threshold=2.0,
    )

    # Should process without errors
    assert "length" in result.columns


def test_process_overture_segments_no_connectors(
    data_sample_segments_gdf: gpd.GeoDataFrame,
) -> None:
    """Test process_overture_segments with None connectors."""
    result = process_overture_segments(data_sample_segments_gdf, connectors_gdf=None)

    # Should not perform endpoint clustering
    assert len(result) == len(data_sample_segments_gdf)


def test_process_overture_segments_empty_connectors(
    data_sample_segments_gdf: gpd.GeoDataFrame,
    data_empty_gdf: gpd.GeoDataFrame,
) -> None:
    """Test process_overture_segments with empty connectors GeoDataFrame."""
    result = process_overture_segments(data_sample_segments_gdf, connectors_gdf=data_empty_gdf)

    # Should not perform splitting or clustering
    assert len(result) == len(data_sample_segments_gdf)


def test_process_overture_segments_invalid_connector_data() -> None:
    """Test process_overture_segments with invalid connector JSON."""
    geometries = [LineString([(0, 0), (1, 1)])]
    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "connectors": ["invalid_json"],
            "level_rules": [""],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf)

    # Should handle invalid JSON gracefully
    assert len(result) == 1


def test_process_overture_segments_malformed_connectors(
    data_sample_connectors_gdf: gpd.GeoDataFrame,
) -> None:
    """Test process_overture_segments with malformed connector data."""
    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "connectors": ['{"invalid": "structure"}'],
            "level_rules": [""],
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        crs=WGS84_CRS,
    )
    result = process_overture_segments(segments_gdf, connectors_gdf=data_sample_connectors_gdf)

    # Should handle malformed data gracefully
    assert len(result) == 1


def test_process_overture_segments_invalid_level_rules() -> None:
    """Test process_overture_segments with invalid level rules JSON."""
    geometries = [LineString([(0, 0), (1, 1)])]
    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "level_rules": ["invalid_json"],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)

    # Should handle invalid JSON gracefully
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_complex_level_rules() -> None:
    """Test process_overture_segments with complex level rules."""
    geometries = [LineString([(0, 0), (1, 1)])]
    level_rules = '[{"value": 1, "between": [0.1, 0.3]}, {"value": 1, "between": [0.7, 0.9]}]'

    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "level_rules": [level_rules],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)

    # Should create barrier geometry
    assert "barrier_geometry" in result.columns
    assert result["barrier_geometry"].iloc[0] is not None


def test_process_overture_segments_full_barrier() -> None:
    """Test process_overture_segments with full barrier level rules."""
    geometries = [LineString([(0, 0), (1, 1)])]
    level_rules = '[{"value": 1}]'  # No "between" means full barrier

    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "level_rules": [level_rules],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)

    # Should create None barrier geometry for full barriers
    assert "barrier_geometry" in result.columns
    assert result["barrier_geometry"].iloc[0] is None


def test_process_overture_segments_zero_value_rules() -> None:
    """Test process_overture_segments with zero value level rules."""
    geometries = [LineString([(0, 0), (1, 1)])]
    level_rules = '[{"value": 0, "between": [0.2, 0.8]}]'

    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "level_rules": [level_rules],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)

    # Zero value rules should be ignored
    assert "barrier_geometry" in result.columns
    # Should return original geometry since no barriers
    barrier_geom = result["barrier_geometry"].iloc[0]
    assert barrier_geom is not None


def test_process_overture_segments_segment_splitting() -> None:
    """Test that segments are properly split at connector positions."""
    geometries = [LineString([(0, 0), (2, 2)])]
    connectors_data = '[{"connector_id": "conn1", "at": 0.0}, {"connector_id": "conn2", "at": 0.5}, {"connector_id": "conn3", "at": 1.0}]'

    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "connectors": [connectors_data],
            "level_rules": [""],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    connectors_gdf = gpd.GeoDataFrame(
        {
            "id": ["conn1", "conn2", "conn3"],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, connectors_gdf=connectors_gdf)

    # Should create multiple segments
    assert len(result) > 1
    # Should have split information
    split_segments = result[result["id"].str.contains("_")]
    assert not split_segments.empty


def test_process_overture_segments_endpoint_clustering() -> None:
    """Test endpoint clustering functionality."""
    # Create segments with nearby endpoints
    geometries = [
        LineString([(0, 0), (1, 1)]),
        LineString([(1.1, 1.1), (2, 2)]),
    ]

    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1", "seg2"],
            "level_rules": ["", ""],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    connectors_gdf = gpd.GeoDataFrame(
        {
            "id": ["conn1"],
            "geometry": [Point(1, 1)],
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(
        segments_gdf,
        connectors_gdf=connectors_gdf,
        threshold=0.5,  # Large enough to cluster nearby points
    )

    # Should process without errors
    assert len(result) >= len(segments_gdf)


def test_process_overture_segments_level_rules_handling() -> None:
    """Test process_overture_segments level_rules column handling."""
    # Test with None values in level_rules
    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "level_rules": [None],
            "geometry": [LineString([(0, 0), (1, 1)])],
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf)
    assert result["level_rules"].iloc[0] == ""


# Integration tests
def test_load_and_process_integration() -> None:
    """Test integration between load_overture_data and process_overture_segments."""
    # Create mock data that resembles real Overture data
    segments_data = {
        "id": ["seg1", "seg2"],
        "connectors": [
            '[{"connector_id": "conn1", "at": 0.0}]',
            '[{"connector_id": "conn2", "at": 1.0}]',
        ],
        "level_rules": ["", ""],
        "geometry": [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (2, 2)]),
        ],
    }

    segments_gdf = gpd.GeoDataFrame(segments_data, crs=WGS84_CRS)

    # Process the segments
    result = process_overture_segments(segments_gdf)

    # Should have all expected columns
    expected_columns = ["id", "connectors", "level_rules", "geometry", "length", "barrier_geometry"]
    for col in expected_columns:
        assert col in result.columns


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
@patch("city2graph.data.Path.exists")
def test_real_world_scenario_simulation(
    mock_exists: Mock,
    mock_read_file: Mock,
    mock_subprocess: Mock,
    realistic_segments_gdf: gpd.GeoDataFrame,
    realistic_connectors_gdf: gpd.GeoDataFrame,
    test_bbox: list[float],
) -> None:
    """Test a scenario that simulates real-world usage."""
    # Mock the file reading to return realistic data
    mock_read_file.side_effect = [realistic_segments_gdf, realistic_connectors_gdf]
    mock_exists.return_value = True

    # Simulate loading data
    data = load_overture_data(test_bbox, types=["segment", "connector"])

    # Process the segments
    processed_segments = process_overture_segments(
        data["segment"],
        connectors_gdf=data["connector"],
    )  # Verify the result
    assert not processed_segments.empty
    assert "barrier_geometry" in processed_segments.columns

    # Verify mocks were called appropriately
    mock_subprocess.assert_called()  # Should be called for data loading
    assert "length" in processed_segments.columns


# Additional edge case tests for comprehensive coverage
def test_process_overture_segments_with_non_dict_connector() -> None:
    """Test process_overture_segments with non-dict connector data."""
    geometries = [LineString([(0, 0), (1, 1)])]
    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "connectors": ['["not_a_dict"]'],
            "level_rules": [""],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf)
    # Should handle non-dict data gracefully
    assert len(result) == 1


def test_process_overture_segments_with_non_dict_level_rule() -> None:
    """Test process_overture_segments with non-dict level rule data."""
    geometries = [LineString([(0, 0), (1, 1)])]
    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "level_rules": ['["not_a_dict"]'],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)
    # Should handle non-dict data gracefully
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_with_short_between_array() -> None:
    """Test process_overture_segments with short between array in level rules."""
    geometries = [LineString([(0, 0), (1, 1)])]
    level_rules = '[{"value": 1, "between": [0.5]}]'  # Only one element

    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "level_rules": [level_rules],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)
    # Should handle short between array gracefully
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_with_empty_geometry() -> None:
    """Test process_overture_segments with empty geometry."""
    # Create an empty LineString
    empty_geom = LineString()
    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "level_rules": [""],
            "geometry": [empty_geom],
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)
    # Should handle empty geometry gracefully
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_with_overlapping_barriers() -> None:
    """Test process_overture_segments with overlapping barrier intervals."""
    geometries = [LineString([(0, 0), (4, 4)])]
    level_rules = '[{"value": 1, "between": [0.1, 0.5]}, {"value": 1, "between": [0.3, 0.7]}]'

    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "level_rules": [level_rules],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)
    # Should handle overlapping barriers correctly
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_with_touching_barriers() -> None:
    """Test process_overture_segments with touching barrier intervals."""
    geometries = [LineString([(0, 0), (4, 4)])]
    level_rules = '[{"value": 1, "between": [0.0, 0.3]}, {"value": 1, "between": [0.3, 0.6]}]'

    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "level_rules": [level_rules],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)
    # Should handle touching barriers correctly
    assert "barrier_geometry" in result.columns


def test_process_overture_segments_with_full_coverage_barriers() -> None:
    """Test process_overture_segments with barriers covering full segment."""
    geometries = [LineString([(0, 0), (4, 4)])]
    level_rules = '[{"value": 1, "between": [0.0, 1.0]}]'

    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1"],
            "level_rules": [level_rules],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(segments_gdf, get_barriers=True)
    # Should return None for full coverage barriers
    assert "barrier_geometry" in result.columns
    assert result["barrier_geometry"].iloc[0] is None


@patch("city2graph.data.subprocess.run")
@patch("city2graph.data.gpd.read_file")
@patch("city2graph.data.Path.mkdir")
def test_load_overture_data_comprehensive_all_types(
    mock_mkdir: Mock,
    mock_read_file: Mock,
    mock_subprocess: Mock,
) -> None:
    """Test load_overture_data with all types (types=None)."""
    # mock_mkdir is set up by @patch decorator but not called in this test with save_to_file=False
    _ = mock_mkdir  # Acknowledge the parameter

    # Mock GeoDataFrame
    mock_gdf = Mock(spec=gpd.GeoDataFrame)
    mock_gdf.empty = False
    mock_read_file.return_value = mock_gdf

    # Test with all types (types=None)
    bbox = [-74.01, 40.70, -73.99, 40.72]
    result = load_overture_data(bbox, types=None, save_to_file=False)

    # Should call subprocess for all valid types
    assert mock_subprocess.call_count == len(VALID_OVERTURE_TYPES)
    assert len(result) == len(VALID_OVERTURE_TYPES)


def test_process_overture_segments_with_non_linestring_endpoints() -> None:
    """Test endpoint clustering with non-LineString geometries."""
    # Mix LineString and Point geometries
    geometries = [
        LineString([(0, 0), (1, 1)]),
        Point(2, 2),  # This should be ignored in endpoint clustering
    ]

    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1", "seg2"],
            "level_rules": ["", ""],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    connectors_gdf = gpd.GeoDataFrame(
        {
            "id": ["conn1"],
            "geometry": [Point(1, 1)],
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(
        segments_gdf,
        connectors_gdf=connectors_gdf,
        threshold=1.0,
    )

    # Should process without errors
    assert len(result) == len(segments_gdf)


def test_process_overture_segments_with_short_linestring() -> None:
    """Test endpoint clustering with LineString having insufficient coordinates."""
    # Create a degenerate LineString (same start and end point)
    invalid_geom = LineString([(0, 0), (0, 0)])  # Degenerate but valid

    geometries = [
        LineString([(0, 0), (1, 1)]),
        invalid_geom,
    ]

    segments_gdf = gpd.GeoDataFrame(
        {
            "id": ["seg1", "seg2"],
            "level_rules": ["", ""],
            "geometry": geometries,
        },
        crs=WGS84_CRS,
    )

    connectors_gdf = gpd.GeoDataFrame(
        {
            "id": ["conn1"],
            "geometry": [Point(1, 1)],
        },
        crs=WGS84_CRS,
    )

    result = process_overture_segments(
        segments_gdf,
        connectors_gdf=connectors_gdf,
        threshold=1.0,
    )

    # Should process without errors
    assert len(result) == len(segments_gdf)
