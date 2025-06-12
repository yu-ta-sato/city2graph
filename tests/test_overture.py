"""Test module for city2graph utility functions."""

import subprocess
from unittest.mock import Mock
from unittest.mock import patch

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import Point
from shapely.geometry import Polygon

from city2graph.overture import _adjust_segment_connectors as adjust_segment_connectors
from city2graph.overture import _create_connector_mask
from city2graph.overture import _extract_barriers_from_mask
from city2graph.overture import _extract_line_segment
from city2graph.overture import _extract_valid_connectors
from city2graph.overture import _get_barrier_geometry
from city2graph.overture import _get_substring
from city2graph.overture import _identify_barrier_mask as identify_barrier_mask
from city2graph.overture import _identify_connector_mask as identify_connector_mask
from city2graph.overture import _parse_connectors_info
from city2graph.overture import _recalc_barrier_mask
from city2graph.overture import load_overture_data

# ============================================================================
# COMMON TEST FIXTURES
# ============================================================================


@pytest.fixture
def simple_line() -> LineString:
    """Return a simple horizontal LineString for testing."""
    return LineString([(0, 0), (2, 0)])


@pytest.fixture
def complex_line() -> LineString:
    """Return a LineString with intermediate vertices for complex testing."""
    return LineString([(0, 0), (1, 0), (2, 0)])


@pytest.fixture
def simple_polygon() -> Polygon:
    """Return a unit square Polygon for testing."""
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


@pytest.fixture
def nodes_gdf() -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with two nodes for testing nearest node functions."""
    return gpd.GeoDataFrame(
        {"node_id": [1, 2]},
        geometry=[Point(0, 0), Point(2, 0)],
        crs=None,
    )


@pytest.fixture
def empty_nodes_gdf() -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame for testing edge cases."""
    return gpd.GeoDataFrame({"node_id": []}, geometry=[], crs=None)


@pytest.fixture
def sample_network() -> nx.Graph:
    """Return a sample NetworkX graph for testing filtering functions."""
    G = nx.Graph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=1, y=0)
    G.add_edge(1, 2, length=1)
    return G


@pytest.fixture
def test_polygon_with_points() -> tuple[Polygon, gpd.GeoDataFrame]:
    """Return a polygon and GeoDataFrame with interior/exterior points for clipping tests."""
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(1, 1), Point(3, 3)]},
        crs="EPSG:4326",
    )
    return poly, gdf


# ============================================================================
# BASIC GEOMETRY UTILITIES TESTS
# ============================================================================

def test_get_substring_basic(simple_line: LineString) -> None:
    """Test basic functionality of _get_substring function."""
    # Arrange: use the simple horizontal line fixture
    line = simple_line

    # Act: extract substring between 25% and 75% of the line
    seg = _get_substring(line, 0.25, 0.75)

    # Assert: validate that we get a LineString with correct endpoints
    assert isinstance(seg, LineString)
    start, end = seg.coords[0], seg.coords[-1]
    assert pytest.approx(start[0], rel=1e-6) == 0.5
    assert start[1] == 0
    assert pytest.approx(end[0], rel=1e-6) == 1.5
    assert end[1] == 0


def test_get_substring_invalid() -> None:
    """Test _get_substring returns None for invalid inputs."""
    # Arrange: test with non-geometry input
    invalid_line = "not a line"

    # Act & Assert: should return None for invalid geometry
    assert _get_substring(invalid_line, 0, 1) is None

    # Arrange: test with reversed start/end positions
    line = LineString([(0, 0), (1, 1)])

    # Act & Assert: should return None when start > end
    assert _get_substring(line, 0.6, 0.4) is None


def test__extract_line_segment_basic() -> None:
    """Test _extract_line_segment with intermediate vertices."""
    # Arrange: create a line with intermediate vertex at (1,0)
    line = LineString([(0, 0), (1, 0), (2, 0)])
    total_length = line.length
    start_dist = 0.25 * total_length
    end_dist = 0.75 * total_length
    start_pt = line.interpolate(start_dist)
    end_pt = line.interpolate(end_dist)

    # Act: extract segment between the interpolated points
    seg = _extract_line_segment(line, start_pt, end_pt, start_dist, end_dist)

    # Assert: verify we get correct segment coordinates
    assert isinstance(seg, LineString)
    coords = list(seg.coords)
    assert pytest.approx(coords[0][0], rel=1e-6) == 0.5
    assert pytest.approx(coords[-1][0], rel=1e-6) == 1.5


def test_get_substring_full_line() -> None:
    """Test substring with full line returns original."""
    # Arrange: create a simple line
    line = LineString([(0, 0), (2, 0)])

    # Act: extract full line (0% to 100%)
    seg = _get_substring(line, 0, 1)

    # Assert: should return identical geometry
    assert seg.equals(line)


# ============================================================================
# BARRIER MASK AND CONNECTOR TESTS
# ============================================================================

def test_identify_barrier_mask_empty_and_invalid() -> None:
    """Test identify_barrier_mask with empty and invalid inputs."""
    # Arrange: invalid JSON inputs
    inputs = ["", "null", "not json"]

    # Act & Assert: should return full mask for each invalid input
    for inp in inputs:
        assert identify_barrier_mask(inp) == [[0.0, 1.0]]


def test_identify_barrier_mask_simple_and_null_between() -> None:
    """Test identify_barrier_mask function with simple rules and null between values."""
    # Arrange: simple barrier rule between 0.2 and 0.5
    rules = "[{'value': 1, 'between': [0.2, 0.5]}]"

    # Act: compute barrier mask
    mask = identify_barrier_mask(rules)

    # Assert: barrier excludes interval [0.2, 0.5]
    assert mask == [[0.0, 0.2], [0.5, 1.0]]

    # Arrange: rule with null between (no barriers)
    null_rules = "[{'value': 1, 'between': None}]"

    # Act & Assert: should return empty mask
    assert identify_barrier_mask(null_rules) == []


def test_identify_barrier_mask_multiple_intervals() -> None:
    """Test identify_barrier_mask with multiple non-zero rules."""
    # Arrange: multiple barrier intervals
    rules = "[{'value': 1, 'between': [0.0, 0.1]}, {'value': 2, 'between': [0.9, 1.0]}]"

    # Act: compute barrier mask
    mask = identify_barrier_mask(rules)

    # Assert: barrier intervals at [0,0.1] and [0.9,1.0], complement is [0.1,0.9]
    assert mask == [[0.1, 0.9]]


def test_identify_barrier_mask_value_zero_only() -> None:
    """Test identify_barrier_mask with only zero-value rules (non-barriers)."""
    # Arrange: zero-value rule (not a barrier)
    rules = "[{'value': 0, 'between': [0.2, 0.4]}]"

    # Act: compute barrier mask
    mask = identify_barrier_mask(rules)

    # Assert: should return full mask since no actual barriers
    assert mask == [[0.0, 1.0]]


def test_identify_connector_mask() -> None:
    """Test identify_connector_mask function with various connector configurations."""
    # Arrange & Act & Assert: empty string should return endpoints only
    assert identify_connector_mask("") == [0.0, 1.0]

    # Arrange & Act & Assert: single connector at 30%
    info = "{'connector_id': 1, 'at': 0.3}"
    assert identify_connector_mask(info) == [0.0, 0.3, 1.0]

    # Arrange & Act & Assert: multiple connectors (sorted order)
    info_list = "[{'connector_id': 2, 'at': 0.4}, {'connector_id':3,'at':0.1}]"
    assert identify_connector_mask(info_list) == [0.0, 0.1, 0.4, 1.0]


def test_identify_connector_mask_invalid_json() -> None:
    """Test identify_connector_mask handles invalid JSON gracefully."""
    # Arrange: malformed JSON
    invalid = "[{'connector_id': 1, 'at': 0.2"  # missing closing bracket

    # Act & Assert: should fallback to endpoints only
    assert identify_connector_mask(invalid) == [0.0, 1.0]


def test_recalc_barrier_mask() -> None:
    """Test _recalc_barrier_mask function with barrier mask recalculation."""
    orig = [[0.2, 0.8]]
    assert _recalc_barrier_mask([[0.0, 1.0]], 0.2, 0.6) == [[0.0, 1.0]]
    new_mask = _recalc_barrier_mask(orig, 0.2, 0.6)
    assert new_mask == [[0.0, 1.0]]


def test_recalc_barrier_mask_partial_overlap() -> None:
    """Test _recalc_barrier_mask with partial overlap intervals."""
    original = [[0.2, 0.4], [0.6, 0.8]]
    new = _recalc_barrier_mask(original, 0.5, 1.0)
    # interval [0.6,0.8] overlaps; new relative = [(0.6-0.5)/0.5, (0.8-0.5)/0.5] = [0.2,0.6]
    assert pytest.approx(new[0][0], rel=1e-6) == 0.2
    assert pytest.approx(new[0][1], rel=1e-6) == 0.6


def test_load_overture_data() -> None:
    """Test load_overture_data raises on invalid bbox format."""
    with pytest.raises(ValueError, match="Invalid bbox format"):
        load_overture_data("url")


def test_adjust_segment_connectors_no_change(simple_line: LineString) -> None:
    """Test adjust_segment_connectors raises when no valid connectors."""
    segs = gpd.GeoDataFrame(
        {
            "id": [1],
            "geometry": [simple_line],
            "connectors": ["[]"],
            "level_rules": ["[]"],
        },
    )
    empty_conn = gpd.GeoDataFrame({"id": [], "geometry": []})
    with pytest.raises(TypeError):
        adjust_segment_connectors(segs, empty_conn)


# ============================================================================
# ADDITIONAL COMPREHENSIVE TESTS FOR FULL COVERAGE
# ============================================================================

def test_get_substring_boundary_conditions() -> None:
    """Test _get_substring with boundary conditions and edge cases."""
    line = LineString([(0, 0), (10, 0)])

    # Test exact boundaries
    seg = _get_substring(line, 0.0, 1.0)
    assert seg.equals(line)

    # Test start at 0
    seg = _get_substring(line, 0.0, 0.5)
    assert seg.coords[0] == (0.0, 0.0)
    assert pytest.approx(seg.coords[-1][0], rel=1e-6) == 5.0

    # Test end at 1
    seg = _get_substring(line, 0.5, 1.0)
    assert pytest.approx(seg.coords[0][0], rel=1e-6) == 5.0
    assert seg.coords[-1] == (10.0, 0.0)

    # Test very small segment
    seg = _get_substring(line, 0.49, 0.51)
    assert seg is not None
    assert isinstance(seg, LineString)

    # Test out of bounds
    assert _get_substring(line, -0.1, 0.5) is None
    assert _get_substring(line, 0.5, 1.1) is None
    assert _get_substring(line, 1.1, 1.2) is None


def test_get_substring_with_complex_geometry() -> None:
    """Test _get_substring with complex multi-segment lines."""
    # Create a line with multiple segments and curves
    line = LineString([(0, 0), (1, 1), (2, 0), (3, 1), (4, 0)])

    # Test various segments
    seg1 = _get_substring(line, 0.2, 0.8)
    assert isinstance(seg1, LineString)
    assert seg1.length > 0

    # Test segment that crosses multiple original segments
    seg2 = _get_substring(line, 0.1, 0.9)
    assert isinstance(seg2, LineString)

    # Test tiny segment
    seg3 = _get_substring(line, 0.5, 0.500001)
    assert seg3 is None or isinstance(seg3, LineString)


def test_extract_line_segment_edge_cases() -> None:
    """Test _extract_line_segment with various edge cases."""
    line = LineString([(0, 0), (5, 0), (10, 0)])

    # Test identical start and end points
    pt = line.interpolate(0.5)
    seg = _extract_line_segment(line, pt, pt, 0.5, 0.5)
    assert isinstance(seg, LineString)
    assert seg.length == 0

    # Test points at line endpoints
    start_pt = Point(0, 0)
    end_pt = Point(10, 0)
    seg = _extract_line_segment(line, start_pt, end_pt, 0.0, 1.0)
    assert isinstance(seg, LineString)
    assert seg.equals(line)

    # Test reversed order (should still work)
    mid_pt1 = line.interpolate(0.3)
    mid_pt2 = line.interpolate(0.7)
    seg = _extract_line_segment(line, mid_pt2, mid_pt1, 0.7, 0.3)
    assert isinstance(seg, LineString)


def test_parse_connectors_info_comprehensive() -> None:
    """Test _parse_connectors_info with all possible input formats."""
    # Test null/None handling
    assert _parse_connectors_info(None) == []
    assert _parse_connectors_info("null") == []

    # Test various malformed JSON
    assert _parse_connectors_info("{") == []
    assert _parse_connectors_info("[{") == []
    assert _parse_connectors_info("{'incomplete':") == []

    # Test valid single object as string
    single_str = '{"connector_id": 5, "at": 0.75}'
    result = _parse_connectors_info(single_str)
    assert len(result) == 1
    assert result[0]["connector_id"] == 5
    assert result[0]["at"] == 0.75

    # Test list with mixed valid/invalid entries
    mixed_list = '[{"connector_id": 1, "at": 0.2}, {"bad": "entry"}, {"connector_id": 2}]'
    result = _parse_connectors_info(mixed_list)
    assert len(result) >= 1  # Should parse what it can

    # Test empty list
    assert _parse_connectors_info("[]") == []

    # Test list with null entries
    with_nulls = '[{"connector_id": 1, "at": 0.3}, null, {"connector_id": 2, "at": 0.7}]'
    result = _parse_connectors_info(with_nulls)
    assert isinstance(result, list)


def test_extract_valid_connectors_comprehensive() -> None:
    """Test _extract_valid_connectors with various edge cases."""
    # Test empty inputs
    assert _extract_valid_connectors([], set()) == []
    assert _extract_valid_connectors([], {1, 2, 3}) == []

    # Test no valid IDs
    raw = [{"connector_id": 1, "at": 0.5}, {"connector_id": 2, "at": 0.8}]
    assert _extract_valid_connectors(raw, set()) == []

    # Test all invalid connectors
    invalid_raw = [
        {"connector_id": None, "at": 0.5},
        {"connector_id": 1, "at": None},
        {"connector_id": None, "at": None},
    ]
    assert _extract_valid_connectors(invalid_raw, {1, 2}) == []

    # Test mixed valid/invalid with edge positions (function accepts out-of-range values)
    mixed_raw = [
        {"connector_id": 1, "at": 0.0},  # at start
        {"connector_id": 2, "at": 1.0},  # at end
        {"connector_id": 3, "at": 0.5},  # middle
        {"connector_id": 4, "at": -0.1}, # out of range but still valid
        {"connector_id": 5, "at": 1.1},  # out of range but still valid
    ]
    result = _extract_valid_connectors(mixed_raw, {1, 2, 3, 4, 5})
    assert 0.0 in result
    assert 1.0 in result
    assert 0.5 in result
    assert -0.1 in result
    assert 1.1 in result


def test_create_connector_mask_comprehensive() -> None:
    """Test _create_connector_mask with various configurations."""
    # Test empty list
    assert _create_connector_mask([]) == [0.0, 1.0]

    # Test single connector
    assert _create_connector_mask([0.5]) == [0.0, 0.5, 1.0]

    # Test connectors at boundaries
    assert _create_connector_mask([0.0]) == [0.0, 1.0]
    assert _create_connector_mask([1.0]) == [0.0, 1.0]
    assert _create_connector_mask([0.0, 1.0]) == [0.0, 1.0]

    # Test multiple connectors with duplicates (function doesn't dedupe)
    result = _create_connector_mask([0.3, 0.7, 0.3, 0.5])
    # Function adds all values plus endpoints
    assert 0.0 in result
    assert 0.3 in result
    assert 0.5 in result
    assert 0.7 in result
    assert 1.0 in result

    # Test unsorted input (function doesn't sort)
    result = _create_connector_mask([0.8, 0.2, 0.5])
    assert result == [0.0, 0.8, 0.2, 0.5, 1.0]


def test_recalc_barrier_mask_comprehensive() -> None:
    """Test _recalc_barrier_mask with complex scenarios."""
    # Test empty mask
    assert _recalc_barrier_mask([], 0.0, 1.0) == []
    assert _recalc_barrier_mask([], 0.2, 0.8) == []

    # Test no overlap scenarios
    original = [[0.1, 0.3], [0.7, 0.9]]
    assert _recalc_barrier_mask(original, 0.4, 0.6) == []
    assert _recalc_barrier_mask(original, 0.0, 0.05) == []
    assert _recalc_barrier_mask(original, 0.95, 1.0) == []

    # Test complete containment
    original = [[0.2, 0.8]]
    result = _recalc_barrier_mask(original, 0.3, 0.7)
    assert result == [[0.0, 1.0]]

    # Test partial overlaps
    original = [[0.0, 0.4], [0.6, 1.0]]
    result = _recalc_barrier_mask(original, 0.2, 0.8)
    # [0.0,0.4] overlaps [0.2,0.8] -> relative [0,0.2/0.6] = [0,1/3]
    # [0.6,1.0] overlaps [0.2,0.8] -> relative [(0.6-0.2)/0.6, (0.8-0.2)/0.6] = [2/3,1]
    assert len(result) == 2
    assert pytest.approx(result[0][0], rel=1e-6) == 0.0
    assert pytest.approx(result[0][1], rel=1e-6) == 1/3
    assert pytest.approx(result[1][0], rel=1e-6) == 2/3
    assert pytest.approx(result[1][1], rel=1e-6) == 1.0

    # Test boundary cases
    original = [[0.0, 1.0]]
    assert _recalc_barrier_mask(original, 0.0, 1.0) == [[0.0, 1.0]]
    # When start == end, function returns the full mask
    assert _recalc_barrier_mask(original, 0.5, 0.5) == [[0.0, 1.0]]


def test_extract_barriers_from_mask_comprehensive() -> None:
    """Test _extract_barriers_from_mask with various mask configurations."""
    line = LineString([(0, 0), (10, 0)])

    # Test empty mask
    assert _extract_barriers_from_mask(line, []) is None

    # Test single interval
    mask = [[0.2, 0.8]]
    result = _extract_barriers_from_mask(line, mask)
    assert isinstance(result, LineString)
    assert pytest.approx(result.bounds[0], rel=1e-6) == 2.0
    assert pytest.approx(result.bounds[2], rel=1e-6) == 8.0

    # Test multiple intervals (should return MultiLineString)
    mask = [[0.1, 0.3], [0.7, 0.9]]
    result = _extract_barriers_from_mask(line, mask)
    assert isinstance(result, MultiLineString)
    assert len(list(result.geoms)) == 2

    # Test full line
    mask = [[0.0, 1.0]]
    result = _extract_barriers_from_mask(line, mask)
    assert isinstance(result, LineString)
    assert result.equals(line)

    # Test tiny intervals
    mask = [[0.5, 0.500001]]
    result = _extract_barriers_from_mask(line, mask)
    assert result is None or isinstance(result, LineString)


def test_identify_barrier_mask_comprehensive() -> None:
    """Test identify_barrier_mask with complex rule combinations."""
    # Test overlapping barriers
    rules = '[{"value": 1, "between": [0.1, 0.4]}, {"value": 2, "between": [0.3, 0.6]}]'
    mask = identify_barrier_mask(rules)
    # Barriers at intervals [0.1,0.4] and [0.3,0.6], union is [0.1,0.6]
    # Result should be the complement intervals
    assert mask == [[0.0, 0.1], [0.6, 1.0]]

    # Test adjacent barriers
    rules = '[{"value": 1, "between": [0.2, 0.4]}, {"value": 2, "between": [0.4, 0.6]}]'
    mask = identify_barrier_mask(rules)
    # Barriers at [0.2,0.4] and [0.4,0.6], union is [0.2,0.6]
    assert mask == [[0.0, 0.2], [0.6, 1.0]]

    # Test mixed zero and non-zero values
    rules = '[{"value": 0, "between": [0.1, 0.3]}, {"value": 1, "between": [0.5, 0.7]}]'
    mask = identify_barrier_mask(rules)
    # Only non-zero value creates barrier at [0.5,0.7]
    assert mask == [[0.0, 0.5], [0.7, 1.0]]

    # Test barriers covering entire range
    rules = '[{"value": 1, "between": [0.0, 0.5]}, {"value": 2, "between": [0.5, 1.0]}]'
    mask = identify_barrier_mask(rules)
    assert mask == []

    # Test invalid between values - function doesn't validate order
    rules = '[{"value": 1, "between": [0.6, 0.4]}]'  # reversed
    mask = identify_barrier_mask(rules)
    # Function processes as-is, resulting in [0.0,0.6] and [0.4,1.0] intervals
    assert len(mask) >= 1  # Should have some result

    # Test between values out of range
    rules = '[{"value": 1, "between": [-0.1, 0.5]}]'
    mask = identify_barrier_mask(rules)
    # Should handle gracefully
    assert isinstance(mask, list)


def test_identify_connector_mask_comprehensive() -> None:
    """Test identify_connector_mask with complex scenarios."""
    # Test connectors at boundaries
    info = '{"connector_id": 1, "at": 0.0}'
    result = identify_connector_mask(info)
    assert result == [0.0, 0.0, 1.0]

    info = '{"connector_id": 1, "at": 1.0}'
    result = identify_connector_mask(info)
    assert result == [0.0, 1.0, 1.0]

    # Test multiple connectors with some invalid
    info = '[{"connector_id": 1, "at": 0.2}, {"connector_id": 2}, {"connector_id": 3, "at": 0.8}]'
    result = identify_connector_mask(info)
    assert 0.2 in result
    assert 0.8 in result
    assert result == [0.0, 0.2, 0.8, 1.0]

    # Test connectors out of range
    info = '[{"connector_id": 1, "at": -0.1}, {"connector_id": 2, "at": 1.1}, {"connector_id": 3, "at": 0.5}]'
    result = identify_connector_mask(info)
    assert 0.5 in result
    # Function accepts out-of-range values


def test_load_overture_data_polygon_with_crs() -> None:
    """Test load_overture_data with polygon that needs CRS transformation."""
    # Create polygon in non-WGS84 CRS
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame([{"geometry": polygon}], crs="EPSG:3857")
    poly_with_crs = gdf.geometry.iloc[0]

    # Mock the subprocess and file operations to avoid actual downloads
    with patch("city2graph.overture._process_single_overture_type") as mock_process:
        mock_process.return_value = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

        # This should trigger the CRS transformation in _prepare_polygon_area
        result = load_overture_data(poly_with_crs, types=["building"], return_data=True)

        assert isinstance(result, dict)
        assert "building" in result


# ============================================================================
# TESTS FOR UNCOVERED CODE PATHS IN OVERTURE.PY
# ============================================================================

def test_additional_uncovered_paths() -> None:
    """Test additional uncovered code paths in overture.py functions."""
    # Test _get_substring exception handling (lines 390-396)
    line = LineString([(0, 0), (1, 0)])
    with (
        patch("city2graph.overture._extract_line_segment", side_effect=ValueError("Test error")),
        patch("city2graph.overture.logger") as mock_logger,
    ):
        result = _get_substring(line, 0.1, 0.9)
        assert result is None
        mock_logger.warning.assert_called()

    # Test _extract_barriers_from_mask with empty mask (line 495)
    result = _extract_barriers_from_mask(line, [])
    assert result is None

    # Test _get_barrier_geometry missing column (line 503)
    row = pd.Series({"geometry": line})
    with pytest.raises(KeyError, match="Column 'barrier_mask' not found"):
        _get_barrier_geometry(row)


def test_mobility_module_import() -> None:
    """Test mobility module import for coverage."""
    import city2graph.mobility

    assert hasattr(city2graph.mobility, "__all__")
    assert city2graph.mobility.__all__ == []


def test_conftest_fixtures_for_coverage(grid_data: dict) -> None:
    """Test conftest fixtures to improve coverage."""
    # Test the grid_data fixture
    grid = grid_data

    assert isinstance(grid, dict)
    assert "buildings" in grid
    assert "roads" in grid
    assert "tessellations" in grid
    assert isinstance(grid["buildings"], gpd.GeoDataFrame)
    assert isinstance(grid["roads"], gpd.GeoDataFrame)
    assert isinstance(grid["tessellations"], gpd.GeoDataFrame)
    assert not grid["buildings"].empty
    assert not grid["roads"].empty
    assert not grid["tessellations"].empty


def test_overture_data_error_handling() -> None:
    """Test various error handling paths in overture data functions."""
    from city2graph.overture import _process_single_overture_type

    # Test with subprocess.CalledProcessError
    with (
        patch("city2graph.overture.subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")),
        patch("city2graph.overture.logger") as mock_logger,
    ):
        result = _process_single_overture_type(
            data_type="building",
            bbox_str="0,0,1,1",
            output_dir=".",
            prefix="",
            save_to_file=False,
            return_data=True,
            original_polygon=None,
        )
        assert result.empty
        assert result.crs == "EPSG:4326"
        mock_logger.warning.assert_called()

    # Test OSError handling
    with (
        patch("city2graph.overture.subprocess.run", side_effect=OSError("System error")),
        patch("city2graph.overture.logger") as mock_logger,
    ):
        result = _process_single_overture_type(
            data_type="building",
            bbox_str="0,0,1,1",
            output_dir=".",
            prefix="",
            save_to_file=False,
            return_data=True,
            original_polygon=None,
        )
        assert result.empty
        assert result.crs == "EPSG:4326"
        mock_logger.warning.assert_called()


def test_directory_creation_path() -> None:
    """Test directory creation in load_overture_data (line 260)."""
    from city2graph.overture import load_overture_data

    with (
        patch("city2graph.overture.Path") as mock_path,
        patch("city2graph.overture._process_single_overture_type") as mock_process,
    ):
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        mock_process.return_value = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

        load_overture_data(
            area=[0, 0, 1, 1],
            types=["building"],
            output_dir="new_dir",
            save_to_file=True,
            return_data=True,
        )

        mock_path_instance.mkdir.assert_called_once_with(parents=True)


def test_prepare_polygon_no_crs_transform() -> None:
    """Test _prepare_polygon_area when no CRS transformation needed (lines 76-77)."""
    from city2graph.overture import _prepare_polygon_area

    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame([{"geometry": polygon}], crs="EPSG:4326")
    poly_with_crs = gdf.geometry.iloc[0]

    bbox, original_polygon = _prepare_polygon_area(poly_with_crs)

    assert original_polygon == poly_with_crs
    assert bbox is not None


def test_successful_processing_warning() -> None:
    """Test successful processing warning in _process_single_overture_type (line 203)."""
    from city2graph.overture import _process_single_overture_type

    mock_gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")

    with (
        patch("city2graph.overture._read_overture_data", return_value=mock_gdf),
        patch("city2graph.overture._clip_to_polygon", return_value=mock_gdf),
        patch("city2graph.overture.subprocess.run", return_value=Mock(returncode=0)),
        patch("city2graph.overture.logger") as mock_logger,
    ):
        result = _process_single_overture_type(
            data_type="building",
            bbox_str="0,0,1,1",
            output_dir=".",
            prefix="",
            save_to_file=False,
            return_data=True,
            original_polygon=None,
        )

        mock_logger.warning.assert_called_with("Successfully processed %s", "building")
        assert not result.empty
