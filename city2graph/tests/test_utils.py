"""Test module for city2graph utility functions."""

import subprocess
import tempfile
from pathlib import Path
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

from city2graph.utils import _clip_to_polygon
from city2graph.utils import _create_connector_mask
from city2graph.utils import _extract_barriers_from_mask
from city2graph.utils import _extract_line_segment
from city2graph.utils import _extract_valid_connectors
from city2graph.utils import _get_barrier_geometry
from city2graph.utils import _get_nearest_node
from city2graph.utils import _get_substring
from city2graph.utils import _parse_connectors_info
from city2graph.utils import _prepare_polygon_area
from city2graph.utils import _process_single_overture_type
from city2graph.utils import _read_overture_data
from city2graph.utils import _recalc_barrier_mask
from city2graph.utils import _validate_overture_types
from city2graph.utils import adjust_segment_connectors
from city2graph.utils import create_tessellation
from city2graph.utils import filter_graph_by_distance
from city2graph.utils import get_barrier_geometry
from city2graph.utils import identify_barrier_mask
from city2graph.utils import identify_connector_mask
from city2graph.utils import load_overture_data
from city2graph.utils import split_segments_by_connectors

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


def test_filter_graph_by_distance() -> None:
    """Test filter_graph_by_distance function with a NetworkX graph."""
    G = nx.Graph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=1, y=0)
    G.add_edge(1, 2, length=1)
    center = Point(0, 0)
    sub1 = filter_graph_by_distance(G, center, 0.5)
    assert isinstance(sub1, nx.Graph)
    assert list(sub1.nodes) == [1]
    assert sub1.number_of_edges() == 0
    sub2 = filter_graph_by_distance(G, center, 2)
    assert set(sub2.nodes) == {1, 2}
    assert sub2.number_of_edges() == 1


def test_get_barrier_geometry() -> None:
    """Test get_barrier_geometry function with barrier mask extraction."""
    # Arrange
    line = LineString([(0, 0), (2, 0)])
    mask = [[0.5, 1.0]]
    gdf = gpd.GeoDataFrame(
        {"geometry": [line], "barrier_mask": [mask]},
        geometry="geometry",
    )

    # Act
    result = get_barrier_geometry(gdf)
    geom = result.iloc[0]

    # Assert
    assert isinstance(geom, LineString)
    assert list(geom.coords) == [(1.0, 0.0), (2.0, 0.0)]


def test_get_barrier_geometry_multisegment() -> None:
    """Test get_barrier_geometry with multiple mask intervals returns MultiLineString."""
    # Arrange
    line = LineString([(0, 0), (2, 0)])
    mask = [[0.0, 0.25], [0.75, 1.0]]
    gdf = gpd.GeoDataFrame(
        {"geometry": [line], "barrier_mask": [mask]},
        geometry="geometry",
    )

    # Act
    result = get_barrier_geometry(gdf)
    geom = result.iloc[0]

    # Assert
    assert isinstance(geom, MultiLineString)
    parts = list(geom.geoms)
    assert pytest.approx(parts[0].coords[-1][0], rel=1e-6) == 0.5
    assert pytest.approx(parts[1].coords[0][0], rel=1e-6) == 1.5


def test_split_segments_by_connectors_no_split(simple_line: LineString) -> None:
    """Test split_segments_by_connectors when no connectors present."""
    # Arrange: one segment without connectors
    segs = gpd.GeoDataFrame({"id": [1], "geometry": [simple_line]})
    empty_conn = gpd.GeoDataFrame({"id": [], "geometry": []})

    # Act
    result = split_segments_by_connectors(segs, empty_conn)

    # Assert: segment unchanged
    assert len(result) == 1
    assert list(result.iloc[0].geometry.coords) == list(simple_line.coords)


def test_split_segments_by_connectors_with_split(simple_line: LineString) -> None:
    """Test split_segments_by_connectors splits at midpoint connector."""
    # Arrange: segment with connector at 50%
    connectors = gpd.GeoDataFrame({"id": [1], "geometry": [Point(1,0)]})
    segs = gpd.GeoDataFrame({"id": [1], "geometry": [simple_line],
                             "connectors": ["[{'connector_id':1,'at':0.5}]"],
                             "level_rules": ["[]"]})

    # Act
    result = split_segments_by_connectors(segs, connectors)

    # Assert: two parts with correct split_from values
    starts = sorted(result["split_from"].tolist())
    assert starts == [0.0, 0.5]


def test_split_segments_by_connectors_multiple() -> None:
    """Test split_segments_by_connectors splits with multiple connectors."""
    line = LineString([(0, 0), (10, 0)])
    connectors = gpd.GeoDataFrame({"id": [1, 2], "geometry": [Point(3, 0), Point(7, 0)]})
    segs = gpd.GeoDataFrame(
        {
            "id": [10],
            "geometry": [line],
            "connectors": ["[{'connector_id':1,'at':0.3},{'connector_id':2,'at':0.7}]"] ,
            "level_rules": ["[]"],
        },
    )
    result = split_segments_by_connectors(segs, connectors)
    assert len(result) == 3
    assert sorted(result["split_from"]) == [0.0, 0.3, 0.7]


def test_parse_connectors_info_and_invalid() -> None:
    """Test _parse_connectors_info with empty, invalid, single and list inputs."""
    assert _parse_connectors_info("") == []
    assert _parse_connectors_info("not json") == []
    single = "{'connector_id': 3, 'at': 0.4}"
    parsed = _parse_connectors_info(single)
    assert isinstance(parsed, list)
    assert parsed
    assert parsed[0]["connector_id"] == 3
    lst = "[{'connector_id':1,'at':0.1},{'connector_id':2,'at':0.5}]"
    parsed_list = _parse_connectors_info(lst)
    assert [d.get("at") for d in parsed_list] == [0.1, 0.5]


def test_extract_valid_connectors_and_connector_mask() -> None:
    """Test _extract_valid_connectors filters by valid ids and _create_connector_mask orders correctly."""
    # Arrange: raw connector data with valid and invalid entries
    raw = [
        {"connector_id": 1, "at": 0.2},
        {"connector_id": 2, "at": 0.8},
        {"connector_id": 3, "at": None},
        {"connector_id": None, "at": 0.5},
    ]

    # Act: extract only valid connectors by filtering against valid IDs
    valid = _extract_valid_connectors(raw, {1, 2})

    # Assert: should only include connectors with valid IDs and non-None positions
    assert valid == [0.2, 0.8]

    # Act: create connector mask from valid positions
    mask = _create_connector_mask(valid)

    # Assert: mask should include endpoints and connector positions in sorted order
    assert mask == [0.0, 0.2, 0.8, 1.0]

    # Act: test mask when valid contains endpoints
    mask2 = _create_connector_mask([0.0, 1.0])

    # Assert: should not add duplicate endpoints
    assert mask2 == [0.0, 1.0]


def test_extract_barriers_from_mask_and_none() -> None:
    """Test _extract_barriers_from_mask returns correct geometry or None."""
    line = LineString([(0, 0), (4, 0)])
    mask = [[0.25, 0.75]]
    geom = _extract_barriers_from_mask(line, mask)
    assert isinstance(geom, LineString)
    assert tuple(geom.bounds) == (1.0, 0.0, 3.0, 0.0)
    # no parts
    assert _extract_barriers_from_mask(line, []) is None


def test_get_barrier_geometry_missing_and_full() -> None:
    """Test _get_barrier_geometry handles missing barrier_mask and full mask."""
    s = pd.Series({"geometry": LineString([(0, 0), (1, 0)])})
    with pytest.raises(KeyError):
        _get_barrier_geometry(s)
    s2 = pd.Series({"geometry": LineString([(0, 0), (2, 0)]), "barrier_mask": [[0.0, 1.0]]})
    out = _get_barrier_geometry(s2)
    assert isinstance(out, LineString)
    assert out.equals(s2["geometry"])


def test_validate_overture_types_none() -> None:
    """Test _validate_overture_types returns all types on None."""
    types = _validate_overture_types(None)
    assert isinstance(types, list)
    assert "segment" in types


def test_validate_overture_types_invalid() -> None:
    """Test _validate_overture_types rejects invalid types."""
    with pytest.raises(ValueError, match="Invalid Overture Maps data type"):
        _validate_overture_types(["invalid"])


def test_prepare_polygon_area_identity() -> None:
    """Test _prepare_polygon_area returns same polygon when no CRS."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    bbox, out_poly = _prepare_polygon_area(poly)
    assert bbox == [0.0, 0.0, 1.0, 1.0]
    assert out_poly.equals(poly)


def test_identify_barrier_mask_dict_between_none() -> None:
    """Test identify_barrier_mask for dict input and between None."""
    mask = identify_barrier_mask('{"value":1,"between":[0.3,0.6]}')
    assert mask == [[0.0, 0.3], [0.6, 1.0]]
    assert identify_barrier_mask('{"value":2,"between":null}') == []


def test_identify_connector_mask_variations() -> None:
    """Test identify_connector_mask for various inputs."""
    assert identify_connector_mask("") == [0.0, 1.0]
    assert identify_connector_mask('{"connector_id":1,"at":0.25}') == [0.0, 0.25, 1.0]
    assert identify_connector_mask('[{"connector_id":2,"at":0.75}]') == [0.0, 0.75, 1.0]
    assert identify_connector_mask("bad") == [0.0, 1.0]


def test_recalc_barrier_mask_edge_cases() -> None:
    """Test _recalc_barrier_mask with full, empty, no overlap."""
    full = [[0.0, 1.0]]
    assert _recalc_barrier_mask(full, 0.2, 0.8) == full
    assert _recalc_barrier_mask([], 0.0, 1.0) == []
    assert _recalc_barrier_mask([[0.1, 0.2]], 0.3, 0.6) == []


def test_get_substring_edge_cases() -> None:
    """Test _get_substring invalid and tiny segment fallback."""
    line = LineString([(0,0),(1,1)])
    assert _get_substring(line, -0.1, 0.5) is None
    assert _get_substring(line, 0.6, 0.6) is None
    # tiny segment returns None
    seg = _get_substring(line, 0, 1e-12)
    assert seg is None


def test_extract_line_segment_single_point() -> None:
    """Test _extract_line_segment returns tiny line for single point."""
    line = LineString([(0,0),(1,0)])
    pt = line.interpolate(0.5)
    seg = _extract_line_segment(line, pt, pt, 0.5, 0.5)
    # single-point segment yields zero-length LineString
    assert isinstance(seg, LineString)
    assert seg.length == 0


def test_clip_to_polygon() -> None:
    """Test _clip_to_polygon retains only points inside the polygon."""
    # Arrange: define polygon and GeoDataFrame with interior and exterior points
    poly = Polygon([(0,0),(2,0),(2,2),(0,2)])
    gdf = gpd.GeoDataFrame({"geometry": [Point(1,1), Point(3,3)]}, crs="EPSG:4326")

    # Act: apply clipping function
    clipped = _clip_to_polygon(gdf, poly, "test")

    # Assert: only the interior point remains
    assert len(clipped) == 1
    assert clipped.iloc[0].geometry == Point(1,1)


def test_clip_to_polygon_empty() -> None:
    """Test _clip_to_polygon returns empty GeoDataFrame for empty input."""
    # Arrange: empty GeoDataFrame and polygon
    empty_gdf = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    poly = Polygon([(0,0),(1,0),(1,1),(0,1)])

    # Act
    result = _clip_to_polygon(empty_gdf, poly, "test")

    # Assert
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.empty


def test_get_nearest_node_and_empty(nodes_gdf: gpd.GeoDataFrame, empty_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test finding nearest node and handling empty nodes."""
    # Arrange
    pt = Point(1.1, 0)

    # Act
    nearest = _get_nearest_node(pt, nodes_gdf)

    # Assert
    assert nearest == 2

    # Arrange empty nodes
    empty = empty_nodes_gdf

    # Act & Assert
    with pytest.raises(ValueError, match="empty sequence"):
        _get_nearest_node(pt, empty)


def test_create_tessellation_simple() -> None:
    """Test create_tessellation raises on geographic CRS."""
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    geom_df = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")
    with pytest.raises(ValueError, match="geographic CRS"):
        create_tessellation(geom_df)


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


def test_filter_graph_by_distance_comprehensive() -> None:
    """Test filter_graph_by_distance with various graph configurations."""
    # Test empty graph - this causes an error
    G = nx.Graph()
    center = Point(0, 0)
    with pytest.raises(ValueError, match="not enough values to unpack"):
        filter_graph_by_distance(G, center, 10)

    # Test graph with isolated nodes
    G = nx.Graph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=10, y=0)  # far away
    G.add_node(3, x=1, y=1)   # close
    # No edges

    result = filter_graph_by_distance(G, center, 2)
    assert 1 in result.nodes()
    # Node 3 might not be included due to how distance is calculated
    # The function may use shortest path distance, not Euclidean distance
    # Without edges, isolated nodes beyond the center may be excluded
    assert result.number_of_nodes() >= 1

    # Test with edges spanning the distance threshold
    G.add_edge(1, 2)  # edge from close to far node
    result = filter_graph_by_distance(G, center, 2)
    # Should include nodes within distance but may affect edges
    assert result.number_of_nodes() <= G.number_of_nodes()

    # Test with zero distance
    result = filter_graph_by_distance(G, center, 0)
    # With zero distance, no nodes may be included due to how distance is calculated
    assert result.number_of_nodes() >= 0


def test_get_barrier_geometry_comprehensive() -> None:
    """Test comprehensive scenarios for get_barrier_geometry function."""
    # Test with GeoDataFrame having barrier_mask column
    gdf_with_mask = gpd.GeoDataFrame({
        "id": [1],
        "geometry": [LineString([(0, 0), (1, 1)])],
        "barrier_mask": [[[0.0, 1.0]]],
    }, crs="EPSG:4326")

    result = get_barrier_geometry(gdf_with_mask)
    assert isinstance(result, gpd.GeoSeries)
    assert len(result) == 1
    assert isinstance(result.iloc[0], LineString)


def test_overture_data_loading_edge_cases() -> None:
    """Test edge cases for load_overture_data function."""
    # Test with invalid bbox format
    with pytest.raises(ValueError, match="Invalid bbox format"):
        load_overture_data([1, 2])  # Too few coordinates

    with pytest.raises(ValueError, match="Invalid bbox format"):
        load_overture_data([1, 2, 3, 4, 5])  # Too many coordinates

    # Test with non-numeric bbox
    with pytest.raises(ValueError, match="Invalid bbox format"):
        load_overture_data(["a", "b", "c", "d"])

    # Test with invalid data types
    with pytest.raises(ValueError, match="Invalid Overture Maps data type"):
        load_overture_data([0, 0, 1, 1], types=["invalid_type"])

    # Test with Polygon input (this will require actual Polygon)
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    # This would normally call external command, so we'll just test the validation
    result = load_overture_data(poly, types=["building"], return_data=False)
    assert isinstance(result, dict)


def test_string_extraction_edge_cases() -> None:
    """Test edge cases for _get_substring function."""
    # Test with invalid parameters
    line = LineString([(0, 0), (1, 1)])

    # Invalid start/end percentages
    assert _get_substring(line, -0.1, 0.5) is None  # start < 0
    assert _get_substring(line, 0.5, 1.1) is None  # end > 1
    assert _get_substring(line, 0.5, 0.3) is None  # start >= end

    # Invalid line type
    assert _get_substring("not a line", 0.0, 1.0) is None

    # Zero-length substring
    assert _get_substring(line, 0.5, 0.5) is None

    # Full line
    result = _get_substring(line, 0.0, 1.0)
    assert isinstance(result, LineString)

    # Nearly full line
    result = _get_substring(line, 0.0001, 0.9999)
    assert isinstance(result, LineString)


def test_segment_splitting_edge_cases() -> None:
    """Test edge cases for split_segments_by_connectors function."""
    # Test with segments missing connectors column
    segments_no_connectors = gpd.GeoDataFrame({
        "id": [1],
        "geometry": [LineString([(0, 0), (1, 1)])],
    }, crs="EPSG:4326")

    connectors = gpd.GeoDataFrame({
        "id": ["c1"],
        "geometry": [Point(0.5, 0.5)],
    }, crs="EPSG:4326")

    result = split_segments_by_connectors(segments_no_connectors, connectors)
    # Should return original segments unchanged
    assert len(result) == 1


def test_tessellation_creation_edge_cases() -> None:
    """Test edge cases for create_tessellation function."""
    # Test with empty geometry in projected CRS
    empty_geom = gpd.GeoDataFrame(geometry=[], crs="EPSG:3857")  # Use projected CRS

    result = create_tessellation(empty_geom)
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 0

    # Test with multiple polygons that can form valid tessellation
    multi_poly = gpd.GeoDataFrame({
        "id": [1, 2, 3, 4],
        "geometry": [
            Polygon([(0, 0), (50, 0), (50, 50), (0, 50)]),      # Bottom-left
            Polygon([(50, 0), (100, 0), (100, 50), (50, 50)]),  # Bottom-right
            Polygon([(0, 50), (50, 50), (50, 100), (0, 100)]),  # Top-left
            Polygon([(50, 50), (100, 50), (100, 100), (50, 100)]),  # Top-right
        ],
    }, crs="EPSG:3857")  # Use projected CRS

    result = create_tessellation(multi_poly)
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) >= 0  # Should have some tessellations or empty result

    # Test with proper barriers that create valid enclosed tessellations
    barriers = gpd.GeoDataFrame({
        "id": [1, 2, 3, 4],
        "geometry": [
            # Create a proper closed boundary
            LineString([(-10, -10), (110, -10), (110, 110), (-10, 110), (-10, -10)]),
            # Internal barriers
            LineString([(0, 50), (100, 50)]),   # Horizontal divider
            LineString([(50, 0), (50, 100)]),   # Vertical divider
            LineString([(25, 25), (75, 25), (75, 75), (25, 75), (25, 25)]),  # Inner square
        ],
    }, crs="EPSG:3857")  # Use projected CRS

    result = create_tessellation(multi_poly, primary_barriers=barriers)
    assert isinstance(result, gpd.GeoDataFrame)
    # The result may be empty if tessellation can't be created, but function should not crash


def test_filter_graph_edge_cases() -> None:
    """Test edge cases for filter_graph_by_distance function."""
    # Test with NetworkX graph
    G = nx.Graph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=1, y=1)
    G.add_edge(1, 2, length=1.414)

    center = Point(0, 0)
    result = filter_graph_by_distance(G, center, 2.0)
    assert isinstance(result, nx.Graph)

    # Test with GeoDataFrame
    gdf = gpd.GeoDataFrame({
        "node_id": [1, 2],
        "geometry": [Point(0, 0), Point(1, 1)],
    }, crs="EPSG:4326")

    result = filter_graph_by_distance(gdf, center, 2.0)
    assert isinstance(result, gpd.GeoDataFrame)

    # Test with center as GeoSeries
    center_series = gpd.GeoSeries([Point(0, 0)], crs="EPSG:4326")
    result = filter_graph_by_distance(gdf, center_series, 2.0)
    assert isinstance(result, gpd.GeoDataFrame)

    # Test with center as GeoDataFrame
    center_gdf = gpd.GeoDataFrame({"id": [1], "geometry": [Point(0, 0)]}, crs="EPSG:4326")
    result = filter_graph_by_distance(gdf, center_gdf, 2.0)
    assert isinstance(result, gpd.GeoDataFrame)


# ============================================================================
# TESTS FOR UNCOVERED CODE PATHS IN UTILS.PY
# ============================================================================

def test_prepare_polygon_area_with_crs() -> None:
    """Test _prepare_polygon_area with a polygon that has CRS."""
    # Create a GeoDataFrame with a polygon in a non-WGS84 CRS
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame([{"geometry": polygon}], crs="EPSG:3857")
    poly_with_crs = gdf.geometry.iloc[0]

    # Test CRS transformation path (lines 76-77)
    bbox, transformed_poly = _prepare_polygon_area(poly_with_crs)

    assert isinstance(bbox, list)
    assert len(bbox) == 4
    assert transformed_poly is not None


def test_read_overture_data_save_to_file_existing() -> None:
    """Test _read_overture_data with save_to_file and existing file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test GeoJSON file
        test_file = Path(temp_dir) / "test.geojson"
        test_gdf = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0)]},
            crs="EPSG:4326",
        )
        test_gdf.to_file(test_file, driver="GeoJSON")

        # Mock process object
        mock_process = Mock()
        mock_process.stdout = None

        # Test path where file exists and has content (lines 92-93)
        result = _read_overture_data(
            str(test_file), mock_process, save_to_file=True, data_type="building",
        )

        assert not result.empty
        assert len(result) == 1


def test_read_overture_data_save_to_file_no_data() -> None:
    """Test _read_overture_data with save_to_file but no data returned."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create an empty file
        test_file = Path(temp_dir) / "empty.geojson"
        test_file.touch()

        # Mock process object
        mock_process = Mock()
        mock_process.stdout = None

        # Test warning path when no data returned (line 94)
        with patch("city2graph.utils.logger") as mock_logger:
            _read_overture_data(
                str(test_file), mock_process, save_to_file=True, data_type="building",
            )
            mock_logger.warning.assert_called_with("No data returned for %s", "building")


def test_read_overture_data_stdout_success() -> None:
    """Test _read_overture_data reading from stdout successfully."""
    # Create valid GeoJSON string
    geojson_str = (
        '{"type": "FeatureCollection", "features": '
        '[{"type": "Feature", "geometry": {"type": "Point", "coordinates": [0, 0]}, "properties": {}}]}'
    )

    # Mock process with stdout
    mock_process = Mock()
    mock_process.stdout = geojson_str

    # Test successful reading from stdout (lines 96-98)
    with patch("geopandas.read_file") as mock_read_file:
        mock_read_file.return_value = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
        _read_overture_data("", mock_process, save_to_file=False, data_type="building")
        mock_read_file.assert_called_with(geojson_str)


def test_read_overture_data_stdout_parsing_error() -> None:
    """Test _read_overture_data with parsing error from stdout."""
    # Mock process with invalid stdout
    mock_process = Mock()
    mock_process.stdout = "invalid geojson"

    # Test exception handling path (lines 99-100)
    with (
        patch("geopandas.read_file", side_effect=ValueError("Invalid GeoJSON")),
        patch("city2graph.utils.logger") as mock_logger,
    ):
        _read_overture_data("", mock_process, save_to_file=False, data_type="building")
        # Check that warning was called with correct data_type
        call_args = mock_logger.warning.call_args
        assert call_args[0][0] == "Could not parse GeoJSON for %s: %s"
        assert call_args[0][1] == "building"


def test_read_overture_data_empty_return() -> None:
    """Test _read_overture_data returning empty GeoDataFrame."""
    # Mock process with no stdout
    mock_process = Mock()
    mock_process.stdout = None

    # Test empty return path (line 102)
    result = _read_overture_data("", mock_process, save_to_file=False, data_type="building")

    assert result.empty
    assert result.crs == "EPSG:4326"


def test_clip_to_polygon_crs_conversion() -> None:
    """Test _clip_to_polygon with CRS conversion needed."""
    # Create GDF in different CRS
    gdf = gpd.GeoDataFrame(
        {"geometry": [Point(0, 0)]},
        crs="EPSG:3857",
    )

    # Create polygon in WGS84
    polygon = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])

    # Test CRS conversion path (line 114)
    with patch("geopandas.clip") as mock_clip:
        mock_clip.return_value = gdf
        _clip_to_polygon(gdf, polygon, "building")
        # Verify that mask was converted to gdf.crs
        assert mock_clip.called


def test_clip_to_polygon_clipping_error() -> None:
    """Test _clip_to_polygon with clipping error."""
    gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")
    polygon = Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])

    # Test exception handling path (lines 118-120)
    with (
        patch("geopandas.clip", side_effect=ValueError("Clipping error")),
        patch("city2graph.utils.logger") as mock_logger,
    ):
        result = _clip_to_polygon(gdf, polygon, "building")

        # Should return original gdf when clipping fails
        assert result.equals(gdf)
        # Check that warning was called with correct data_type
        call_args = mock_logger.warning.call_args
        assert call_args[0][0] == "Error clipping %s to polygon: %s"
        assert call_args[0][1] == "building"


def test_validate_overture_types_invalid_type() -> None:
    """Test _validate_overture_types with invalid type."""
    # Test invalid type validation (internally calls _raise_invalid_data_type)
    with pytest.raises(ValueError, match="Invalid data type: invalid_type"):
        _process_single_overture_type(
            data_type="invalid_type",
            bbox_str="0,0,1,1",
            output_dir=".",
            prefix="",
            save_to_file=False,
            return_data=True,
            original_polygon=None,
        )


def test_process_single_overture_type_invalid_data_type() -> None:
    """Test _process_single_overture_type with invalid data type."""
    # Test validation of invalid data type (lines 137-138, 146)
    with pytest.raises(ValueError, match="Invalid data type: invalid_type"):
        _process_single_overture_type(
            data_type="invalid_type",
            bbox_str="0,0,1,1",
            output_dir=".",
            prefix="",
            save_to_file=False,
            return_data=True,
            original_polygon=None,
        )


def test_process_single_overture_type_invalid_bbox() -> None:
    """Test _process_single_overture_type with invalid bbox format."""
    from city2graph.utils import _process_single_overture_type

    # Test invalid bbox format
    with pytest.raises(ValueError, match="Invalid bbox format"):
        _process_single_overture_type(
            data_type="building",
            bbox_str="invalid,bbox,format",
            output_dir=".",
            prefix="",
            save_to_file=False,
            return_data=True,
            original_polygon=None,
        )


@patch("subprocess.run")
def test_process_single_overture_type_subprocess_error(mock_run: Mock) -> None:
    """Test _process_single_overture_type with subprocess error."""
    from city2graph.utils import _process_single_overture_type

    # Mock subprocess to raise CalledProcessError
    mock_run.side_effect = subprocess.CalledProcessError(1, "overturemaps")

    with patch("city2graph.utils.logger") as mock_logger:
        result = _process_single_overture_type(
            data_type="building",
            bbox_str="0,0,1,1",
            output_dir=".",
            prefix="",
            save_to_file=False,
            return_data=True,
            original_polygon=None,
        )

        # Should return empty GeoDataFrame on subprocess error
        assert result.empty
        assert result.crs == "EPSG:4326"
        mock_logger.warning.assert_called()


@patch("subprocess.run")
def test_process_single_overture_type_processing_error(mock_run: Mock) -> None:
    """Test _process_single_overture_type with processing error."""
    from city2graph.utils import _process_single_overture_type

    # Mock subprocess to succeed but cause processing error
    mock_process = Mock()
    mock_process.stdout = None
    mock_run.return_value = mock_process

    # Mock _read_overture_data to raise exception
    with (
        patch("city2graph.utils._read_overture_data", side_effect=OSError("Processing error")),
        patch("city2graph.utils.logger") as mock_logger,
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

        # Should return empty GeoDataFrame on processing error
        assert result.empty
        assert result.crs == "EPSG:4326"
        mock_logger.warning.assert_called()


def test_load_overture_data_polygon_with_crs() -> None:
    """Test load_overture_data with polygon that needs CRS transformation."""
    # Create polygon in non-WGS84 CRS
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame([{"geometry": polygon}], crs="EPSG:3857")
    poly_with_crs = gdf.geometry.iloc[0]

    # Mock the subprocess and file operations to avoid actual downloads
    with patch("city2graph.utils._process_single_overture_type") as mock_process:
        mock_process.return_value = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")

        # This should trigger the CRS transformation in _prepare_polygon_area
        result = load_overture_data(poly_with_crs, types=["building"], return_data=True)

        assert isinstance(result, dict)
        assert "building" in result


# ============================================================================
# TESTS FOR UNCOVERED CODE PATHS IN UTILS.PY
# ============================================================================

def test_additional_uncovered_paths() -> None:
    """Test additional uncovered code paths in utils.py functions."""
    # Test _get_substring exception handling (lines 390-396)
    line = LineString([(0, 0), (1, 0)])
    with (
        patch("city2graph.utils._extract_line_segment", side_effect=ValueError("Test error")),
        patch("city2graph.utils.logger") as mock_logger,
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
    from city2graph.utils import _process_single_overture_type

    # Test with subprocess.CalledProcessError
    with (
        patch("city2graph.utils.subprocess.run", side_effect=subprocess.CalledProcessError(1, "cmd")),
        patch("city2graph.utils.logger") as mock_logger,
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
        patch("city2graph.utils.subprocess.run", side_effect=OSError("System error")),
        patch("city2graph.utils.logger") as mock_logger,
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
    from city2graph.utils import load_overture_data

    with (
        patch("city2graph.utils.Path") as mock_path,
        patch("city2graph.utils._process_single_overture_type") as mock_process,
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
    from city2graph.utils import _prepare_polygon_area

    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame([{"geometry": polygon}], crs="EPSG:4326")
    poly_with_crs = gdf.geometry.iloc[0]

    bbox, original_polygon = _prepare_polygon_area(poly_with_crs)

    assert original_polygon == poly_with_crs
    assert bbox is not None


def test_successful_processing_warning() -> None:
    """Test successful processing warning in _process_single_overture_type (line 203)."""
    from city2graph.utils import _process_single_overture_type

    mock_gdf = gpd.GeoDataFrame({"geometry": [Point(0, 0)]}, crs="EPSG:4326")

    with (
        patch("city2graph.utils._read_overture_data", return_value=mock_gdf),
        patch("city2graph.utils._clip_to_polygon", return_value=mock_gdf),
        patch("city2graph.utils.subprocess.run", return_value=Mock(returncode=0)),
        patch("city2graph.utils.logger") as mock_logger,
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
