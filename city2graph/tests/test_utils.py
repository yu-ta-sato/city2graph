"""Test module for city2graph utility functions."""

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
