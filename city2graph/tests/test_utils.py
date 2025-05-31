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
from city2graph.utils import _create_split_row
from city2graph.utils import _extract_barriers_from_mask
from city2graph.utils import _extract_line_segment
from city2graph.utils import _extract_valid_connectors
from city2graph.utils import _get_barrier_geometry
from city2graph.utils import _get_nearest_node
from city2graph.utils import _get_substring
from city2graph.utils import _parse_connectors_info
from city2graph.utils import _prepare_polygon_area
from city2graph.utils import _rebuild_geometry
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


# Common test fixtures
@pytest.fixture
def simple_line() -> LineString:
    """Return a simple horizontal LineString."""
    return LineString([(0, 0), (2, 0)])


@pytest.fixture
def simple_polygon() -> Polygon:
    """Return a unit square Polygon."""
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_get_substring_basic(simple_line: LineString) -> None:
    """Test basic functionality of _get_substring function."""
    # Arrange
    line = simple_line

    # Act
    seg = _get_substring(line, 0.25, 0.75)

    # Assert
    assert isinstance(seg, LineString)
    start, end = seg.coords[0], seg.coords[-1]
    assert pytest.approx(start[0], rel=1e-6) == 0.5
    assert start[1] == 0
    assert pytest.approx(end[0], rel=1e-6) == 1.5
    assert end[1] == 0


def test_get_substring_invalid() -> None:
    """Test _get_substring returns None for invalid inputs."""
    # Arrange
    invalid_line = "not a line"

    # Act & Assert
    assert _get_substring(invalid_line, 0, 1) is None

    # Arrange
    line = LineString([(0, 0), (1, 1)])

    # Act & Assert
    assert _get_substring(line, 0.6, 0.4) is None


def test__extract_line_segment_basic() -> None:
    """Test _extract_line_segment with intermediate vertices."""
    line = LineString([(0, 0), (1, 0), (2, 0)])
    total_length = line.length
    start_dist = 0.25 * total_length
    end_dist = 0.75 * total_length
    start_pt = line.interpolate(start_dist)
    end_pt = line.interpolate(end_dist)
    seg = _extract_line_segment(line, start_pt, end_pt, start_dist, end_dist)
    assert isinstance(seg, LineString)
    coords = list(seg.coords)
    assert pytest.approx(coords[0][0], rel=1e-6) == 0.5
    assert pytest.approx(coords[-1][0], rel=1e-6) == 1.5


def test_get_substring_full_line() -> None:
    """Test substring with full line returns original."""
    line = LineString([(0, 0), (2, 0)])
    seg = _get_substring(line, 0, 1)
    assert seg.equals(line)


def test_identify_barrier_mask_empty_and_invalid() -> None:
    """Test identify_barrier_mask function with empty and invalid inputs."""
    assert identify_barrier_mask("") == [[0.0, 1.0]]
    assert identify_barrier_mask("null") == [[0.0, 1.0]]
    assert identify_barrier_mask("not json") == [[0.0, 1.0]]


def test_identify_barrier_mask_simple_and_null_between() -> None:
    """Test identify_barrier_mask function with simple rules and null between values."""
    rules = "[{'value': 1, 'between': [0.2, 0.5]}]"
    mask = identify_barrier_mask(rules)
    assert mask == [[0.0, 0.2], [0.5, 1.0]]
    null_rules = "[{'value': 1, 'between': None}]"
    assert identify_barrier_mask(null_rules) == []


def test_identify_barrier_mask_multiple_intervals() -> None:
    """Test identify_barrier_mask with multiple non-zero rules."""
    rules = "[{'value': 1, 'between': [0.0, 0.1]}, {'value': 2, 'between': [0.9, 1.0]}]"
    mask = identify_barrier_mask(rules)
    # barrier intervals at [0,0.1] and [0.9,1.0], complement is [0.1,0.9]
    assert mask == [[0.1, 0.9]]


def test_identify_barrier_mask_value_zero_only() -> None:
    """Test identify_barrier_mask with only zero-value rules (non-barriers)."""
    rules = "[{'value': 0, 'between': [0.2, 0.4]}]"
    mask = identify_barrier_mask(rules)
    assert mask == [[0.0, 1.0]]


def test_identify_connector_mask() -> None:
    """Test identify_connector_mask function with various connector configurations."""
    assert identify_connector_mask("") == [0.0, 1.0]
    info = "{'connector_id': 1, 'at': 0.3}"
    assert identify_connector_mask(info) == [0.0, 0.3, 1.0]
    info_list = "[{'connector_id': 2, 'at': 0.4}, {'connector_id':3,'at':0.1}]"
    assert identify_connector_mask(info_list) == [0.0, 0.1, 0.4, 1.0]


def test_identify_connector_mask_invalid_json() -> None:
    """Test identify_connector_mask handles invalid JSON gracefully."""
    invalid = "[{'connector_id': 1, 'at': 0.2"  # missing closing bracket
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
    coords = list(geom.coords)
    assert coords[0] == (1.0, 0.0)
    assert coords[-1] == (2.0, 0.0)


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
    assert pytest.approx(parts[0].coords[0][0], rel=1e-6) == 0.0
    assert pytest.approx(parts[0].coords[-1][0], rel=1e-6) == 0.5
    assert pytest.approx(parts[1].coords[0][0], rel=1e-6) == 1.5
    assert pytest.approx(parts[1].coords[-1][0], rel=1e-6) == 2.0


def test_split_segments_by_connectors_no_split() -> None:
    """Test split_segments_by_connectors function with no connectors to split on."""
    line = LineString([(0, 0), (2, 0)])
    segs = gpd.GeoDataFrame({"id": [1], "geometry": [line]})
    empty_conn = gpd.GeoDataFrame({"id": [], "geometry": []})
    result = split_segments_by_connectors(segs, empty_conn)
    assert len(result) == 1
    assert list(result.iloc[0].geometry.coords) == list(line.coords)


def test_split_segments_by_connectors_with_split() -> None:
    """Test split_segments_by_connectors function with connectors that split segments."""
    line = LineString([(0, 0), (2, 0)])
    connectors = gpd.GeoDataFrame({"id": [1], "geometry": [Point(1, 0)]})
    segs = gpd.GeoDataFrame(
        {
            "id": [1],
            "geometry": [line],
            "connectors": ["[{'connector_id': 1, 'at': 0.5}]"],
            "level_rules": ["[]"],
        },
    )
    result = split_segments_by_connectors(segs, connectors)
    assert len(result) == 2
    starts = sorted(result["split_from"].tolist())
    assert starts == [0.0, 0.5]


def test_split_segments_by_connectors_invalid_connector_id() -> None:
    """Test split_segments_by_connectors ignores connectors with invalid IDs."""
    line = LineString([(0, 0), (2, 0)])
    # connectors_gdf has id 2, but segment references 1
    connectors = gpd.GeoDataFrame(
        {"id": [2], "geometry": [Point(1, 0)]},
        geometry="geometry", crs=None,
    )
    segs = gpd.GeoDataFrame(
        {
            "id": [1],
            "geometry": [line],
            "connectors": ["[{'connector_id': 1, 'at': 0.5}+]'"],
            "level_rules": ["[]"],
        },
        geometry="geometry", crs=None,
    )
    result = split_segments_by_connectors(segs, connectors)
    # no valid split, single segment preserved
    assert len(result) == 1
    assert list(result.iloc[0].geometry.coords) == list(line.coords)


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
    raw = [
        {"connector_id": 1, "at": 0.2},
        {"connector_id": 2, "at": 0.8},
        {"connector_id": 3, "at": None},
        {"connector_id": None, "at": 0.5},
    ]
    valid = _extract_valid_connectors(raw, {1, 2})
    assert valid == [0.2, 0.8]
    mask = _create_connector_mask(valid)
    assert mask == [0.0, 0.2, 0.8, 1.0]
    # mask when valid contains endpoints
    mask2 = _create_connector_mask([0.0, 1.0])
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


# Additional tests for improved coverage
def test_clip_to_polygon() -> None:
    """Test _clip_to_polygon retains points inside polygon and drops outside."""
    poly = Polygon([(0,0),(2,0),(2,2),(0,2)])
    gdf = gpd.GeoDataFrame({"geometry":[Point(1,1), Point(3,3)]}, crs="EPSG:4326")
    clipped = _clip_to_polygon(gdf, poly, "test")
    assert len(clipped)==1
    assert clipped.iloc[0].geometry == Point(1,1)


def test_get_nearest_node() -> None:
    """Test internal _get_nearest_node selects closest node."""
    nodes = gpd.GeoDataFrame({"node_id":[1,2],"geometry":[Point(0,0), Point(10,0)]}, crs="EPSG:4326")
    assert _get_nearest_node(Point(1,0), nodes, node_id="node_id") == 1
    assert _get_nearest_node(Point(9,0), nodes, node_id="node_id") == 2


def test_adjust_segment_connectors_merge_endpoints() -> None:
    """Test adjust_segment_connectors merges close endpoints."""
    line1 = LineString([(0,0),(1,0)])
    line2 = LineString([(1.01,0),(2,0)])
    gdf2 = gpd.GeoDataFrame({"geometry":[line1,line2]}, crs="EPSG:4326")
    result = adjust_segment_connectors(gdf2.copy(), threshold=0.05)
    c1 = list(result.iloc[0].geometry.coords)
    c2 = list(result.iloc[1].geometry.coords)
    assert pytest.approx(c1[-1][0], rel=1e-6) == pytest.approx(c2[0][0], rel=1e-6)


def test_extract_barriers_from_mask_multisegment() -> None:
    """Test _extract_barriers_from_mask returns MultiLineString for multiple intervals."""
    line = LineString([(0,0),(4,0)])
    mask = [[0.0,0.25],[0.75,1.0]]
    geom = _extract_barriers_from_mask(line, mask)
    assert isinstance(geom, MultiLineString)
    parts = list(geom.geoms)
    # check endpoints of each part only
    assert parts[0].coords[0] == (0.0, 0.0)
    assert parts[0].coords[-1] == (1.0, 0.0)
    assert parts[1].coords[0] == (3.0, 0.0)
    assert parts[1].coords[-1] == (4.0, 0.0)


def test_parse_connectors_info_nonlist() -> None:
    """Test _parse_connectors_info returns empty for non-dict/list JSON."""
    assert _parse_connectors_info("123") == []


def test_validate_overture_types_valid() -> None:
    """Test _validate_overture_types returns list when types valid."""
    types2 = _validate_overture_types(["segment","connector"])
    assert types2 == ["segment","connector"]


def test_filter_graph_by_distance_gdf() -> None:
    """Test filter_graph_by_distance with GeoDataFrame input."""
    line5 = LineString([(0,0),(1,0)])
    gdf5 = gpd.GeoDataFrame({"geometry":[line5],"length":[1]}, crs="EPSG:4326")
    sub = filter_graph_by_distance(gdf5, Point(0,0), 2)
    assert isinstance(sub, gpd.GeoDataFrame)
    assert len(sub) == 1
    # small distance may yield no graph output: expect ValueError from momepy
    with pytest.raises(ValueError, match="not enough values to unpack"):
        filter_graph_by_distance(gdf5, Point(0,0), 0.5)


def test_load_overture_data_invalid_type() -> None:
    """Test load_overture_data raises ValueError for invalid types."""
    with pytest.raises(ValueError, match="Invalid Overture Maps data type"):
        load_overture_data([0.0, 0.0, 1.0, 1.0], types=["invalid"], save_to_file=False, return_data=False)


def test_create_tessellation_morphological() -> None:
    """Test morphological tessellation without primary_barriers."""
    line = LineString([(0,0),(1,0),(1,1),(0,1)])
    gdf = gpd.GeoDataFrame({"geometry":[line]}, crs="EPSG:4326")
    # geographic CRS not supported: expect error
    with pytest.raises(ValueError, match="Geometry is in a geographic CRS"):
        create_tessellation(gdf)


def test_create_tessellation_enclosed_crs_mismatch() -> None:
    """Test enclosed tessellation raises on CRS mismatch."""
    poly = Polygon([(0,0),(1,0),(1,1),(0,1)])
    geom_gdf = gpd.GeoDataFrame({"geometry":[poly]}, crs="EPSG:4326")
    barriers = gpd.GeoDataFrame({"geometry":[poly]}, crs="EPSG:3857")
    with pytest.raises(ValueError, match="CRS mismatch"):
        create_tessellation(geom_gdf, primary_barriers=barriers)


def test_load_overture_data_no_return_data() -> None:
    """Test load_overture_data returns empty dict when return_data is False."""
    result = load_overture_data([0.0,0.0,0.1,0.1], types=["segment"], save_to_file=False, return_data=False)
    assert result == {}


def test_load_overture_data_polygon_input() -> None:
    """Test load_overture_data handles Polygon input and returns GeoDataFrames."""
    poly = Polygon([(0,0),(1,0),(1,1),(0,1)])
    result = load_overture_data(poly, types=["segment"], save_to_file=False, return_data=True)
    assert isinstance(result, dict)
    assert "segment" in result
    gdf = result["segment"]
    assert isinstance(gdf, gpd.GeoDataFrame)
    assert gdf.crs == "EPSG:4326"


def test_create_split_row_and_rebuild_geometry() -> None:
    """Test _create_split_row and _rebuild_geometry functions."""
    # Create a sample row and part
    row = pd.Series({"geometry": LineString([(0, 0), (2, 0)]), "barrier_mask": [[0.0, 1.0]], "id": "orig"})
    part = LineString([(0, 0), (1, 0)])
    conn_mask = [0.0, 0.5, 1.0]
    new_row = _create_split_row(
        row,
        part,
        0.0,
        0.5,
        conn_mask,
        row["barrier_mask"],
        original_id="orig",
        counter=2,
    )
    assert isinstance(new_row, pd.Series)
    assert new_row.geometry.equals(part)
    assert new_row["split_from"] == 0.0
    assert new_row["split_to"] == 0.5
    assert new_row["connector_mask"] == conn_mask
    assert new_row["barrier_mask"] == [[0.0, 1.0]]
    assert new_row.id == "orig_2"

    # Test _rebuild_geometry
    seg_id = 10
    geom = LineString([(0, 0), (1, 1), (2, 2)])
    pivot_df = pd.DataFrame({
        ("x_centroid", "start"): {seg_id: 0.1},
        ("y_centroid", "start"): {seg_id: 0.2},
        ("x_centroid", "end"): {seg_id: 0.9},
        ("y_centroid", "end"): {seg_id: 0.8},
    })
    coords = _rebuild_geometry(seg_id, geom, pivot_df)
    assert coords[0] == (0.1, 0.2)
    assert coords[-1] == (0.9, 0.8)
    assert coords[1] == (1, 1)


def test_clip_to_polygon_empty() -> None:
    """Test _clip_to_polygon returns empty GeoDataFrame for empty input."""
    empty = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:4326")
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    result = _clip_to_polygon(empty, poly, "test")
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.empty


def test_identify_barrier_mask_dict_input() -> None:
    """Test identify_barrier_mask accepts dict JSON input."""
    mask = identify_barrier_mask('{"value":5,"between":[0.2,0.4]}')
    assert mask == [[0.0, 0.2], [0.4, 1.0]]
    # value zero yields full mask
    mask0 = identify_barrier_mask('{"value":0,"between":null}')
    assert mask0 == [[0.0, 1.0]]


def test_filter_graph_by_distance_no_nodes() -> None:
    """Test filter_graph_by_distance returns empty graph when no nodes within distance."""
    G = nx.Graph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=10, y=0)
    G.add_edge(1, 2, length=10)
    sub = filter_graph_by_distance(G, Point(0, 0), 0)
    assert isinstance(sub, nx.Graph)
    assert list(sub.nodes) == []
    assert sub.number_of_edges() == 0
