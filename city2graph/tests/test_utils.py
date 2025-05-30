"""Test module for city2graph utility functions."""

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph.utils import _get_substring
from city2graph.utils import _recalc_barrier_mask
from city2graph.utils import filter_network_by_distance
from city2graph.utils import get_barrier_geometry
from city2graph.utils import identify_barrier_mask
from city2graph.utils import identify_connector_mask
from city2graph.utils import split_segments_by_connectors


def test_get_substring_basic() -> None:
    """Test basic functionality of _get_substring function."""
    line = LineString([(0, 0), (2, 0)])
    seg = _get_substring(line, 0.25, 0.75)
    assert isinstance(seg, LineString)
    start, end = seg.coords[0], seg.coords[-1]
    assert pytest.approx(start[0], rel=1e-6) == 0.5
    assert start[1] == 0
    assert pytest.approx(end[0], rel=1e-6) == 1.5
    assert end[1] == 0


def test_get_substring_invalid() -> None:
    """Test _get_substring function with invalid inputs."""
    assert _get_substring("not a line", 0, 1) is None
    line = LineString([(0, 0), (1, 1)])
    assert _get_substring(line, 0.6, 0.4) is None


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


def test_identify_connector_mask() -> None:
    """Test identify_connector_mask function with various connector configurations."""
    assert identify_connector_mask("") == [0.0, 1.0]
    info = "{'connector_id': 1, 'at': 0.3}"
    assert identify_connector_mask(info) == [0.0, 0.3, 1.0]
    info_list = "[{'connector_id': 2, 'at': 0.4}, {'connector_id':3,'at':0.1}]"
    assert identify_connector_mask(info_list) == [0.0, 0.1, 0.4, 1.0]


def test_recalc_barrier_mask() -> None:
    """Test _recalc_barrier_mask function with barrier mask recalculation."""
    orig = [[0.2, 0.8]]
    assert _recalc_barrier_mask([[0.0, 1.0]], 0.2, 0.6) == [[0.0, 1.0]]
    new_mask = _recalc_barrier_mask(orig, 0.2, 0.6)
    assert new_mask == [[0.0, 1.0]]


def test_filter_network_by_distance_graph() -> None:
    """Test filter_network_by_distance function with a NetworkX graph."""
    G = nx.Graph()
    G.add_node(1, x=0, y=0)
    G.add_node(2, x=1, y=0)
    G.add_edge(1, 2, length=1)
    center = Point(0, 0)
    sub1 = filter_network_by_distance(G, center, 0.5)
    assert isinstance(sub1, nx.Graph)
    assert list(sub1.nodes) == [1]
    assert sub1.number_of_edges() == 0
    sub2 = filter_network_by_distance(G, center, 2)
    assert set(sub2.nodes) == {1, 2}
    assert sub2.number_of_edges() == 1


def test_get_barrier_geometry() -> None:
    """Test get_barrier_geometry function with barrier mask extraction."""
    line = LineString([(0, 0), (2, 0)])
    mask = [[0.5, 1.0]]
    gdf = gpd.GeoDataFrame(
        {"geometry": [line], "barrier_mask": [mask]}, geometry="geometry",
    )
    result = get_barrier_geometry(gdf)
    geom = result.iloc[0]
    assert isinstance(geom, LineString)
    coords = list(geom.coords)
    assert coords[0] == (1.0, 0.0)
    assert coords[-1] == (2.0, 0.0)


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
