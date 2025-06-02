"""Tests for the utils module."""
import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph.utils import _compute_nodes_within_distance
from city2graph.utils import _create_empty_result
from city2graph.utils import _create_nodes_gdf
from city2graph.utils import _extract_node_positions
from city2graph.utils import _get_nearest_node
from city2graph.utils import _normalize_center_points
from city2graph.utils import _validate_gdf
from city2graph.utils import _validate_nx
from city2graph.utils import create_tessellation
from city2graph.utils import filter_graph_by_distance
from city2graph.utils import gdf_to_nx
from city2graph.utils import nx_to_gdf


def test_get_nearest_node() -> None:
    """Test finding the nearest node in a GeoDataFrame."""
    nodes = gpd.GeoDataFrame({"node_id": [1, 2],
                             "geometry": [Point(0, 0), Point(1, 1)]},
                             crs=None)
    result = _get_nearest_node(Point(0.1, 0.1), nodes, node_id="node_id")
    assert result == 1


def test_extract_node_positions() -> None:
    """Test extracting node positions from NetworkX graph."""
    # Test with pos attributes
    G = nx.Graph()
    G.add_node(1, pos=(0, 0))
    G.add_node(2, pos=(1, 1))
    positions = _extract_node_positions(G)
    assert positions == {1: (0, 0), 2: (1, 1)}

    # Test with coordinate tuples as node IDs
    G2 = nx.Graph()
    G2.add_node((0, 0))
    G2.add_node((1, 1))
    positions2 = _extract_node_positions(G2)
    assert positions2 == {(0, 0): (0, 0), (1, 1): (1, 1)}

    # Test with x,y attributes
    G3 = nx.Graph()
    G3.add_node(1, x=0, y=0)
    G3.add_node(2, x=1, y=1)
    positions3 = _extract_node_positions(G3)
    assert positions3 == {1: (0, 0), 2: (1, 1)}


def test_create_nodes_gdf() -> None:
    """Test creating nodes GeoDataFrame from positions."""
    pos_dict = {1: (0, 0), 2: (1, 1)}
    nodes_gdf = _create_nodes_gdf(pos_dict, "node_id", "EPSG:4326")
    assert len(nodes_gdf) == 2
    assert list(nodes_gdf["node_id"]) == [1, 2]
    assert nodes_gdf.crs == "EPSG:4326"

    # Test empty dictionary
    empty_gdf = _create_nodes_gdf({}, "node_id", "EPSG:4326")
    assert len(empty_gdf) == 0


def test_compute_nodes_within_distance() -> None:
    """Test computing nodes within distance from center points."""
    # Create a simple path graph
    G = nx.path_graph(4)  # nodes 0-1-2-3
    nx.set_node_attributes(G, {i: (i, 0) for i in G.nodes()}, "pos")
    for u, v in G.edges():
        G.edges[u, v]["length"] = 1.0

    # Create nodes GeoDataFrame
    pos_dict = {i: (i, 0) for i in G.nodes()}
    nodes_gdf = _create_nodes_gdf(pos_dict, "node_id", None)

    # Test with single center point
    center_points = [Point(0, 0)]
    nodes_within = _compute_nodes_within_distance(
        G, center_points, nodes_gdf, 1.5, "length", "node_id",
    )
    assert nodes_within == {0, 1}

    # Test with multiple center points
    center_points = [Point(0, 0), Point(3, 0)]
    nodes_within = _compute_nodes_within_distance(
        G, center_points, nodes_gdf, 1.5, "length", "node_id",
    )
    assert nodes_within == {0, 1, 2, 3}


def test_normalize_center_points() -> None:
    """Test normalizing different center point input formats."""
    # Single point
    point = Point(0, 0)
    result = _normalize_center_points(point)
    assert result == [point]

    # GeoSeries
    geoseries = gpd.GeoSeries([Point(0, 0), Point(1, 1)])
    result = _normalize_center_points(geoseries)
    assert result is geoseries

    # GeoDataFrame
    geodf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)])
    result = _normalize_center_points(geodf)
    assert result is geodf.geometry


def test_create_empty_result() -> None:
    """Test creating empty results in appropriate formats."""
    # Graph input
    graph_result = _create_empty_result(True, "EPSG:4326")
    assert isinstance(graph_result, nx.Graph)

    # GeoDataFrame input
    gdf_result = _create_empty_result(False, "EPSG:4326")
    assert isinstance(gdf_result, gpd.GeoDataFrame)
    assert gdf_result.crs == "EPSG:4326"


def test_validate_gdf_errors() -> None:
    """Test validation errors for GeoDataFrames."""
    bad_nodes = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])], crs=None)
    with pytest.raises(ValueError, match="Nodes GeoDataFrame must have Point geometries"):
        _validate_gdf(bad_nodes, None)

    bad_edges = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=None)
    with pytest.raises(ValueError, match="Edges GeoDataFrame must have LineString geometries"):
        _validate_gdf(None, bad_edges)


def test_gdf_to_nx_and_roundtrip() -> None:
    """Test conversion from GeoDataFrame to NetworkX and back."""
    nodes = gpd.GeoDataFrame({"attr": [10, 20],
                              "geometry": [Point(0, 0), Point(1, 0)]}, crs="EPSG:3857")
    edges = gpd.GeoDataFrame({"length": [1.0],
                              "geometry": [LineString([(0, 0), (1, 0)])]}, crs="EPSG:3857")
    G = gdf_to_nx(nodes=nodes, edges=edges)
    nodes_out, edges_out = nx_to_gdf(G, nodes=True, edges=True)
    # Round-trip retains data and geometry
    assert set(nodes_out["attr"]) == {10, 20}
    assert edges_out.iloc[0]["length"] == pytest.approx(1.0)


def test_validate_nx_errors() -> None:
    """Test validation errors for NetworkX graphs."""
    G = nx.Graph()
    # missing crs
    with pytest.raises(ValueError, match="Missing CRS in graph attributes"):
        _validate_nx(G, nodes=False)

    G.graph["crs"] = "EPSG:3857"
    # missing pos for nodes
    with pytest.raises(ValueError, match="Missing 'pos' attribute for nodes"):
        _validate_nx(G, nodes=True)

    # missing geometry and pos on edges - create a graph without pos
    G.add_node(1)
    G.add_node(2)
    G.add_edge(1, 2)
    with pytest.raises(ValueError, match="Missing edge geometry and node positions"):
        _validate_nx(G, nodes=False)


def test_filter_graph_by_distance_gdf() -> None:
    """Test filtering graph by distance with GeoDataFrame input."""
    # build simple chain 0-1-2
    edges = gpd.GeoDataFrame({"length": [1.0, 1.0],
                              "geometry": [LineString([(0, 0), (1, 0)]),
                                         LineString([(1, 0), (2, 0)])]}, crs=None)
    gdf = filter_graph_by_distance(edges, Point(0, 0), distance=1.5)
    # should include only first edge
    assert len(gdf) == 1


def test_filter_graph_by_distance_nx() -> None:
    """Test filtering graph by distance with NetworkX input."""
    G = nx.path_graph(3)
    # assign attributes
    nx.set_node_attributes(G, {i: {"pos": (i, 0)} for i in G.nodes()})
    for u, v in G.edges():
        G.edges[u, v]["length"] = 1.0
    result = filter_graph_by_distance(G, Point(2, 0), distance=1.1)
    assert isinstance(result, nx.Graph)
    assert set(result.nodes()) == {1, 2}


@pytest.mark.parametrize(("crs_geom", "crs_barrier"), [("EPSG:4326", "EPSG:3857"), (None, "EPSG:3857")])
def test_create_tessellation_error_mismatch_crs(crs_geom: str | None, crs_barrier: str) -> None:
    """Test tessellation error with mismatched CRS."""
    geom = gpd.GeoSeries([Point(0, 0)], crs=crs_geom)
    barrier = gpd.GeoSeries([Point(1, 1)], crs=crs_barrier)
    with pytest.raises(ValueError, match="CRS mismatch"):
        create_tessellation(geom, barrier)


@pytest.mark.parametrize("crs", ["EPSG:4326"])
def test_create_tessellation_error_geographic(crs: str) -> None:
    """Test tessellation error with geographic CRS."""
    geom = gpd.GeoSeries([Point(0, 0)], crs=crs)
    with pytest.raises(ValueError, match="Geometry is in a geographic CRS"):
        create_tessellation(geom)
