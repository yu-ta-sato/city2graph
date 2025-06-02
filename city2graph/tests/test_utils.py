"""Tests for the utils module."""
import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph.utils import _compute_nodes_within_distance
from city2graph.utils import _create_empty_result
from city2graph.utils import _create_nodes_gdf
from city2graph.utils import _extract_dual_graph_nodes
from city2graph.utils import _extract_node_connections
from city2graph.utils import _extract_node_positions
from city2graph.utils import _find_additional_connections
from city2graph.utils import _get_nearest_node
from city2graph.utils import _normalize_center_points
from city2graph.utils import _validate_gdf
from city2graph.utils import _validate_nx
from city2graph.utils import create_tessellation
from city2graph.utils import dual_graph
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
    with pytest.raises(ValueError, match="Edges GeoDataFrame must have LineString or MultiLineString geometries"):
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


def test_dual_graph_basic() -> None:
    """Test basic dual graph functionality."""
    # Create simple line network
    lines = gpd.GeoDataFrame({
        "id": ["L1", "L2"],
    }, geometry=[
        LineString([(0, 0), (1, 0)]),
        LineString([(1, 0), (2, 0)]),
    ], crs="EPSG:4326")
    
    nodes_gdf, connections = dual_graph(lines, id_col="id")
    
    assert isinstance(nodes_gdf, gpd.GeoDataFrame)
    assert isinstance(connections, dict)
    assert len(nodes_gdf) == 2
    assert "L1" in connections
    assert "L2" in connections


def test_dual_graph_errors() -> None:
    """Test dual graph error handling."""
    # Test invalid input type
    with pytest.raises(TypeError, match="Input must be a GeoDataFrame"):
        dual_graph("not_a_gdf")
    
    # Test invalid tolerance type
    lines = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 0)])], crs="EPSG:4326")
    with pytest.raises(TypeError, match="Tolerance must be a number"):
        dual_graph(lines, tolerance="invalid")
    
    # Test negative tolerance
    with pytest.raises(ValueError, match="Tolerance must be non-negative"):
        dual_graph(lines, tolerance=-1)


def test_dual_graph_empty() -> None:
    """Test dual graph with empty input."""
    empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    with pytest.warns(RuntimeWarning, match="Input GeoDataFrame is empty"):
        nodes_gdf, connections = dual_graph(empty_gdf)
    
    assert isinstance(nodes_gdf, gpd.GeoDataFrame)
    assert connections == {}


def test_dual_graph_invalid_geometries() -> None:
    """Test dual graph with invalid geometries."""
    # Test with non-LineString geometries
    mixed_gdf = gpd.GeoDataFrame({
        "id": ["P1", "L1"],
    }, geometry=[
        Point(0, 0),  # Invalid
        LineString([(0, 0), (1, 0)]),
    ], crs="EPSG:4326")
    
    with pytest.warns(RuntimeWarning):
        nodes_gdf, connections = dual_graph(mixed_gdf, id_col="id")
    
    assert isinstance(nodes_gdf, gpd.GeoDataFrame)
    assert isinstance(connections, dict)


def test_dual_graph_null_geometries() -> None:
    """Test dual graph with null geometries."""
    null_gdf = gpd.GeoDataFrame({
        "id": ["L1", "L2"],
    }, geometry=[
        LineString([(0, 0), (1, 0)]),
        None,
    ], crs="EPSG:4326")
    
    with pytest.warns(RuntimeWarning, match="Found null geometries"):
        nodes_gdf, connections = dual_graph(null_gdf, id_col="id")
    
    assert isinstance(nodes_gdf, gpd.GeoDataFrame)
    assert isinstance(connections, dict)


def test_extract_dual_graph_nodes() -> None:
    """Test extracting nodes from dual graph."""
    # Create simple graph with coordinate nodes
    g = nx.Graph()
    g.add_node((0, 0), id="node1")
    g.add_node((1, 1), id="node2")
    
    result = _extract_dual_graph_nodes(g, "id", "EPSG:4326")
    assert isinstance(result, gpd.GeoDataFrame)
    assert len(result) == 2
    assert result.crs == "EPSG:4326"


def test_extract_dual_graph_nodes_empty() -> None:
    """Test extracting nodes from empty graph."""
    empty_graph = nx.Graph()
    result = _extract_dual_graph_nodes(empty_graph, "id", "EPSG:4326")
    
    assert isinstance(result, gpd.GeoDataFrame)
    assert result.empty
    assert list(result.columns) == ["id", "geometry"]


def test_extract_node_connections() -> None:
    """Test extracting node connections from graph."""
    g = nx.Graph()
    g.add_node(1, id="A")
    g.add_node(2, id="B")
    g.add_edge(1, 2)
    
    connections = _extract_node_connections(g, "id")
    assert isinstance(connections, dict)
    assert "A" in connections
    assert "B" in connections
    assert "B" in connections["A"]
    assert "A" in connections["B"]


def test_find_additional_connections() -> None:
    """Test finding additional connections between lines."""
    # Create lines with close endpoints
    lines = gpd.GeoDataFrame({
        "id": ["A", "B"],
    }, geometry=[
        LineString([(0, 0), (1, 0)]),
        LineString([(1.001, 0), (2, 0)]),  # Close endpoint
    ], crs="EPSG:4326")
    
    connections = _find_additional_connections(lines, "id", tolerance=0.01)
    assert isinstance(connections, dict)
    assert "A" in connections
    assert "B" in connections


def test_find_additional_connections_empty() -> None:
    """Test finding connections with empty input."""
    empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    result = _find_additional_connections(empty_gdf, "id", 1.0)
    assert result == {}


def test_find_additional_connections_no_linestrings() -> None:
    """Test finding connections with non-LineString geometries."""
    from shapely.geometry import MultiLineString
    
    multi_gdf = gpd.GeoDataFrame({
        "id": ["M1"],
    }, geometry=[
        MultiLineString([[(0, 0), (1, 1)], [(1, 1), (2, 2)]]),
    ], crs="EPSG:4326")
    
    result = _find_additional_connections(multi_gdf, "id", 1.0)
    assert isinstance(result, dict)
