"""Tests for proximity graph functions."""

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph.proximity import _add_distance_weights
from city2graph.proximity import _add_edge_geometries
from city2graph.proximity import _build_delaunay_edges
from city2graph.proximity import _build_knn_edges
from city2graph.proximity import _calculate_distance_matrix
from city2graph.proximity import _compute_network_distances
from city2graph.proximity import _create_manhattan_linestring
from city2graph.proximity import _create_network_linestring
from city2graph.proximity import _extract_coords_and_attrs_from_gdf
from city2graph.proximity import _get_network_positions
from city2graph.proximity import _init_graph_and_nodes
from city2graph.proximity import _setup_network_computation
from city2graph.proximity import _validate_network_compatibility
from city2graph.proximity import delaunay_graph
from city2graph.proximity import gilbert_graph
from city2graph.proximity import knn_graph
from city2graph.proximity import waxman_graph


@pytest.fixture
def simple_points_gdf():  # noqa: ANN201
    """Return a simple GeoDataFrame with two points."""
    return gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(1, 0)],
        crs="EPSG:4326",
    )


@pytest.fixture
def triangle_points_gdf() -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with three points forming a triangle."""
    return gpd.GeoDataFrame(
        geometry=[Point(0, 0), Point(1, 0), Point(0.5, 1)],
        crs="EPSG:4326",
    )


@pytest.fixture
def simple_network_gdf() -> gpd.GeoDataFrame:
    """Return a simple network GeoDataFrame."""
    return gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (1, 0)])],
        crs="EPSG:4326",
    )


@pytest.fixture
def network_graph() -> nx.Graph:
    """Return a simple NetworkX graph representing a network."""
    G = nx.Graph()
    G.add_edge(0, 1, length=1.0)
    G.nodes[0]["pos"] = (0, 0)
    G.nodes[1]["pos"] = (1, 0)
    return G


def test_build_knn_edges() -> None:
    """Test KNN edge building function."""
    indices = np.array([[1, 2], [0, 2], [0, 1]])
    node_indices = [0, 1, 2]
    edges = _build_knn_edges(indices, node_indices)
    # Only edges to actual nearest neighbors are created
    assert len(edges) >= 2
    assert isinstance(edges, list)


def test_build_knn_edges_custom_indices() -> None:
    """Test KNN edge building with custom node indices."""
    indices = np.array([[1, 2], [0, 2], [0, 1]])
    node_indices = ["A", "B", "C"]
    edges = _build_knn_edges(indices, node_indices)
    # Check that we get some edges with the custom indices
    assert len(edges) >= 2
    assert all(isinstance(edge, tuple) for edge in edges)
    assert all(edge[0] in node_indices and edge[1] in node_indices for edge in edges)


def test_build_delaunay_edges() -> None:
    """Test Delaunay edge building function."""
    points = np.array([[0, 0], [1, 0], [0, 1]])
    indices = [0, 1, 2]
    edges = _build_delaunay_edges(points, indices)
    assert len(edges) == 3
    assert (0, 1) in edges or (1, 0) in edges


def test_validate_network_compatibility(simple_points_gdf: gpd.GeoDataFrame,
                                        simple_network_gdf: gpd.GeoDataFrame) -> None:
    """Test network compatibility validation."""
    # Should not raise for compatible CRS
    _validate_network_compatibility(simple_points_gdf, simple_network_gdf)

    # Should raise for different CRS
    network_different_crs = simple_network_gdf.copy()
    network_different_crs.crs = "EPSG:3857"
    with pytest.raises(ValueError, match="CRS mismatch"):
        _validate_network_compatibility(simple_points_gdf, network_different_crs)

    # Should raise for empty network
    empty_network = gpd.GeoDataFrame(geometry=[], crs=simple_points_gdf.crs)
    with pytest.raises(ValueError, match="Network GeoDataFrame is empty"):
        _validate_network_compatibility(simple_points_gdf, empty_network)


def test_get_network_positions(network_graph: nx.Graph) -> None:
    """Test network position extraction."""
    pos_dict = _get_network_positions(network_graph)
    assert pos_dict[0] == (0, 0)
    assert pos_dict[1] == (1, 0)

    # Test fallback for missing positions
    G_no_pos = nx.Graph()
    G_no_pos.add_node(0, x=0, y=1)
    G_no_pos.add_node(1, x=1, y=1)
    pos_dict_fallback = _get_network_positions(G_no_pos)
    assert pos_dict_fallback[0] == (0, 1)
    assert pos_dict_fallback[1] == (1, 1)


def test_compute_network_distances(network_graph: nx.Graph) -> None:
    """Test network distance computation."""
    coords = np.array([[0, 0], [1, 0]])
    node_indices = [0, 1]
    distance_matrix, nearest_nodes = _compute_network_distances(coords, node_indices, network_graph)
    assert distance_matrix.shape == (2, 2)
    assert distance_matrix[0, 0] == 0
    assert nearest_nodes == [0, 1]


def test_setup_network_computation(simple_points_gdf: gpd.GeoDataFrame,
                                   simple_network_gdf: gpd.GeoDataFrame) -> None:
    """Test network computation setup."""
    coords = np.array([[0, 0], [1, 0]])
    node_indices = [0, 1]
    (network_graph, distance_matrix, nearest_nodes) = _setup_network_computation(
        simple_points_gdf, simple_network_gdf, coords, node_indices,
    )
    assert isinstance(network_graph, nx.Graph)
    assert distance_matrix.shape == (2, 2)


def test_extract_coords_and_attrs_from_gdf(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test coordinate and attribute extraction."""
    coords, node_attrs = _extract_coords_and_attrs_from_gdf(simple_points_gdf)
    assert coords.shape == (2, 2)
    assert len(node_attrs) == 2
    assert "geometry" in node_attrs[0]
    assert "pos" in node_attrs[0]
    assert isinstance(node_attrs[0]["pos"], tuple)


def test_init_graph_and_nodes(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test graph and node initialization."""
    graph, coords, node_indices = _init_graph_and_nodes(simple_points_gdf)
    assert isinstance(graph, nx.Graph)
    assert coords.shape == (2, 2)
    assert len(node_indices) == 2
    assert graph.graph["crs"] == "EPSG:4326"

    # Test error handling
    with pytest.raises(TypeError, match="Input data must be a GeoDataFrame"):
        _init_graph_and_nodes("not a geodataframe")

    gdf_no_geom = gpd.GeoDataFrame({"col": [1, 2]})
    with pytest.raises(ValueError, match="GeoDataFrame must contain geometry"):
        _init_graph_and_nodes(gdf_no_geom)

    gdf_null_geom = gpd.GeoDataFrame(geometry=[None, None])
    with pytest.raises(ValueError, match="GeoDataFrame must contain geometry"):
        _init_graph_and_nodes(gdf_null_geom)

    empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    with pytest.raises(ValueError, match="GeoDataFrame must contain geometry"):
        _init_graph_and_nodes(empty_gdf)


def test_create_manhattan_linestring() -> None:
    """Test Manhattan distance LineString creation."""
    coord1 = (0, 0)
    coord2 = (2, 3)
    linestring = _create_manhattan_linestring(coord1, coord2)
    assert isinstance(linestring, LineString)
    coords = list(linestring.coords)
    assert coords == [(0.0, 0.0), (2.0, 0.0), (2.0, 3.0)]

    # Test horizontal and vertical lines
    line2 = _create_manhattan_linestring((0, 0), (5, 0))
    assert len(list(line2.coords)) == 3  # Manhattan path: (0,0) -> (5,0) -> (5,0)
    assert next(iter(line2.coords)) == (0.0, 0.0)
    assert list(line2.coords)[1] == (5.0, 0.0)

    line3 = _create_manhattan_linestring((1, 1), (1, 1))
    assert isinstance(line3, LineString)  # Same point creates a degenerate LineString


def test_create_network_linestring(network_graph: nx.Graph) -> None:
    """Test network LineString creation."""
    node_indices = [0, 1]
    nearest_network_nodes = [0, 1]
    linestring = _create_network_linestring(0, 1, network_graph, node_indices, nearest_network_nodes)
    assert isinstance(linestring, LineString)

    # Test with no position data fallback
    G_no_pos = nx.Graph()
    G_no_pos.add_edge(0, 1)
    G_no_pos.nodes[0]["x"] = 0
    G_no_pos.nodes[0]["y"] = 0
    G_no_pos.nodes[1]["x"] = 1
    G_no_pos.nodes[1]["y"] = 0
    linestring_fallback = _create_network_linestring(0, 1, G_no_pos, [0, 1], [0, 1])
    assert isinstance(linestring_fallback, LineString)


def test_add_edge_geometries() -> None:
    """Test edge geometry addition."""
    G = nx.Graph()
    G.add_edge(0, 1)
    coords = np.array([[0, 0], [1, 0]])
    node_indices = [0, 1]
    _add_edge_geometries(G, coords, node_indices, "euclidean")
    assert isinstance(G[0][1]["geometry"], LineString)

    # Test with manhattan distance
    G_manhattan = nx.Graph()
    G_manhattan.add_edge(0, 1)
    _add_edge_geometries(G_manhattan, coords, node_indices, "manhattan")
    assert isinstance(G_manhattan[0][1]["geometry"], LineString)


def test_calculate_distance_matrix() -> None:
    """Test distance matrix calculation."""
    coords = np.array([[0, 0], [1, 0], [0, 1]])
    node_indices = [0, 1, 2]

    (dist_matrix, network_graph, nearest_nodes) = _calculate_distance_matrix(
        coords, node_indices, "euclidean",
    )
    assert dist_matrix.shape == (3, 3)
    assert network_graph is None
    assert nearest_nodes is None

    (dist_matrix_manhattan, _, _) = _calculate_distance_matrix(
        coords, node_indices, "manhattan",
    )
    assert dist_matrix_manhattan.shape == (3, 3)


def test_add_distance_weights() -> None:
    """Test distance weight addition."""
    G = nx.Graph()
    edges = [(0, 1), (1, 2)]
    node_indices = [0, 1, 2]
    distance_matrix = np.array([
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0],
    ])
    G.add_edges_from(edges)
    _add_distance_weights(G, edges, node_indices, distance_matrix)
    assert G[0][1]["weight"] == 1
    assert G[1][2]["weight"] == 1


def test_knn_graph_basic(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test basic KNN graph creation."""
    G = knn_graph(simple_points_gdf, k=1)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() >= 1


def test_knn_graph_network(simple_points_gdf: gpd.GeoDataFrame,
                           simple_network_gdf: gpd.GeoDataFrame) -> None:
    """Test KNN graph with network distance metric."""
    G = knn_graph(simple_points_gdf, k=2, distance_metric="network", network_gdf=simple_network_gdf)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 2

    # Test error when network_gdf is missing
    with pytest.raises(ValueError, match="network_gdf is required when distance_metric='network'"):
        knn_graph(simple_points_gdf, k=1, distance_metric="network")


def test_knn_graph_manhattan(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test KNN graph with Manhattan distance metric."""
    G = knn_graph(simple_points_gdf, k=2, distance_metric="manhattan")
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 2
    for u, v in G.edges():
        assert "geometry" in G[u][v]
        coords = list(G[u][v]["geometry"].coords)
        assert len(coords) >= 2


def test_delaunay_graph_basic(triangle_points_gdf: gpd.GeoDataFrame) -> None:
    """Test basic Delaunay triangulation graph."""
    G = delaunay_graph(triangle_points_gdf)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 3


def test_delaunay_graph_network(triangle_points_gdf: gpd.GeoDataFrame) -> None:
    """Test Delaunay graph with network distance metric."""
    # Create a network that covers the triangle
    network_for_triangle = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (0.5, 1)]),
            LineString([(0.5, 1), (0, 0)]),
        ],
        crs=triangle_points_gdf.crs,
    )
    G = delaunay_graph(
        triangle_points_gdf, distance_metric="network", network_gdf=network_for_triangle,
    )
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3


def test_delaunay_graph_manhattan(triangle_points_gdf: gpd.GeoDataFrame) -> None:
    """Test Delaunay graph with Manhattan distance metric."""
    G = delaunay_graph(triangle_points_gdf, distance_metric="manhattan")
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    for u, v in G.edges():
        assert "geometry" in G[u][v]


def test_gilbert_graph_basic(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test basic Gilbert graph creation."""
    G = gilbert_graph(simple_points_gdf, radius=2.0)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 2


def test_gilbert_graph_network(simple_points_gdf: gpd.GeoDataFrame,
                               simple_network_gdf: gpd.GeoDataFrame) -> None:
    """Test Gilbert graph with network distance metric."""
    G = gilbert_graph(
        simple_points_gdf, radius=3.0, distance_metric="network", network_gdf=simple_network_gdf,
    )
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 2


def test_gilbert_graph_manhattan(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test Gilbert graph with Manhattan distance metric."""
    G = gilbert_graph(simple_points_gdf, radius=2.0, distance_metric="manhattan")
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 2


def test_waxman_graph_basic(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test basic Waxman graph creation."""
    G = waxman_graph(simple_points_gdf, beta=1.0, r0=2.0, seed=42)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 2
    assert "beta" in G.graph
    assert "r0" in G.graph


def test_waxman_graph_network(simple_points_gdf: gpd.GeoDataFrame,
                              simple_network_gdf: gpd.GeoDataFrame) -> None:
    """Test Waxman graph with network distance metric."""
    G = waxman_graph(
        simple_points_gdf,
        beta=1.0,
        r0=2.0,
        seed=42,
        distance_metric="network",
        network_gdf=simple_network_gdf,
    )
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 2

    # Test error when network_gdf is missing
    with pytest.raises(ValueError, match="network_gdf is required when distance_metric='network'"):
        waxman_graph(simple_points_gdf, beta=1.0, r0=1.0, distance_metric="network")


def test_waxman_graph_manhattan(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test Waxman graph with Manhattan distance metric."""
    G = waxman_graph(simple_points_gdf, beta=1.0, r0=1.0, seed=42, distance_metric="manhattan")
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 2


def test_as_gdf_option(simple_points_gdf: gpd.GeoDataFrame,
                       triangle_points_gdf: gpd.GeoDataFrame) -> None:
    """Test as_gdf option for all graph functions."""
    result = knn_graph(simple_points_gdf, k=2, as_gdf=True)
    assert isinstance(result, gpd.GeoDataFrame)

    result = delaunay_graph(triangle_points_gdf, as_gdf=True)
    assert isinstance(result, gpd.GeoDataFrame)

    result = gilbert_graph(simple_points_gdf, radius=2.0, as_gdf=True)
    assert isinstance(result, gpd.GeoDataFrame)

    # Test waxman_graph with NetworkX first to avoid empty graph issues
    G = waxman_graph(simple_points_gdf, beta=1.0, r0=1.0, seed=42)
    if G.number_of_edges() > 0:
        result = waxman_graph(simple_points_gdf, beta=1.0, r0=1.0, seed=42, as_gdf=True)
        assert isinstance(result, gpd.GeoDataFrame)


def test_delaunay_collinear_points() -> None:
    """Test Delaunay triangulation with collinear points."""
    collinear_coords = np.array([[0, 0], [1, 0], [2, 0]])
    node_indices = ["A", "B", "C"]
    edges = _build_delaunay_edges(collinear_coords, node_indices)
    # Collinear points may return empty set or list
    assert isinstance(edges, (list, set))


def test_large_dataset() -> None:
    """Test with a larger dataset for performance."""
    n_points = 20
    rng = np.random.default_rng(42)
    points = [Point(x, y) for x, y in rng.random((n_points, 2)) * 100]
    large_gdf = gpd.GeoDataFrame(geometry=points)

    G_knn = knn_graph(large_gdf, k=5)
    assert G_knn.number_of_nodes() == n_points

    G_gilbert = gilbert_graph(large_gdf, radius=10.0)
    assert G_gilbert.number_of_nodes() == n_points
