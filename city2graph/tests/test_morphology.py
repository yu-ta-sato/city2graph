"""Tests for morphological network graph functions."""

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph.graph import from_morphological_network
from city2graph.graph import heterogeneous_graph
from city2graph.graph import homogeneous_graph
from city2graph.graph import is_torch_available
from city2graph.graph import to_networkx


def create_nodes() -> gpd.GeoDataFrame:
    """Create a sample GeoDataFrame with nodes for testing."""
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3], "value": [0.1, 0.2, 0.3], "y": [0, 1, 0]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
    )


def create_edges() -> gpd.GeoDataFrame:
    """Create a sample GeoDataFrame with edges for testing."""
    return gpd.GeoDataFrame(
        {"from_id": [1, 2], "to_id": [2, 3], "weight": [1.0, 2.0]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
    )


def test_homogeneous_graph_simple() -> None:
    """Test creation of a simple homogeneous graph."""
    nodes = create_nodes()
    edges = create_edges()
    data = homogeneous_graph(
        nodes_gdf=nodes,
        edges_gdf=edges,
        node_id_col="id",
        node_feature_cols=["value"],
        node_label_cols=["y"],
        edge_source_col="from_id",
        edge_target_col="to_id",
        edge_feature_cols=["weight"],
    )
    assert data.x.shape == (3, 1)
    assert data.y.shape == (3, 1)
    assert data.edge_index.shape == (2, 2)
    assert data.edge_attr.shape == (2, 1)
    assert hasattr(data, "pos")
    assert data.crs == nodes.crs or nodes.crs is None


def test_heterogeneous_graph_simple() -> None:
    """Test creation of a simple heterogeneous graph."""
    nodes_a = create_nodes()
    nodes_b = create_nodes()
    nodes_b = nodes_b.copy()
    nodes_b["id"] = [10, 20, 30]
    edges_ab = gpd.GeoDataFrame(
        {"source": [1, 2], "target": [10, 20]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
    )
    nodes_dict = {"a": nodes_a, "b": nodes_b}
    edges_dict = {("a", "link", "b"): edges_ab}
    data = heterogeneous_graph(
        nodes_dict=nodes_dict,
        edges_dict=edges_dict,
        node_id_cols={"a": "id", "b": "id"},
        node_feature_cols={"a": ["value"], "b": ["value"]},
        edge_source_cols={("a", "link", "b"): "source"},
        edge_target_cols={("a", "link", "b"): "target"},
        edge_feature_cols=None,
    )
    assert "a" in data.node_types
    assert "b" in data.node_types
    assert data["a"].x.shape == (3, 1)
    assert data["b"].x.shape == (3, 1)
    assert data[("a", "link", "b")].edge_index.shape[1] == 2


def test_from_morphological_network_homo() -> None:
    """Test creation of homogeneous graph from morphological network."""
    tess = create_nodes().rename(columns={"id": "tess_id"})
    private_to_private = create_edges().rename(
        columns={"from_id": "from_private_id", "to_id": "to_private_id"},
    )
    net = {
        "tessellations": tess,
        "segments": gpd.GeoDataFrame(),
        "private_to_private": private_to_private,
        "public_to_public": gpd.GeoDataFrame(),
        "private_to_public": gpd.GeoDataFrame(),
    }
    data = from_morphological_network(net)
    # Should return homogeneous graph Data or HeteroData
    assert hasattr(data, "edge_index")
    assert data.edge_index.shape[0] == 2


def test_to_networkx_homo() -> None:
    """Test conversion of homogeneous graph to NetworkX."""
    nodes = create_nodes()
    edges = create_edges()
    data = homogeneous_graph(
        nodes, edges, "id", ["value"], ["y"], "from_id", "to_id", ["weight"],
    )
    G = to_networkx(data)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2


def test_is_torch_available() -> None:
    """Test that is_torch_available returns a boolean value."""
    assert isinstance(is_torch_available(), bool)
