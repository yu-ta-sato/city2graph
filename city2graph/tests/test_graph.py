"""Tests for graph module functionality."""

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
    """Create a sample GeoDataFrame with nodes for testing.

    Returns
    -------
        gpd.GeoDataFrame: A GeoDataFrame containing test node data with id, value, y columns and Point
            geometries.
    """
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3], "value": [0.1, 0.2, 0.3], "y": [0, 1, 0]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
    )


def create_edges() -> gpd.GeoDataFrame:
    """Create a sample GeoDataFrame with edges for testing.

    Returns
    -------
        gpd.GeoDataFrame: A GeoDataFrame containing test edge data with from_id, to_id, weight columns and
            LineString geometries.
    """
    return gpd.GeoDataFrame(
        {"from_id": [1, 2], "to_id": [2, 3], "weight": [1.0, 2.0]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
    )


def test_homogeneous_graph_simple() -> None:
    """Test homogeneous graph creation with simple node and edge data."""
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
    if data.x.shape != (3, 1):
        msg = f"Expected data.x.shape to be (3, 1), got {data.x.shape}"
        raise AssertionError(msg)
    if data.y.shape != (3, 1):
        msg = f"Expected data.y.shape to be (3, 1), got {data.y.shape}"
        raise AssertionError(msg)
    if data.edge_index.shape != (2, 2):
        msg = f"Expected data.edge_index.shape to be (2, 2), got {data.edge_index.shape}"
        raise AssertionError(msg)
    if data.edge_attr.shape != (2, 1):
        msg = f"Expected data.edge_attr.shape to be (2, 1), got {data.edge_attr.shape}"
        raise AssertionError(msg)


def test_homogeneous_graph_no_edges() -> None:
    """Test homogeneous graph creation with nodes only (no edges)."""
    nodes = create_nodes()
    data = homogeneous_graph(
        nodes_gdf=nodes,
        edges_gdf=None,
        node_id_col="id",
        node_feature_cols=None,
        node_label_cols=None,
        edge_source_col=None,
        edge_target_col=None,
        edge_feature_cols=None,
    )
    assert data.x.shape == (3, 0)
    assert data.edge_index.shape == (2, 0)
    assert data.edge_attr.shape == (0, 0)


def test_heterogeneous_graph_simple() -> None:
    """Test heterogeneous graph creation with simple node and edge data."""
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
    """Test homogeneous graph creation from morphological network with only tessellations."""
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


def test_from_morphological_network_hetero() -> None:
    """Test heterogeneous graph creation from morphological network with both tessellations and segments."""
    tess = create_nodes().rename(columns={"id": "tess_id"})
    seg = create_nodes().rename(columns={"id": "id"})
    private_to_private = create_edges().rename(
        columns={"from_id": "from_private_id", "to_id": "to_private_id"},
    )
    public_to_public = create_edges().rename(
        columns={"from_id": "from_public_id", "to_id": "to_public_id"},
    )
    private_to_public = gpd.GeoDataFrame(
        {"private_id": [1], "public_id": [1]}, geometry=[LineString([(0, 0), (1, 1)])],
    )
    net = {
        "tessellations": tess,
        "segments": seg,
        "private_to_private": private_to_private,
        "public_to_public": public_to_public,
        "private_to_public": private_to_public,
    }
    data = from_morphological_network(net)
    assert hasattr(data, "node_types")
    assert "private" in data.node_types
    assert "public" in data.node_types
    assert data["private"].x.shape == (3, 0)
    assert data["public"].x.shape == (3, 0)
    assert data[("private", "connects_to", "private")].edge_index.shape == (2, 2)
    assert data[("public", "connects_to", "public")].edge_index.shape == (2, 2)
    assert data[("private", "adjacent_to", "public")].edge_index.shape == (2, 1)


def test_to_networkx_homo() -> None:
    """Test conversion of homogeneous graph to NetworkX format."""
    nodes = create_nodes()
    edges = create_edges()
    data = homogeneous_graph(
        nodes, edges, "id", ["value"], ["y"], "from_id", "to_id", ["weight"],
    )
    G = to_networkx(data)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2


def test_to_networkx_hetero() -> None:
    """Test conversion of heterogeneous graph to NetworkX format."""
    nodes_a = create_nodes()
    nodes_b = create_nodes().copy()
    nodes_b["id"] = [10, 20, 30]
    edges_ab = gpd.GeoDataFrame(
        {"source": [1, 2], "target": [10, 20]},
        geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
    )
    data = heterogeneous_graph(
        nodes_dict={"a": nodes_a, "b": nodes_b},
        edges_dict={("a", "link", "b"): edges_ab},
        node_id_cols={"a": "id", "b": "id"},
        node_feature_cols={"a": ["value"], "b": ["value"]},
        edge_source_cols={("a", "link", "b"): "source"},
        edge_target_cols={("a", "link", "b"): "target"},
        edge_feature_cols=None,
    )
    G = to_networkx(data)
    assert isinstance(G, nx.MultiDiGraph)
    assert G.number_of_nodes() == 6
    assert G.number_of_edges() == 2


def test_is_torch_available() -> None:
    """Test that is_torch_available returns a boolean value."""
    assert isinstance(is_torch_available(), bool)
