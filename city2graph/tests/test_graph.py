"""Tests for graph module functionality."""

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest
import torch
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph.graph import _create_edge_features
from city2graph.graph import _create_edge_idx_pairs
from city2graph.graph import _create_node_features
from city2graph.graph import _detect_edge_columns
from city2graph.graph import _extract_node_id_mapping
from city2graph.graph import _get_device
from city2graph.graph import _get_edge_columns
from city2graph.graph import _is_valid_edge_df
from city2graph.graph import _map_edge_strings
from city2graph.graph import heterogeneous_graph
from city2graph.graph import homogeneous_graph
from city2graph.graph import is_torch_available
from city2graph.graph import to_networkx


def make_simple_nodes() -> gpd.GeoDataFrame:
    """Return a tiny nodes GeoDataFrame with id and feat columns."""
    return gpd.GeoDataFrame(
        {"id": [1, 2], "feat": [0.5, 1.5]},
        geometry=[Point(0, 0), Point(1, 1)],
    )


def make_simple_edges() -> gpd.GeoDataFrame:
    """Return a tiny edges GeoDataFrame with src, dst and w columns."""
    return gpd.GeoDataFrame(
        {"src": [1, 2], "dst": [2, 1], "w": [2.0, 3.0]},
        geometry=[
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (0, 0)]),
        ],
    )


def test_get_device_cpu_and_invalid() -> None:
    """Test that _get_device returns a CPU device and raises ValueError for invalid input."""
    d = _get_device("cpu")
    assert isinstance(d, torch.device)
    assert d.type == "cpu"

    with pytest.raises(ValueError, match="Device must"):
        _get_device("not_a_device")


def test_detect_edge_columns_and_fallback() -> None:
    """"Test that _detect_edge_columns works with hints and falls back to defaults."""
    empty = gpd.GeoDataFrame(columns=["a", "b"])
    assert _detect_edge_columns(empty) == (None, None)

    ed = make_simple_edges()
    out = _detect_edge_columns(ed, source_hint=["src"], target_hint=["dst"])
    # we allow either None or a two-tuple
    assert out is None or (isinstance(out, tuple) and len(out) == 2)


def test_get_edge_columns_with_and_without_hints() -> None:
    """Test that _get_edge_columns returns correct columns with and without hints."""
    ed = make_simple_edges()
    assert _get_edge_columns(ed, "src", "dst", {}, {}, id_col=None) == ("src", "dst")
    assert _get_edge_columns(ed, None, None, {}, {}, id_col=None) == ("src", "dst")


def test_extract_node_id_mapping_errors_and_defaults() -> None:
    """Test that _extract_node_id_mapping raises errors for invalid id_col and returns correct mapping."""
    nd = make_simple_nodes()
    with pytest.raises(ValueError, match="not found"):
        _extract_node_id_mapping(nd, id_col="no_such")

    mapping, used = _extract_node_id_mapping(nd, id_col="id")
    assert used == "id"
    assert mapping == {"1": 0, "2": 1}


def test_create_node_features_various() -> None:
    """"Test that _create_node_features works with various inputs."""
    nd = make_simple_nodes()
    t0 = _create_node_features(nd, None, device="cpu")
    assert t0.shape == (2, 0)

    t1 = _create_node_features(nd, ["feat"], device="cpu")
    np.testing.assert_allclose(t1.cpu().numpy(), [[0.5], [1.5]])

    assert _create_node_features(nd, ["nope"], device="cpu") is None


def test_create_edge_features_various() -> None:
    """Test that _create_edge_features works with various inputs."""
    ed = make_simple_edges()
    e0 = _create_edge_features(ed, None, device="cpu")
    assert e0.shape == (2, 0)

    e1 = _create_edge_features(ed, ["w"], device="cpu")
    np.testing.assert_allclose(e1.cpu().numpy(), [[2.0], [3.0]])

    e2 = _create_edge_features(ed, ["bad"], device="cpu")
    assert e2.shape == (2, 0)


def test_map_edge_strings_and_idx_pairs() -> None:
    """Test that _map_edge_strings and _create_edge_idx_pairs work correctly."""
    ed = make_simple_edges().copy()
    ed2 = _map_edge_strings(ed, "src", "dst")
    assert "__src_str" in ed2.columns
    assert "__dst_str" in ed2.columns

    mapping = {"1": 0, "2": 1}
    pairs = _create_edge_idx_pairs(ed2, mapping, mapping, "src", "dst")
    assert sorted(pairs) == [[0, 1], [1, 0]]

    ed2_bad = ed2.copy()
    ed2_bad["src"] = 999
    assert _create_edge_idx_pairs(ed2_bad, mapping, mapping, "src", "dst") == []


def test_is_valid_edge_df_none_empty_nonempty() -> None:
    """Test that _is_valid_edge_df correctly identifies valid and invalid edge DataFrames."""
    assert not _is_valid_edge_df(None)
    assert not _is_valid_edge_df(gpd.GeoDataFrame())
    assert _is_valid_edge_df(make_simple_edges())


def test_homogeneous_graph_minimal_and_no_edges() -> None:
    """"Test that homogeneous_graph creates a valid graph with minimal nodes and no edges."""
    nd = make_simple_nodes()
    ed = make_simple_edges()

    data = homogeneous_graph(
        nodes_gdf=nd,
        edges_gdf=ed,
        node_id_col="id",
        node_feature_cols=["feat"],
        node_label_cols=None,
        edge_source_col="src",
        edge_target_col="dst",
        edge_feature_cols=["w"],
    )
    assert data.x.shape == (2, 1)
    assert data.edge_index.shape[1] == 2

    data2 = homogeneous_graph(nd, None, "id", None, None, None, None, None)
    assert data2.x.shape == (2, 0)
    assert data2.edge_index.shape == (2, 0)
    assert data2.edge_attr.shape == (0, 0)


def test_heterogeneous_graph_and_to_networkx_behavior() -> None:
    """"Test that heterogeneous_graph creates a valid graph and to_networkx works as expected."""
    na = make_simple_nodes()
    nb = make_simple_nodes().copy()
    nb["id"] = [10, 20]

    e = gpd.GeoDataFrame(
        {"u": [1], "v": [10], "w": [9.0]},
        geometry=[LineString([(0, 0), (1, 1)])],
    )
    het = heterogeneous_graph(
        nodes_dict={"A": na, "B": nb},
        edges_dict={("A", "rel", "B"): e},
        node_id_cols={"A": "id", "B": "id"},
        node_feature_cols={"A": ["feat"], "B": ["feat"]},
        node_label_cols=None,
        edge_source_cols={("A", "rel", "B"): "u"},
        edge_target_cols={("A", "rel", "B"): "v"},
        edge_feature_cols={("A", "rel", "B"): ["w"]},
    )

    G = to_networkx(het)
    assert isinstance(G, nx.MultiDiGraph)
    assert G.number_of_nodes() == 4


def test_is_torch_available() -> None:
    """"Test that is_torch_available returns a boolean."""
    assert isinstance(is_torch_available(), bool)
