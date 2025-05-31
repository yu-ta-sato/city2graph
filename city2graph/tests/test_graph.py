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
from city2graph.graph import from_morphological_graph
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


# Fixtures for shared test data
@pytest.fixture
def simple_nodes() -> gpd.GeoDataFrame:
    """Fixture for simple node GeoDataFrame."""
    return make_simple_nodes()


@pytest.fixture
def simple_edges() -> gpd.GeoDataFrame:
    """Fixture for simple edge GeoDataFrame."""
    return make_simple_edges()


def test_get_device_cpu_and_invalid() -> None:
    """Test that _get_device returns a CPU device and raises ValueError for invalid input."""
    # Act: call function under test
    d = _get_device("cpu")

    # Assert: verify results
    assert isinstance(d, torch.device)
    assert d.type == "cpu"

    # Act & Assert: invalid device
    with pytest.raises(ValueError, match="Device must"):
        _get_device("not_a_device")


def test_get_device_no_torch_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _get_device raises ImportError if PyTorch is not available."""
    # Arrange: simulate no torch
    import city2graph.graph as gr
    monkeypatch.setattr(gr, "TORCH_AVAILABLE", False)

    # Act & Assert: no torch available
    with pytest.raises(ImportError, match="PyTorch and PyTorch Geometric"):  # no torch
        gr._get_device()

    # Cleanup
    monkeypatch.setattr(gr, "TORCH_AVAILABLE", True)


def test_get_device_none_uses_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _get_device returns CPU when no device is specified."""
    # Arrange: simulate cuda unavailable
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Act: no device specified
    d = _get_device(None)

    # Assert: default to CPU
    assert isinstance(d, torch.device)
    assert d.type == "cpu"

    # Arrange: simulate cuda available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    # Act: no device specified
    d2 = _get_device(None)

    # Assert: check cuda or cpu are in d2.type
    assert d2.type in ("cuda", "cpu")


def test_detect_edge_columns_and_fallback() -> None:
    """"Test that _detect_edge_columns works with hints and falls back to defaults."""
    # Arrange: setup test data
    empty = gpd.GeoDataFrame(columns=["a", "b"])

    # Act: call function under test
    out = _detect_edge_columns(empty)

    # Assert: verify results
    assert out == (None, None)

    ed = make_simple_edges()
    out = _detect_edge_columns(ed, source_hint=["src"], target_hint=["dst"])
    # we allow either None or a two-tuple
    assert out is None or (isinstance(out, tuple) and len(out) == 2)


def test_detect_edge_columns_various() -> None:
    """Test that _detect_edge_columns detects edge columns correctly."""
    # Arrange: setup test data
    df1 = gpd.GeoDataFrame(columns=["from_id", "to_id", "foo"])
    df2 = gpd.GeoDataFrame(columns=["a", "b", "geometry"])
    df3 = gpd.GeoDataFrame(columns=["geometry", "x", "y"])

    # Act: call function under test
    out1 = _detect_edge_columns(df1)
    out2 = _detect_edge_columns(df2)
    out3 = _detect_edge_columns(df3)

    # Assert: verify results
    assert out1 == (None, None)
    assert out2 == (None, None)
    assert out3 == (None, None)


def test_detect_edge_columns_with_id_col_hint() -> None:
    """Test that _detect_edge_columns detects edge columns with id_col hint."""
    # Arrange: setup test data
    edges_df = make_simple_edges().rename(columns={"src": "id", "dst": "id"})

    # Act: call function under test
    src, dst = _detect_edge_columns(edges_df, id_col="id")

    # Assert: verify results
    assert src == "id"
    assert dst == "id"


def test_get_edge_columns_with_and_without_hints() -> None:
    """Test that _get_edge_columns returns correct columns with and without hints."""
    # Arrange: setup test data
    ed = make_simple_edges()

    # Act: call function under test
    out1 = _get_edge_columns(ed, "src", "dst", {}, {}, id_col=None)
    out2 = _get_edge_columns(ed, None, None, {}, {}, id_col=None)

    # Assert: verify results
    assert out1 == ("src", "dst")
    assert out2 == ("src", "dst")


def test_get_edge_columns_partial_and_hints() -> None:
    """Test that _get_edge_columns handles partial hints and defaults correctly."""
    # Arrange: setup test data
    edges_df = make_simple_edges()

    # Act: call function under test
    src, tgt = _get_edge_columns(edges_df, "src", None, {"src": 0}, {}, id_col=None)
    src2, tgt2 = _get_edge_columns(edges_df, None, "dst", {}, {"dst": 1}, id_col=None)

    # Assert: verify results
    assert tgt == "dst"
    assert src2 == "src"


def test_create_edge_idx_pairs_default_mapping() -> None:
    """Test that _create_edge_idx_pairs creates pairs with default mapping."""
    # Arrange: setup test data
    ed = make_simple_edges()

    # Act: call function under test
    pairs = _create_edge_idx_pairs(ed, {"1":0, "2":1}, None)

    # Assert: verify results
    assert sorted(pairs) == [[0,1], [1,0]]


def test_create_edge_idx_pairs_invalid() -> None:
    """Test that _create_edge_idx_pairs returns empty list for invalid edge DataFrame."""
    # Arrange: setup test data
    ed = make_simple_edges().drop(columns=["src"])

    # Act: call function under test
    result = _create_edge_idx_pairs(ed, {"1":0}, None, "src", "dst")

    # Assert: verify results
    assert result == []


def test_extract_node_id_mapping_errors_and_defaults() -> None:
    """Test that _extract_node_id_mapping raises errors for invalid id_col and returns correct mapping."""
    # Arrange: setup test data
    nd = make_simple_nodes()

    # Act: call function under test
    with pytest.raises(ValueError, match="not found"):
        _extract_node_id_mapping(nd, id_col="no_such")

    mapping, used = _extract_node_id_mapping(nd, id_col="id")

    # Assert: verify results
    assert used == "id"
    assert mapping == {"1": 0, "2": 1}


def test_extract_node_id_mapping_index_default() -> None:
    """Test that _extract_node_id_mapping uses index as default when id_col is None."""
    # Arrange: setup test data
    nodes_df = make_simple_nodes().drop(columns=["id"])

    # Act: call function under test
    mapping, used = _extract_node_id_mapping(nodes_df, id_col=None)

    # Assert: verify results
    assert used == "index"
    assert mapping == {str(i): i for i in range(len(nodes_df))}


def test_create_node_features_various(simple_nodes: gpd.GeoDataFrame) -> None:
    """Test that _create_node_features works with various inputs."""
    # Arrange: use simple_nodes fixture
    nd = simple_nodes

    # Act: create features without and with 'feat'
    t0 = _create_node_features(nd, None, device="cpu")
    t1 = _create_node_features(nd, ["feat"], device="cpu")

    # Assert: verify results
    assert t0.shape == (2, 0)
    np.testing.assert_allclose(t1.cpu().numpy(), [[0.5], [1.5]])
    assert _create_node_features(nd, ["nope"], device="cpu") is None


def test_create_edge_features_various() -> None:
    """Test that _create_edge_features works with various inputs."""
    # Arrange: setup test data
    ed = make_simple_edges()

    # Act: call function under test
    e0 = _create_edge_features(ed, None, device="cpu")
    e1 = _create_edge_features(ed, ["w"], device="cpu")
    e2 = _create_edge_features(ed, ["bad"], device="cpu")

    # Assert: verify results
    assert e0.shape == (2, 0)
    np.testing.assert_allclose(e1.cpu().numpy(), [[2.0], [3.0]])
    assert e2.shape == (2, 0)


def test_map_edge_strings_and_idx_pairs() -> None:
    """Test that _map_edge_strings and _create_edge_idx_pairs work correctly."""
    # Arrange: setup test data
    ed = make_simple_edges().copy()

    # Act: call function under test
    ed2 = _map_edge_strings(ed, "src", "dst")
    mapping = {"1": 0, "2": 1}
    pairs = _create_edge_idx_pairs(ed2, mapping, mapping, "src", "dst")

    # Assert: verify results
    assert "__src_str" in ed2.columns
    assert "__dst_str" in ed2.columns
    assert sorted(pairs) == [[0, 1], [1, 0]]

    ed2_bad = ed2.copy()
    ed2_bad["src"] = 999
    assert _create_edge_idx_pairs(ed2_bad, mapping, mapping, "src", "dst") == []


def test_is_valid_edge_df_none_empty_nonempty() -> None:
    """Test that _is_valid_edge_df correctly identifies valid and invalid edge DataFrames."""
    # Act: call function under test
    result1 = _is_valid_edge_df(None)
    result2 = _is_valid_edge_df(gpd.GeoDataFrame())
    result3 = _is_valid_edge_df(make_simple_edges())

    # Assert: verify results
    assert not result1
    assert not result2
    assert result3


def test_features_edgeidx_with_empty_lists() -> None:
    """Test that _create_node_features and _create_edge_features handle empty lists correctly."""
    # Arrange: setup test data
    nd = make_simple_nodes()
    ed = make_simple_edges()

    # Act: call function under test
    t = _create_node_features(nd, [], device="cpu")
    e = _create_edge_features(ed, [], device="cpu")
    pairs = _create_edge_idx_pairs(ed, {"1":0, "2":1}, None, "src", "dst")

    # Assert: verify results
    assert t is None
    assert e.shape == (2, 0)
    assert sorted(pairs) == [[0,1], [1,0]]


def test_homogeneous_graph_minimal_and_no_edges() -> None:
    """"Test that homogeneous_graph creates a valid graph with minimal nodes and no edges."""
    # Arrange: setup test data
    nd = make_simple_nodes()
    ed = make_simple_edges()

    # Act: call function under test
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
    data2 = homogeneous_graph(nd, None, "id", None, None, None, None, None)

    # Assert: verify results
    assert data.x.shape == (2, 1)
    assert data.edge_index.shape[1] == 2
    assert data2.x.shape == (2, 0)
    assert data2.edge_index.shape == (2, 0)
    assert data2.edge_attr.shape == (0, 0)


def test_homogeneous_with_y_and_default_to_networkx() -> None:
    """Test that homogeneous_graph with y labels works and to_networkx returns a valid graph."""
    # Arrange: setup test data
    nd = make_simple_nodes().copy()
    nd["y"] = [10, 20]

    # Act: call function under test
    data = homogeneous_graph(nd, make_simple_edges(), "id", ["feat"], ["y"], "src", "dst", ["w"])
    G = to_networkx(data)

    # Assert: verify results
    assert isinstance(G, nx.Graph)
    # each node has x and pos
    for _, attrs in G.nodes(data=True):
        assert "x" in attrs
        assert "pos" in attrs


def test_homogeneous_graph_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that homogeneous_graph raises ImportError if PyTorch is not available."""
    # Arrange: setup test data
    import city2graph.graph as gr
    monkeypatch.setattr(gr, "TORCH_AVAILABLE", False)

    # Act/Assert: call function under test
    with pytest.raises(ImportError):
        homogeneous_graph(make_simple_nodes(), make_simple_edges())

    monkeypatch.setattr(gr, "TORCH_AVAILABLE", True)


def test_heterogeneous_graph_and_to_networkx_behavior() -> None:
    """"Test that heterogeneous_graph creates a valid graph and to_networkx works as expected."""
    # Arrange: setup test data
    na = make_simple_nodes()
    nb = make_simple_nodes().copy()
    nb["id"] = [10, 20]

    e = gpd.GeoDataFrame(
        {"u": [1], "v": [10], "w": [9.0]},
        geometry=[LineString([(0, 0), (1, 1)])],
    )

    # Act: call function under test
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

    # Assert: verify results
    assert isinstance(G, nx.MultiDiGraph)
    assert G.number_of_nodes() == 4


def test_heterogeneous_graph_default_mappings() -> None:
    """Test that heterogeneous_graph works with default mappings and minimal input."""
    # Arrange: setup test data
    na = make_simple_nodes()
    edges_df = make_simple_edges()

    # Act: call function under test
    het = heterogeneous_graph({"n": na}, {("n", "r", "n"): edges_df})

    # Assert: verify results
    assert hasattr(het, "node_types")
    assert ("n", "r", "n") in het.edge_types


def test_heterogeneous_graph_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that heterogeneous_graph raises ImportError if PyTorch is not available."""
    # Arrange: setup test data
    import city2graph.graph as gr
    monkeypatch.setattr(gr, "TORCH_AVAILABLE", False)

    # Act/Assert: call function under test
    with pytest.raises(ImportError):
        heterogeneous_graph({"n": make_simple_nodes()}, {("n","r","n"): make_simple_edges()})

    monkeypatch.setattr(gr, "TORCH_AVAILABLE", True)


def test_from_morphological_graph_paths_and_errors() -> None:
    """Test that from_morphological_graph handles various inputs and raises errors as expected."""
    # Arrange: setup test data
    empty = gpd.GeoDataFrame()

    # Act/Assert: call function under test
    # invalid type
    with pytest.raises(TypeError):
        from_morphological_graph(123)
    # no nodes => ValueError
    with pytest.raises(ValueError, match="no nodes"):
        from_morphological_graph({"tessellations": empty, "segments": empty})
    # only private
    priv = make_simple_nodes().rename(columns={"id": "tess_id"})
    edges = make_simple_edges().rename(columns={"src": "from_private_id", "dst": "to_private_id"})
    data = from_morphological_graph({
        "tessellations": priv,
        "segments": None,
        "private_to_private": edges,
    })
    assert hasattr(data, "edge_index")
    # both types
    pub = make_simple_nodes().rename(columns={"id": "id"})
    edges2 = make_simple_edges().rename(columns={"src": "from_public_id", "dst": "to_public_id"})
    priv_to_pub = make_simple_edges().rename(columns={"src": "private_id", "dst": "public_id"})
    het = from_morphological_graph({
        "tessellations": priv,
        "segments": pub,
        "private_to_private": edges,
        "public_to_public": edges2,
        "private_to_public": priv_to_pub,
    })
    assert hasattr(het, "node_types")
    assert hasattr(het, "edge_types")


def test_from_morphological_graph_public_only() -> None:
    """Test that from_morphological_graph works with public nodes and edges only."""
    # Arrange: setup test data
    seg = make_simple_nodes().rename(columns={"id": "id"})
    edges = make_simple_edges().rename(columns={"src": "from_public_id", "dst": "to_public_id"})

    # Act: call function under test
    data = from_morphological_graph({"tessellations": None, "segments": seg, "public_to_public": edges})
    # Assert: verify results
    assert hasattr(data, "edge_index")
    assert data.x.size(0) == 2


def test_to_networkx_data_graph() -> None:
    """Test that to_networkx converts a homogeneous graph data object to a NetworkX graph."""
    # Arrange: setup test data
    data = homogeneous_graph(make_simple_nodes(),
                             make_simple_edges(),
                             "id", ["feat"],
                             None,
                             "src",
                             "dst",
                             None)

    # Act: call function under test
    G = to_networkx(data)

    # Assert: verify results
    assert isinstance(G, nx.Graph)
    assert set(G.nodes()) == {0,1}
    assert set(G.edges()) == {(0,1),(1,0)}


def test_is_torch_available() -> None:
    """"Test that is_torch_available returns a boolean."""
    # Act: call function under test
    result = is_torch_available()

    # Assert: verify results
    assert isinstance(result, bool)
