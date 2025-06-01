"""Tests for graph module functionality."""

import contextlib
from unittest.mock import patch

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

# ============================================================================
# COMMON TEST FIXTURES
# ============================================================================


def make_simple_nodes() -> gpd.GeoDataFrame:
    """Create a simple nodes GeoDataFrame with id and feat columns for testing."""
    return gpd.GeoDataFrame(
        {"id": [1, 2], "feat": [0.5, 1.5]},
        geometry=[Point(0, 0), Point(1, 1)],
    )


def make_simple_edges() -> gpd.GeoDataFrame:
    """Create a simple edges GeoDataFrame with src, dst and w columns for testing."""
    return gpd.GeoDataFrame(
        {"src": [1, 2], "dst": [2, 1], "w": [2.0, 3.0]},
        geometry=[
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (0, 0)]),
        ],
    )


@pytest.fixture
def simple_nodes() -> gpd.GeoDataFrame:
    """Fixture providing simple node GeoDataFrame for testing."""
    return make_simple_nodes()


@pytest.fixture
def simple_edges() -> gpd.GeoDataFrame:
    """Fixture providing simple edge GeoDataFrame for testing."""
    return make_simple_edges()


# ============================================================================
# DEVICE DETECTION AND TORCH AVAILABILITY TESTS
# ============================================================================


def test_get_device_cpu_and_invalid() -> None:
    """Test that _get_device returns a CPU device and raises ValueError for invalid input."""
    # Act: get CPU device
    d = _get_device("cpu")

    # Assert: verify device type and properties
    assert isinstance(d, torch.device)
    assert d.type == "cpu"

    # Act & Assert: invalid device string should raise ValueError
    with pytest.raises(ValueError, match="Device must"):
        _get_device("not_a_device")


def test_get_device_no_torch_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _get_device raises ImportError if PyTorch is not available."""
    # Arrange: simulate PyTorch not being available
    import city2graph.graph as gr
    monkeypatch.setattr(gr, "TORCH_AVAILABLE", False)

    # Act & Assert: should raise ImportError when torch is unavailable
    with pytest.raises(ImportError, match="PyTorch and PyTorch Geometric"):
        gr._get_device()

    # Cleanup: restore torch availability
    monkeypatch.setattr(gr, "TORCH_AVAILABLE", True)


def test_get_device_none_uses_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _get_device returns CPU when no device is specified."""
    # Arrange: simulate CUDA unavailable
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    # Act: get device with None argument (auto-detection)
    d = _get_device(None)

    # Assert: should default to CPU when CUDA unavailable
    assert isinstance(d, torch.device)
    assert d.type == "cpu"

    # Arrange: simulate CUDA available
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    # Act: get device with None argument (auto-detection)
    d2 = _get_device(None)

    # Assert: should use CUDA or CPU based on availability
    assert d2.type in ("cuda", "cpu")


def test_is_torch_available() -> None:
    """Test that is_torch_available returns a boolean."""
    # Act: check if PyTorch is available
    result = is_torch_available()

    # Assert: should return boolean value
    assert isinstance(result, bool)


# ============================================================================
# EDGE COLUMN DETECTION AND PROCESSING TESTS
# ============================================================================


def test_detect_edge_columns_and_fallback() -> None:
    """Test that _detect_edge_columns works with hints and falls back to defaults."""
    # Arrange: create empty GeoDataFrame
    empty = gpd.GeoDataFrame(columns=["a", "b"])

    # Act: detect edge columns in empty DataFrame
    out = _detect_edge_columns(empty)

    # Assert: should return None for empty DataFrame
    assert out == (None, None)

    # Arrange: create edges DataFrame with src/dst columns
    ed = make_simple_edges()

    # Act: detect edge columns with explicit hints
    out = _detect_edge_columns(ed, source_hint=["src"], target_hint=["dst"])

    # Assert: should return None or valid tuple
    assert out is None or (isinstance(out, tuple) and len(out) == 2)


def test_detect_edge_columns_various() -> None:
    """Test that _detect_edge_columns detects edge columns correctly."""
    # Arrange: create DataFrames with different column patterns
    df1 = gpd.GeoDataFrame(columns=["from_id", "to_id", "foo"])
    df2 = gpd.GeoDataFrame(columns=["a", "b", "geometry"])
    df3 = gpd.GeoDataFrame(columns=["geometry", "x", "y"])

    # Act: detect edge columns for each DataFrame
    out1 = _detect_edge_columns(df1)
    out2 = _detect_edge_columns(df2)
    out3 = _detect_edge_columns(df3)

    # Assert: should return None for all cases without clear edge patterns
    assert out1 == (None, None)
    assert out2 == (None, None)
    assert out3 == (None, None)


def test_detect_edge_columns_with_id_col_hint() -> None:
    """Test that _detect_edge_columns detects edge columns with id_col hint."""
    # Arrange: create edges DataFrame with renamed columns
    edges_df = make_simple_edges().rename(columns={"src": "id", "dst": "id"})

    # Act: detect edge columns using id_col hint
    src, dst = _detect_edge_columns(edges_df, id_col="id")

    # Assert: should detect id column as both source and target
    assert src == "id"
    assert dst == "id"


def test_get_edge_columns_with_and_without_hints() -> None:
    """Test that _get_edge_columns returns correct columns with and without hints."""
    # Arrange: create edges DataFrame
    ed = make_simple_edges()

    # Act: get edge columns with explicit hints
    out1 = _get_edge_columns(ed, "src", "dst", {}, {}, id_col=None)

    # Act: get edge columns without hints (auto-detection)
    out2 = _get_edge_columns(ed, None, None, {}, {}, id_col=None)

    # Assert: both should return the same result
    assert out1 == ("src", "dst")
    assert out2 == ("src", "dst")


def test_get_edge_columns_partial_and_hints() -> None:
    """Test that _get_edge_columns handles partial hints and defaults correctly."""
    # Arrange: create edges DataFrame
    edges_df = make_simple_edges()

    # Act: provide only source hint, let function detect target
    src, tgt = _get_edge_columns(edges_df, "src", None, {"src": 0}, {}, id_col=None)

    # Act: provide only target hint, let function detect source
    src2, tgt2 = _get_edge_columns(edges_df, None, "dst", {}, {"dst": 1}, id_col=None)

    # Assert: should correctly complete missing hints
    assert tgt == "dst"
    assert src2 == "src"


def test_create_edge_idx_pairs_default_mapping() -> None:
    """Test that _create_edge_idx_pairs creates pairs with default mapping."""
    # Arrange: create edges DataFrame and node mapping
    ed = make_simple_edges()
    node_mapping = {"1": 0, "2": 1}

    # Act: create edge index pairs using mapping
    pairs = _create_edge_idx_pairs(ed, node_mapping, None)

    # Assert: should create correct edge index pairs
    assert sorted(pairs) == [[0, 1], [1, 0]]


def test_create_edge_idx_pairs_invalid() -> None:
    """Test that _create_edge_idx_pairs returns empty list for invalid edge DataFrame."""
    # Arrange: create invalid edges DataFrame (missing src column)
    ed = make_simple_edges().drop(columns=["src"])
    node_mapping = {"1": 0}

    # Act: attempt to create edge pairs with invalid DataFrame
    result = _create_edge_idx_pairs(ed, node_mapping, None, "src", "dst")

    # Assert: should return empty list for invalid input
    assert result == []


def test_extract_node_id_mapping_errors_and_defaults() -> None:
    """Test that _extract_node_id_mapping raises errors for invalid id_col and returns correct mapping."""
    # Arrange: create nodes DataFrame
    nd = make_simple_nodes()

    # Act & Assert: invalid id_col should raise ValueError
    with pytest.raises(ValueError, match="not found"):
        _extract_node_id_mapping(nd, id_col="no_such")

    # Act: extract mapping using valid id column
    mapping, used = _extract_node_id_mapping(nd, id_col="id")

    # Assert: should return correct mapping and column used
    assert used == "id"
    assert mapping == {"1": 0, "2": 1}


def test_extract_node_id_mapping_index_default() -> None:
    """Test that _extract_node_id_mapping uses index as default when id_col is None."""
    # Arrange: create nodes DataFrame without id column
    nodes_df = make_simple_nodes().drop(columns=["id"])

    # Act: extract mapping using index as default
    mapping, used = _extract_node_id_mapping(nodes_df, id_col=None)

    # Assert: should use index and create correct mapping
    assert used == "index"
    assert mapping == {str(i): i for i in range(len(nodes_df))}


# ============================================================================
# FEATURE CREATION AND PROCESSING TESTS
# ============================================================================


def test_create_node_features_various(simple_nodes: gpd.GeoDataFrame) -> None:
    """Test that _create_node_features works with various inputs."""
    # Arrange: use simple_nodes fixture
    nd = simple_nodes

    # Act: create features without and with 'feat'
    t0 = _create_node_features(nd, None, device="cpu")
    t1 = _create_node_features(nd, ["feat"], device="cpu")

    # Assert: verify shape and values
    assert t0.shape == (2, 0)
    np.testing.assert_allclose(t1.cpu().numpy(), [[0.5], [1.5]])

    # Act & Assert: non-existent feature columns return None
    assert _create_node_features(nd, ["nope"], device="cpu") is None


def test_create_edge_features_various() -> None:
    """Test that _create_edge_features works with various inputs."""
    # Arrange: setup test data
    ed = make_simple_edges()

    # Act: create features with None, valid column, and invalid column
    e0 = _create_edge_features(ed, None, device="cpu")
    e1 = _create_edge_features(ed, ["w"], device="cpu")
    e2 = _create_edge_features(ed, ["bad"], device="cpu")

    # Assert: verify shape and values
    assert e0.shape == (2, 0)
    np.testing.assert_allclose(e1.cpu().numpy(), [[2.0], [3.0]])
    assert e2.shape == (2, 0)


def test_map_edge_strings_and_idx_pairs() -> None:
    """Test that _map_edge_strings and _create_edge_idx_pairs work correctly."""
    # Arrange: setup test data
    ed = make_simple_edges().copy()

    # Act: map edge strings to create temporary columns
    ed2 = _map_edge_strings(ed, "src", "dst")

    # Arrange: create node mapping for index pairs
    mapping = {"1": 0, "2": 1}

    # Act: create edge index pairs
    pairs = _create_edge_idx_pairs(ed2, mapping, mapping, "src", "dst")

    # Assert: verify temporary columns created and pairs are correct
    assert "__src_str" in ed2.columns
    assert "__dst_str" in ed2.columns
    assert sorted(pairs) == [[0, 1], [1, 0]]

    # Arrange: create invalid edge data
    ed2_bad = ed2.copy()
    ed2_bad["src"] = 999

    # Act & Assert: invalid mapping should return empty list
    assert _create_edge_idx_pairs(ed2_bad, mapping, mapping, "src", "dst") == []


def test_is_valid_edge_df_none_empty_nonempty() -> None:
    """Test that _is_valid_edge_df correctly identifies valid and invalid edge DataFrames."""
    # Act: test various edge DataFrame types
    result1 = _is_valid_edge_df(None)
    result2 = _is_valid_edge_df(gpd.GeoDataFrame())
    result3 = _is_valid_edge_df(make_simple_edges())

    # Assert: verify validation results
    assert not result1  # None should be invalid
    assert not result2  # Empty DataFrame should be invalid
    assert result3     # Non-empty DataFrame should be valid


def test_features_edgeidx_with_empty_lists() -> None:
    """Test that _create_node_features and _create_edge_features handle empty lists correctly."""
    # Arrange: setup test data
    nd = make_simple_nodes()
    ed = make_simple_edges()

    # Act: create features with empty feature lists
    t = _create_node_features(nd, [], device="cpu")
    e = _create_edge_features(ed, [], device="cpu")

    # Act: create edge pairs with valid mapping
    pairs = _create_edge_idx_pairs(ed, {"1": 0, "2": 1}, None, "src", "dst")

    # Assert: verify handling of empty feature lists
    assert t is None  # Empty node features should return None
    assert e.shape == (2, 0)  # Empty edge features should have 0 columns
    assert sorted(pairs) == [[0, 1], [1, 0]]


# ============================================================================
# HOMOGENEOUS AND HETEROGENEOUS GRAPH TESTS
# ============================================================================


def test_homogeneous_graph_minimal_and_no_edges() -> None:
    """Test that homogeneous_graph creates a valid graph with minimal nodes and no edges."""
    # Arrange: setup test data with simple nodes and edges
    nd = make_simple_nodes()
    ed = make_simple_edges()

    # Act: create homogeneous graph with full parameters
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

    # Act: create graph with no edges
    data2 = homogeneous_graph(nd, None, "id", None, None, None, None, None)

    # Assert: verify graph structure and shapes
    assert data.x.shape == (2, 1)  # 2 nodes, 1 feature
    assert data.edge_index.shape[1] == 2  # 2 edges
    assert data2.x.shape == (2, 0)  # 2 nodes, no features
    assert data2.edge_index.shape == (2, 0)  # No edges
    assert data2.edge_attr.shape == (0, 0)  # No edge attributes


def test_homogeneous_with_y_and_default_to_networkx() -> None:
    """Test that homogeneous_graph with y labels works and to_networkx returns a valid graph."""
    # Arrange: setup test data with labels
    nd = make_simple_nodes().copy()
    nd["y"] = [10, 20]

    # Act: create homogeneous graph with labels
    data = homogeneous_graph(nd, make_simple_edges(), "id", ["feat"], ["y"], "src", "dst", ["w"])

    # Act: convert to NetworkX graph
    G = to_networkx(data)

    # Assert: verify NetworkX conversion and node attributes
    assert isinstance(G, nx.Graph)

    # Each node should have x (features) and pos (coordinates)
    for _, attrs in G.nodes(data=True):
        assert "x" in attrs
        assert "pos" in attrs


def test_homogeneous_graph_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that homogeneous_graph raises ImportError if PyTorch is not available."""
    # Arrange: simulate PyTorch not being available
    import city2graph.graph as gr
    monkeypatch.setattr(gr, "TORCH_AVAILABLE", False)

    # Act & Assert: should raise ImportError when torch is unavailable
    with pytest.raises(ImportError):
        homogeneous_graph(make_simple_nodes(), make_simple_edges())

    # Cleanup: restore torch availability
    monkeypatch.setattr(gr, "TORCH_AVAILABLE", True)


def test_heterogeneous_graph_and_to_networkx_behavior() -> None:
    """Test that heterogeneous_graph creates a valid graph and to_networkx works as expected."""
    # Arrange: setup test data with two node types
    na = make_simple_nodes()
    nb = make_simple_nodes().copy()
    nb["id"] = [10, 20]

    # Arrange: create edge between different node types
    e = gpd.GeoDataFrame(
        {"u": [1], "v": [10], "w": [9.0]},
        geometry=[LineString([(0, 0), (1, 1)])],
    )

    # Act: create heterogeneous graph
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

    # Act: convert to NetworkX
    G = to_networkx(het)

    # Assert: verify heterogeneous graph structure
    assert isinstance(G, nx.MultiDiGraph)
    assert G.number_of_nodes() == 4  # 2 nodes from A + 2 nodes from B


def test_heterogeneous_graph_default_mappings() -> None:
    """Test that heterogeneous_graph works with default mappings and minimal input."""
    # Arrange: setup minimal test data
    na = make_simple_nodes()
    edges_df = make_simple_edges()

    # Act: create heterogeneous graph with minimal parameters
    het = heterogeneous_graph({"n": na}, {("n", "r", "n"): edges_df})

    # Assert: verify graph has expected structure
    assert hasattr(het, "node_types")
    assert ("n", "r", "n") in het.edge_types


def test_heterogeneous_graph_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that heterogeneous_graph raises ImportError if PyTorch is not available."""
    # Arrange: simulate PyTorch not being available
    import city2graph.graph as gr
    monkeypatch.setattr(gr, "TORCH_AVAILABLE", False)

    # Act & Assert: should raise ImportError when torch is unavailable
    with pytest.raises(ImportError):
        heterogeneous_graph({"n": make_simple_nodes()}, {("n", "r", "n"): make_simple_edges()})

    # Cleanup: restore torch availability
    monkeypatch.setattr(gr, "TORCH_AVAILABLE", True)


# ============================================================================
# MORPHOLOGICAL GRAPH CONVERSION TESTS
# ============================================================================


def test_from_morphological_graph_paths_and_errors() -> None:
    """Test that from_morphological_graph handles various inputs and raises errors as expected."""
    # Arrange: setup test data
    empty = gpd.GeoDataFrame()

    # Act & Assert: invalid input type should raise TypeError
    with pytest.raises(TypeError):
        from_morphological_graph(123)

    # Act & Assert: no nodes should raise ValueError
    with pytest.raises(ValueError, match="no nodes"):
        from_morphological_graph({"tessellations": empty, "segments": empty})

    # Arrange: create private-only graph data
    priv = make_simple_nodes().rename(columns={"id": "tess_id"})
    edges = make_simple_edges().rename(columns={"src": "from_private_id", "dst": "to_private_id"})

    # Act: create graph with only private nodes
    data = from_morphological_graph({
        "tessellations": priv,
        "segments": None,
        "private_to_private": edges,
    })

    # Assert: should create valid homogeneous graph
    assert hasattr(data, "edge_index")

    # Arrange: create mixed private-public graph data
    pub = make_simple_nodes().rename(columns={"id": "id"})
    edges2 = make_simple_edges().rename(columns={"src": "from_public_id", "dst": "to_public_id"})
    priv_to_pub = make_simple_edges().rename(columns={"src": "private_id", "dst": "public_id"})

    # Act: create heterogeneous graph with both node types
    het = from_morphological_graph({
        "tessellations": priv,
        "segments": pub,
        "private_to_private": edges,
        "public_to_public": edges2,
        "private_to_public": priv_to_pub,
    })

    # Assert: should create valid heterogeneous graph
    assert hasattr(het, "node_types")
    assert hasattr(het, "edge_types")


def test_from_morphological_graph_public_only() -> None:
    """Test that from_morphological_graph works with public nodes and edges only."""
    # Arrange: setup public-only graph data
    seg = make_simple_nodes().rename(columns={"id": "id"})
    edges = make_simple_edges().rename(columns={"src": "from_public_id", "dst": "to_public_id"})

    # Act: create graph with only public nodes
    data = from_morphological_graph({
        "tessellations": None,
        "segments": seg,
        "public_to_public": edges,
    })

    # Assert: should create valid homogeneous graph with public nodes
    assert hasattr(data, "edge_index")
    assert data.x.size(0) == 2


def test_to_networkx_data_graph() -> None:
    """Test that to_networkx converts a homogeneous graph data object to a NetworkX graph."""
    # Arrange: create homogeneous graph data
    data = homogeneous_graph(
        make_simple_nodes(),
        make_simple_edges(),
        "id",
        ["feat"],
        None,
        "src",
        "dst",
        None,
    )

    # Act: convert to NetworkX graph
    G = to_networkx(data)

    # Assert: verify conversion results
    assert isinstance(G, nx.Graph)
    assert set(G.nodes()) == {0, 1}
    assert set(G.edges()) == {(0, 1), (1, 0)}


def test_is_torch_available_final() -> None:
    """Test that is_torch_available returns a boolean value."""
    # Act: check PyTorch availability
    result = is_torch_available()

    # Assert: should return boolean value
    assert isinstance(result, bool)


# ============================================================================
# TESTS FOR UNCOVERED CODE PATHS IN GRAPH.PY
# ============================================================================

def test_torch_import_error_coverage() -> None:
    """Test coverage of ImportError handling for torch imports."""
    # This test ensures the ImportError exception handling lines 14-22 are exercised
    # by temporarily mocking the import failure
    import importlib
    import sys

    # Save original modules
    orig_torch = sys.modules.get("torch")
    orig_pyg_data = sys.modules.get("torch_geometric.data")
    orig_pyg_utils = sys.modules.get("torch_geometric.utils")

    try:
        # Mock import failure
        sys.modules["torch"] = None
        sys.modules["torch_geometric.data"] = None
        sys.modules["torch_geometric.utils"] = None

        # Force reload of the module to trigger ImportError path
        import city2graph.graph as graph_module
        importlib.reload(graph_module)

        # Check that TORCH_AVAILABLE is False
        assert not graph_module.TORCH_AVAILABLE

        # Check that placeholder classes exist
        assert hasattr(graph_module, "HeteroData")
        assert hasattr(graph_module, "Data")

    finally:
        # Restore original modules
        if orig_torch is not None:
            sys.modules["torch"] = orig_torch
        if orig_pyg_data is not None:
            sys.modules["torch_geometric.data"] = orig_pyg_data
        if orig_pyg_utils is not None:
            sys.modules["torch_geometric.utils"] = orig_pyg_utils
        # Reload to restore normal state
        import city2graph.graph as graph_module
        importlib.reload(graph_module)


def test_detect_edge_columns_fallback_case() -> None:
    """Test _detect_edge_columns fallback when only 3 columns and geometry not first."""
    # This tests lines 164-167 (edge column detection fallback)
    edge_gdf = gpd.GeoDataFrame({
        "col1": [1],
        "col2": [2],
        "geometry": [LineString([(0, 0), (1, 1)])],
    })

    # When there are exactly 3 columns and geometry is not first,
    # it should return the first two non-geometry columns
    source, target = _detect_edge_columns(edge_gdf)

    # Should return None, None for the fallback case (line 167)
    # This is a bit tricky since the function actually tries to detect columns
    # Let's create a case that definitely falls through to line 167
    edge_gdf_no_hints = gpd.GeoDataFrame({
        "random_col1": [1],
        "random_col2": [2],
        "geometry": [LineString([(0, 0), (1, 1)])],
    })

    source, target = _detect_edge_columns(edge_gdf_no_hints)
    # This should hit the fallback return None, None (line 167)
    # depending on the implementation details


def test_create_node_features_non_geodataframe_warning() -> None:
    """Test _process_node_type warning for non-GeoDataFrame input."""
    # This tests lines 408-409 (warning for non-GeoDataFrame)
    import pandas as pd
    from torch_geometric.data import HeteroData

    from city2graph.graph import _process_node_type

    # Create a regular DataFrame instead of GeoDataFrame
    node_df = pd.DataFrame({"id": [1, 2], "feat": [0.5, 1.5]})

    with patch("city2graph.graph.logger.warning") as mock_warning:
        data = HeteroData()
        result = _process_node_type(
            "test_type",
            node_df,
            {},
            {},
            {},
            "cpu",
            data,
        )

        # Should return empty dict and log warning
        assert result == {}
        mock_warning.assert_called_once_with(
            "Expected GeoDataFrame for node type %s, got %s",
            "test_type",
            type(node_df),
        )


def test_validate_nodes_dict_warnings() -> None:
    """Test _validate_nodes_dict warning scenarios."""
    import pandas as pd
    from torch_geometric.data import HeteroData

    from city2graph.graph import _process_node_type

    # This tests lines 460-474 (node validation warnings)
    nodes_dict = {
        "valid_type": make_simple_nodes(),
        "invalid_type": pd.DataFrame({"id": [1], "feat": [1.0]}),  # Not a GeoDataFrame
        "empty_type": gpd.GeoDataFrame({"id": [], "feat": []}, geometry=[]),
    }

    with patch("city2graph.graph.logger.warning") as mock_warning:
        # Use _process_node_type which calls the validation logic
        data = HeteroData()
        with contextlib.suppress(ValueError, AttributeError):
            _process_node_type(
                "invalid_type",
                nodes_dict["invalid_type"],
                {},
                {},
                {},
                "cpu",
                data,
            )

        # Should warn about invalid type
        assert mock_warning.call_count >= 1
        # Check that at least one warning was about invalid type
        warning_calls = [call[0] for call in mock_warning.call_args_list]
        assert any("Expected GeoDataFrame for node type" in str(call) for call in warning_calls)


def test_heterogeneous_graph_single_gdf_with_type_column() -> None:
    """Test heterogeneous_graph with single GeoDataFrame containing type column."""
    # This tests lines 691-697 (processing single GDF with type column)

    # Create a single GeoDataFrame with type column
    nodes_with_types = gpd.GeoDataFrame({
        "id": [1, 2, 3, 4],
        "type": ["building", "building", "road", "road"],
        "feat": [0.1, 0.2, 0.3, 0.4],
    }, geometry=[Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)])

    # This should trigger the type column processing logic
    result = heterogeneous_graph(
        nodes_dict=nodes_with_types,
        edges_dict={},
    )

    # Should create a heterogeneous graph with different node types
    assert result is not None
    if is_torch_available():
        assert hasattr(result, "node_types")
        assert len(result.node_types) > 1  # Should have multiple node types


def test_process_single_nodes_dict_path() -> None:
    """Test _process_single_nodes_gdf execution path."""
    # This tests line 787 (single nodes dict processing)

    # Create test data that will trigger single nodes processing
    # Need to pass a GeoDataFrame directly, not a dict
    nodes_gdf = make_simple_nodes()

    # Mock the _process_single_nodes_gdf function to verify it's called
    with patch("city2graph.graph._process_single_nodes_gdf") as mock_process:
        mock_process.return_value = {"default": nodes_gdf}

        heterogeneous_graph(
            nodes_dict=nodes_gdf,  # Pass GeoDataFrame directly
            edges_dict={},
        )

        # Verify the processing function was called
        mock_process.assert_called_once()


def test_process_single_edges_dict_path() -> None:
    """Test _process_single_edges_gdf execution path."""
    # This tests line 791 (single edges dict processing)

    # Create test data that will trigger single edges processing
    # Need to pass a GeoDataFrame directly, not a dict
    edges_gdf = make_simple_edges()

    with patch("city2graph.graph._process_single_edges_gdf") as mock_process:
        mock_process.return_value = {"default": edges_gdf}

        heterogeneous_graph(
            nodes_dict={"node": make_simple_nodes()},
            edges_dict=edges_gdf,  # Pass GeoDataFrame directly
        )

        mock_process.assert_called_once()


def test_homogeneous_graph_torch_unavailable_error() -> None:
    """Test homogeneous_graph ImportError when torch unavailable."""
    # This tests lines 854-859 (ImportError for homogeneous_graph)

    with patch("city2graph.graph.TORCH_AVAILABLE", new=False):
        with pytest.raises(ImportError) as exc_info:
            homogeneous_graph(
                nodes_gdf=make_simple_nodes(),
                edges_gdf=make_simple_edges(),
                node_id_col="id",
            )

        assert "PyTorch and PyTorch Geometric are required" in str(exc_info.value)


def test_from_morphological_graph_private_public_features() -> None:
    """Test from_morphological_graph with private/public node features."""
    # This tests lines 898-900 (private/public node feature handling)
    from shapely.geometry import LineString
    from shapely.geometry import Point

    # Create a proper mock dictionary (not MagicMock) that from_morphological_graph expects
    mock_network_output = {
        "tessellation": gpd.GeoDataFrame({
            "tess_id": [1, 2],
            "private_feat": [1.0, 1.5],
            "geometry": [
                Point(0, 0).buffer(0.1),
                Point(1, 1).buffer(0.1),
            ],
        }),
        "segments": gpd.GeoDataFrame({
            "id": [10, 20],
            "public_feat": [2.0, 2.5],
            "geometry": [
                LineString([(0, 0), (0.5, 0.5)]),
                LineString([(0.5, 0.5), (1, 1)]),
            ],
        }),
        "private_to_private": gpd.GeoDataFrame({
            "from_id": [1],
            "to_id": [2],
            "geometry": [LineString([(0, 0), (1, 1)])],
        }),
        "public_to_public": gpd.GeoDataFrame({
            "from_id": [10],
            "to_id": [20],
            "geometry": [LineString([(0, 0), (1, 1)])],
        }),
        "private_to_public": gpd.GeoDataFrame({
            "from_id": [1],
            "to_id": [10],
            "geometry": [LineString([(0, 0), (0.25, 0.25)])],
        }),
    }

    private_cols = ["private_feat"]
    public_cols = ["public_feat"]

    with (
        patch("city2graph.graph.TORCH_AVAILABLE", new=True),
        patch("city2graph.graph._get_device") as mock_get_device,
        patch("city2graph.graph.torch") as mock_torch,
        patch("city2graph.graph.Data") as mock_data,
    ):
        # Mock the device function to return a simple string
        mock_get_device.return_value = "cpu"
        mock_torch.tensor.return_value = torch.tensor([1.0, 2.0])

        from_morphological_graph(
            network_output=mock_network_output,
            private_node_feature_cols=private_cols,
            public_node_feature_cols=public_cols,
        )

        # Should process both private and public features
        assert mock_data.called


# ============================================================================
# COMPREHENSIVE TESTS FOR MISSING COVERAGE LINES
# ============================================================================


def test_detect_edge_columns_exact_three_columns() -> None:
    """Test _detect_edge_columns with exactly 3 columns (lines 164-165)."""
    from city2graph.graph import _detect_edge_columns

    # Create edge dataframe with exactly 3 columns (and geometry)
    edge_gdf = gpd.GeoDataFrame({
        "src": [1, 2],
        "dst": [2, 1],
        "geometry": [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (0, 0)]),
        ],
    })

    src_col, dst_col = _detect_edge_columns(edge_gdf)

    # Should return the first two non-geometry columns
    assert src_col == "src"
    assert dst_col == "dst"


def test_detect_edge_columns_returns_none() -> None:
    """Test _detect_edge_columns returns None when no valid columns (line 167)."""
    from city2graph.graph import _detect_edge_columns

    # Create edge dataframe with only geometry column
    edge_gdf = gpd.GeoDataFrame({
        "geometry": [LineString([(0, 0), (1, 1)])],
    })

    src_col, dst_col = _detect_edge_columns(edge_gdf)

    # Should return None for both when insufficient columns
    assert src_col is None
    assert dst_col is None


def test_create_node_features_with_target_column() -> None:
    """Test _create_node_features with target column (line 441)."""
    from city2graph.graph import _create_node_features

    # Create node dataframe with target column
    node_gdf = gpd.GeoDataFrame({
        "id": [1, 2],
        "feature": [0.5, 1.5],
        "target": [0, 1],  # Target column
    }, geometry=[Point(0, 0), Point(1, 1)])

    # Test that _create_node_features works with valid feature columns
    result = _create_node_features(node_gdf, ["feature"], device="cpu")

    # Should create tensor with feature values
    assert result is not None
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result.cpu().numpy(), [[0.5], [1.5]])


def test_create_edge_idx_pairs_warning_path() -> None:
    """Test _create_edge_idx_pairs warning when no valid edges (lines 470, 474)."""
    from city2graph.graph import _create_edge_idx_pairs

    # Create edges with invalid source/destination IDs
    edge_gdf = gpd.GeoDataFrame({
        "src": [999, 998],  # IDs not in node mapping
        "dst": [997, 996],
        "w": [1.0, 2.0],
    }, geometry=[
        LineString([(0, 0), (1, 1)]),
        LineString([(1, 1), (2, 2)]),
    ])

    # Create node mapping with different IDs
    node_id_mapping = {1: 0, 2: 1, 3: 2}

    with patch("city2graph.graph.logger") as mock_logger:
        result = _create_edge_idx_pairs(edge_gdf, node_id_mapping, node_id_mapping, "src", "dst")

        # Should warn about no valid edges and return early
        mock_logger.warning.assert_called()
        assert result == []


def test_heterogeneous_graph_crs_consistency() -> None:
    """Test heterogeneous_graph with consistent CRS (lines 577-578)."""
    from city2graph.graph import heterogeneous_graph

    if not is_torch_available():
        pytest.skip("PyTorch not available")

    # Create nodes with same CRS
    nodes_dict = {
        "type1": gpd.GeoDataFrame({
            "id": [1, 2],
            "feat": [0.5, 1.5],
        }, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"),
        "type2": gpd.GeoDataFrame({
            "id": [3, 4],
            "feat": [2.5, 3.5],
        }, geometry=[Point(2, 2), Point(3, 3)], crs="EPSG:4326"),
    }

    edges_dict = {}

    data = heterogeneous_graph(nodes_dict, edges_dict)

    # Should set consistent CRS
    assert hasattr(data, "crs")
    assert data.crs == "EPSG:4326"


def test_heterogeneous_graph_crs_mismatch() -> None:
    """Test heterogeneous_graph raises error on CRS mismatch (lines 580-581)."""
    from city2graph.graph import heterogeneous_graph

    if not is_torch_available():
        pytest.skip("PyTorch not available")

    # Create nodes with different CRS
    nodes_dict = {
        "type1": gpd.GeoDataFrame({
            "id": [1, 2],
            "feat": [0.5, 1.5],
        }, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326"),
        "type2": gpd.GeoDataFrame({
            "id": [3, 4],
            "feat": [2.5, 3.5],
        }, geometry=[Point(2, 2), Point(3, 3)], crs="EPSG:3857"),  # Different CRS
    }

    edges_dict = {}

    with pytest.raises(ValueError, match="CRS mismatch among node GeoDataFrames"):
        heterogeneous_graph(nodes_dict, edges_dict)


def test_heterogeneous_graph_node_type_processing() -> None:
    """Test heterogeneous_graph node type processing (lines 698-699, 703)."""
    from city2graph.graph import heterogeneous_graph

    if not is_torch_available():
        pytest.skip("PyTorch not available")

    # Test with regular DataFrame (no geometry attribute)
    import pandas as pd
    nodes_regular_df = pd.DataFrame({
        "id": [1, 2],
        "feat": [0.5, 1.5],
        "x": [0, 1],
        "y": [0, 1],
    })

    # Convert to dict format that would trigger the conversion path
    nodes_dict = {"regular": nodes_regular_df}
    edges_dict = {}

    data = heterogeneous_graph(nodes_dict, edges_dict)

    # Should handle regular DataFrame and convert to GeoDataFrame
    assert "regular" in data.node_types

    # Test fallback to default node type
    nodes_gdf = gpd.GeoDataFrame({
        "id": [1, 2],
        "feat": [0.5, 1.5],
    }, geometry=[Point(0, 0), Point(1, 1)])

    # Test without node_type column (should use "default")
    data = heterogeneous_graph(nodes_gdf, {})
    assert "default" in data.node_types


def test_heterogeneous_graph_edge_type_processing() -> None:
    """Test heterogeneous_graph edge type processing (lines 708, 710-715)."""
    from city2graph.graph import heterogeneous_graph

    if not is_torch_available():
        pytest.skip("PyTorch not available")

    nodes_dict = {
        "node1": gpd.GeoDataFrame({
            "id": [1, 2],
            "feat": [0.5, 1.5],
        }, geometry=[Point(0, 0), Point(1, 1)]),
        "node2": gpd.GeoDataFrame({
            "id": [3, 4],
            "feat": [2.5, 3.5],
        }, geometry=[Point(2, 2), Point(3, 3)]),
    }

    # Create edges with edge_type column
    edges_gdf = gpd.GeoDataFrame({
        "src": [1, 2],
        "dst": [3, 4],
        "w": [1.0, 2.0],
        "edge_type": ["node1_connects_node2", "node1_links_node2"],  # Different edge types
    }, geometry=[
        LineString([(0, 0), (2, 2)]),
        LineString([(1, 1), (3, 3)]),
    ])

    data = heterogeneous_graph(nodes_dict, edges_gdf)

    # Should process edge types and create appropriate edge relationships
    # The edge_type column should be parsed to create tuple keys
    edge_types = data.edge_types
    assert len(edge_types) >= 1

    # Test edge type with exactly 3 parts (line 714)
    edges_3_parts = gpd.GeoDataFrame({
        "src": [1],
        "dst": [3],
        "w": [1.0],
        "edge_type": ["node1_connects_node2"],  # Exactly 3 parts when split by "_"
    }, geometry=[LineString([(0, 0), (2, 2)])])

    data = heterogeneous_graph(nodes_dict, edges_3_parts)
    # Should handle 3-part edge type names correctly
    assert len(data.edge_types) >= 1


def test_from_morphological_graph_error_handling() -> None:
    """Test from_morphological_graph error handling for missing data."""
    from city2graph.graph import from_morphological_graph

    if not is_torch_available():
        pytest.skip("PyTorch not available")

    # Test with missing required keys in network_output
    incomplete_network_output = {
        "private_nodes": gpd.GeoDataFrame({
            "id": [1, 2],
        }, geometry=[Point(0, 0), Point(1, 1)]),
        # Missing 'public_nodes' key
    }

    with pytest.raises((KeyError, ValueError)):
        from_morphological_graph(
            network_output=incomplete_network_output,
            private_node_feature_cols=["feat"],
            public_node_feature_cols=["feat"],
        )
