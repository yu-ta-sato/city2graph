"""Tests for the graph module."""

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest

# Import functions from the module to test without importing torch directly
from city2graph.utils import GraphMetadata

# Try to import torch, skip tests if not available
try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.data import HeteroData

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define dummy classes if torch_geometric is not available for type hinting
    class Data:
        pass

    class HeteroData:
        pass

# Import the graph module functions after torch is imported
if TORCH_AVAILABLE:
    from city2graph.graph import gdf_to_pyg
    from city2graph.graph import is_torch_available
    from city2graph.graph import nx_to_pyg
    from city2graph.graph import pyg_to_gdf
    from city2graph.graph import pyg_to_nx
    from city2graph.graph import validate_pyg


def test_is_torch_available() -> None:
    """Test the is_torch_available function."""
    assert is_torch_available() == TORCH_AVAILABLE

# ============================================================================
# gdf_to_pyg and pyg_to_gdf Tests
# ============================================================================


@pytest.mark.parametrize(
    ("node_feature_cols", "node_label_cols", "edge_feature_cols"),
    [
        # Test with None for all
        (None, None, None),
        # Test with existing columns
        (["feature1"], ["label1"], ["edge_feature1"]),
        # Test with some None, some existing
        (["feature1"], None, None),
        (None, ["label1"], None),
        (None, None, ["edge_feature1"]),
        # Test with empty lists
        ([], [], []),
        # Test with non-existent columns
        (["non_existent_node_feat"], ["non_existent_node_label"], ["non_existent_edge_feat"]),
    ],
)
def test_homogeneous_gdf_to_pyg_and_pyg_to_gdf_round_trip(
    sample_nodes_gdf, sample_edges_gdf, sample_crs,
    node_feature_cols, node_label_cols, edge_feature_cols,
) -> None:
    """Test homogeneous gdf_to_pyg -> pyg_to_gdf round trip with various feature/label column combinations."""
    pyg_data = gdf_to_pyg(
        sample_nodes_gdf,
        sample_edges_gdf,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_feature_cols=edge_feature_cols,
    )

    assert isinstance(pyg_data, Data)
    assert pyg_data.crs == sample_crs
    assert pyg_data.num_nodes == len(sample_nodes_gdf)
    assert pyg_data.num_edges == len(sample_edges_gdf)

    nodes_gdf, edges_gdf = pyg_to_gdf(pyg_data)

    assert isinstance(nodes_gdf, gpd.GeoDataFrame)
    assert nodes_gdf.crs == sample_crs
    pd.testing.assert_index_equal(nodes_gdf.index, sample_nodes_gdf.index)
    assert nodes_gdf.geom_equals_exact(sample_nodes_gdf.geometry, tolerance=1e-6).all()

    assert isinstance(edges_gdf, gpd.GeoDataFrame)
    assert edges_gdf.crs == sample_crs
    pd.testing.assert_index_equal(edges_gdf.index, sample_edges_gdf.index)
    assert edges_gdf.geom_equals_exact(sample_edges_gdf.geometry, tolerance=1e-6).all()

    # Verify features and labels are correctly handled (presence/absence and content)
    if node_feature_cols and any(col in sample_nodes_gdf.columns for col in node_feature_cols):
        expected_features = sample_nodes_gdf[list(set(node_feature_cols) & set(sample_nodes_gdf.columns))].values
        assert pyg_data.x.shape == expected_features.shape
        assert torch.allclose(pyg_data.x, torch.tensor(expected_features, dtype=pyg_data.x.dtype))
        # Check if reconstructed GDF has the features
        for col in list(set(node_feature_cols) & set(sample_nodes_gdf.columns)):
            pd.testing.assert_series_equal(nodes_gdf[col], sample_nodes_gdf[col], check_dtype=False)
    else:
        assert pyg_data.x.numel() == 0 # No features expected

    if node_label_cols and any(col in sample_nodes_gdf.columns for col in node_label_cols):
        expected_labels = sample_nodes_gdf[list(set(node_label_cols) & set(sample_nodes_gdf.columns))].values
        assert pyg_data.y.shape == expected_labels.shape
        assert torch.allclose(pyg_data.y, torch.tensor(expected_labels, dtype=pyg_data.y.dtype))
        # Check if reconstructed GDF has the labels
        for col in list(set(node_label_cols) & set(sample_nodes_gdf.columns)):
            pd.testing.assert_series_equal(nodes_gdf[col], sample_nodes_gdf[col], check_dtype=False)
    else:
        assert pyg_data.y is None or pyg_data.y.numel() == 0 # No labels expected

    if edge_feature_cols and any(col in sample_edges_gdf.columns for col in edge_feature_cols):
        expected_edge_features = sample_edges_gdf[list(set(edge_feature_cols) & set(sample_edges_gdf.columns))].select_dtypes(include=np.number).values
        assert pyg_data.edge_attr.shape == expected_edge_features.shape
        assert torch.allclose(pyg_data.edge_attr, torch.tensor(expected_edge_features, dtype=pyg_data.edge_attr.dtype))
        # Check if reconstructed GDF has the edge features
        for col in list(set(edge_feature_cols) & set(sample_edges_gdf.columns)):
            if col in edges_gdf.columns: # Only check if column exists in reconstructed GDF
                pd.testing.assert_series_equal(edges_gdf[col], sample_edges_gdf[col], check_dtype=False)
    else:
        assert pyg_data.edge_attr.numel() == 0 # No edge features expected


@pytest.mark.parametrize(
    ("nodes_gdf_fixture", "edges_gdf_fixture"),
    [
        ("empty_nodes_gdf", None),
        ("empty_nodes_gdf", "empty_edges_gdf"),
    ],
)
def test_gdf_to_pyg_empty_nodes_gdf(nodes_gdf_fixture, edges_gdf_fixture, request) -> None:
    """Test gdf_to_pyg with an empty nodes GeoDataFrame."""
    nodes_gdf = request.getfixturevalue(nodes_gdf_fixture)
    edges_gdf = request.getfixturevalue(edges_gdf_fixture) if edges_gdf_fixture else None

    pyg_data = gdf_to_pyg(nodes_gdf, edges_gdf)

    assert isinstance(pyg_data, Data)
    assert pyg_data.num_nodes == 0
    assert pyg_data.num_edges == 0
    assert pyg_data.x.numel() == 0
    assert pyg_data.edge_index.numel() == 0
    assert pyg_data.edge_attr.numel() == 0
    assert pyg_data.pos.numel() == 0

    nodes_reconstructed, edges_reconstructed = pyg_to_gdf(pyg_data)
    assert nodes_reconstructed.empty
    assert edges_reconstructed.empty


@pytest.mark.parametrize(
    ("node_feature_cols", "node_label_cols", "edge_feature_cols"),
    [
        # Test with None for all
        (None, None, None),
        # Test with existing columns
        ({"building": ["b_feat1"]}, {"building": ["b_label"]}, {"connects_to": ["conn_feat1"]}),
        # Test with some None, some existing
        ({"building": ["b_feat1"]}, None, None),
        (None, {"road": ["r_label"]}, None),
        (None, None, {"links_to": ["link_feat1"]}),
        # Test with empty dicts/lists
        ({"building": []}, {"road": []}, {"connects_to": []}),
        # Test with non-existent columns
        ({"building": ["non_existent_feat"]}, None, None),
    ],
)
def test_heterogeneous_gdf_to_pyg_and_pyg_to_gdf_round_trip(
    sample_hetero_nodes_dict, sample_hetero_edges_dict, sample_crs,
    node_feature_cols, node_label_cols, edge_feature_cols,
) -> None:
    """Test heterogeneous gdf_to_pyg -> pyg_to_gdf round trip with various feature/label column combinations."""
    pyg_data = gdf_to_pyg(
        sample_hetero_nodes_dict,
        sample_hetero_edges_dict,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_feature_cols=edge_feature_cols,
    )

    assert isinstance(pyg_data, HeteroData)
    assert pyg_data.crs == sample_crs

    nodes_dict, edges_dict = pyg_to_gdf(pyg_data)

    for ntype, gdf in sample_hetero_nodes_dict.items():
        assert ntype in nodes_dict
        re_gdf = nodes_dict[ntype]
        assert isinstance(re_gdf, gpd.GeoDataFrame)
        assert re_gdf.crs == sample_crs
        pd.testing.assert_index_equal(re_gdf.index, gdf.index)
        assert re_gdf.geom_equals_exact(gdf.geometry, tolerance=1e-6).all()

        # Verify features and labels for each node type
        if node_feature_cols and ntype in node_feature_cols and node_feature_cols[ntype]:
            expected_features = gdf[list(set(node_feature_cols[ntype]) & set(gdf.columns))].values
            assert pyg_data[ntype].x.shape == expected_features.shape
            assert torch.allclose(pyg_data[ntype].x, torch.tensor(expected_features, dtype=pyg_data[ntype].x.dtype))
            for col in list(set(node_feature_cols[ntype]) & set(gdf.columns)):
                pd.testing.assert_series_equal(re_gdf[col], gdf[col], check_dtype=False)
        else:
            assert pyg_data[ntype].x.numel() == 0

        if node_label_cols and ntype in node_label_cols and node_label_cols[ntype]:
            expected_labels = gdf[list(set(node_label_cols[ntype]) & set(gdf.columns))].values
            if hasattr(pyg_data[ntype], 'y') and pyg_data[ntype].y is not None:
                assert pyg_data[ntype].y.shape == expected_labels.shape
                assert torch.allclose(pyg_data[ntype].y, torch.tensor(expected_labels, dtype=pyg_data[ntype].y.dtype))
                for col in list(set(node_label_cols[ntype]) & set(gdf.columns)):
                    pd.testing.assert_series_equal(re_gdf[col], gdf[col], check_dtype=False)
            else:
                # If labels were expected but not found in pyg_data, it's a failure
                pytest.fail(f"Expected labels for {ntype} but pyg_data[{ntype}].y is missing or None")
        # If no labels were expected, ensure 'y' is None or empty
        elif hasattr(pyg_data[ntype], 'y'):
            assert pyg_data[ntype].y is None or pyg_data[ntype].y.numel() == 0

    for etype, gdf in sample_hetero_edges_dict.items():
        assert etype in edges_dict
        re_gdf = edges_dict[etype]
        assert isinstance(re_gdf, gpd.GeoDataFrame)
        assert re_gdf.crs == sample_crs
        pd.testing.assert_index_equal(re_gdf.index, gdf.index)
        assert re_gdf.geom_equals_exact(gdf.geometry, tolerance=1e-6).all()

        # Verify edge features for each edge type
        src_type, rel_type, dst_type = etype
        if edge_feature_cols and rel_type in edge_feature_cols and edge_feature_cols[rel_type]:
            expected_edge_features = gdf[list(set(edge_feature_cols[rel_type]) & set(gdf.columns))].select_dtypes(include=np.number).values
            assert pyg_data[etype].edge_attr.shape == expected_edge_features.shape
            assert torch.allclose(pyg_data[etype].edge_attr, torch.tensor(expected_edge_features, dtype=pyg_data[etype].edge_attr.dtype))
            for col in list(set(edge_feature_cols[rel_type]) & set(gdf.columns)):
                if col in re_gdf.columns:
                    pd.testing.assert_series_equal(re_gdf[col], gdf[col], check_dtype=False)
        else:
            assert pyg_data[etype].edge_attr.numel() == 0


@pytest.mark.parametrize(
    ("nodes_dict_fixture", "edges_dict_fixture"),
    [
        ("empty_hetero_nodes_dict", None),
        ("empty_hetero_nodes_dict", "empty_hetero_edges_dict"),
    ],
)
def test_gdf_to_pyg_empty_hetero_nodes_dict(nodes_dict_fixture, edges_dict_fixture, request) -> None:
    """Test gdf_to_pyg with an empty heterogeneous nodes dictionary."""
    nodes_dict = request.getfixturevalue(nodes_dict_fixture)
    edges_dict = request.getfixturevalue(edges_dict_fixture) if edges_dict_fixture else None

    pyg_data = gdf_to_pyg(nodes_dict, edges_dict)

    assert isinstance(pyg_data, HeteroData)
    for ntype in pyg_data.node_types:
        assert pyg_data[ntype].num_nodes == 0
    assert len(pyg_data.edge_types) == 0
    assert pyg_data.num_nodes == 0
    assert pyg_data.num_edges == 0

    nodes_reconstructed, edges_reconstructed = pyg_to_gdf(pyg_data)
    assert len(nodes_reconstructed) == 0
    assert len(edges_reconstructed) == 0


@pytest.mark.parametrize(
    ("device", "expected_error", "error_match"),
    [
        ("cpu", None, None),
        pytest.param("cuda", None, None, marks=pytest.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(), reason="CUDA not available")),
        pytest.param("cuda", ValueError, "CUDA selected, but not available", marks=pytest.mark.skipif(TORCH_AVAILABLE and torch.cuda.is_available(), reason="CUDA is available, testing non-available case") ),
        ("tpu", ValueError, "Device must be"),
        (123, TypeError, "Device must be"),
    ],
)
def test_gdf_to_pyg_device_handling(sample_nodes_gdf, device, expected_error, error_match) -> None:
    """Test device handling in gdf_to_pyg, including invalid devices."""
    if expected_error:
        with pytest.raises(expected_error, match=error_match):
            gdf_to_pyg(sample_nodes_gdf, device=device)
    else:
        pyg_data = gdf_to_pyg(sample_nodes_gdf, device=device)
        assert pyg_data.x.device.type == device
        if hasattr(pyg_data, "pos") and pyg_data.pos is not None:
            assert pyg_data.pos.device.type == device


def test_gdf_to_pyg_dtype(sample_nodes_gdf) -> None:
    """Test gdf_to_pyg with a specific dtype."""
    pyg_data = gdf_to_pyg(sample_nodes_gdf, dtype=torch.float64)
    assert pyg_data.x.dtype == torch.float64
    if hasattr(pyg_data, "pos") and pyg_data.pos is not None:
        assert pyg_data.pos.dtype == torch.float32  # Position is always float32


@pytest.mark.parametrize(
    ("nodes", "edges", "expected_edges"),
    [
        ("sample_nodes_gdf", None, 0),
        ("sample_nodes_gdf", "empty_edges_gdf", 0),
        ("sample_hetero_nodes_dict", None, {}),
        ("sample_hetero_nodes_dict", "edges_dict_with_empty", {("building", "connects", "road"): 0}),
    ],
)
def test_gdf_to_pyg_empty_and_no_edges(nodes, edges, expected_edges, request) -> None:
    """Test gdf_to_pyg with empty or no edges."""
    nodes_fix = request.getfixturevalue(nodes)
    edges_fix = request.getfixturevalue(edges) if edges else None

    pyg_data = gdf_to_pyg(nodes_fix, edges_fix)

    if isinstance(pyg_data, Data):
        assert pyg_data.num_edges == expected_edges
    else:
        for etype, num_edges in expected_edges.items():
            assert pyg_data[etype].num_edges == num_edges


def test_pyg_to_gdf_specific_types(sample_pyg_hetero_data) -> None:
    """Test pyg_to_gdf with specific node_types and edge_types."""
    nodes, edges = pyg_to_gdf(sample_pyg_hetero_data, node_types=["building"])
    assert "building" in nodes
    assert "road" not in nodes
    assert len(edges) == 2  # All edges are returned if not specified

    nodes, edges = pyg_to_gdf(sample_pyg_hetero_data, edge_types=[("building", "connects_to", "road")] )
    assert len(nodes) == 2  # All nodes are returned
    assert ("building", "connects_to", "road") in edges
    assert ("road", "links_to", "road") not in edges

@pytest.mark.parametrize(
    ("is_hetero", "node_cols", "label_cols", "edge_cols", "error", "match"),
    [
        (True, ["f1"], None, None, TypeError, "node_feature_cols must be a dict"),
        (True, None, ["l1"], None, TypeError, "node_label_cols must be a dict"),
        (True, None, None, ["e1"], TypeError, "edge_feature_cols must be a dict"),
        (False, {"n": ["f1"]}, None, None, TypeError, "node_feature_cols must be a list"),
        (False, None, {"n": ["l1"]}, None, TypeError, "node_label_cols must be a list"),
        (False, None, None, {"e": ["e1"]}, TypeError, "edge_feature_cols must be a list"),
    ]
)
def test_gdf_to_pyg_invalid_feature_cols(
    sample_nodes_gdf, sample_hetero_nodes_dict, is_hetero, node_cols, label_cols, edge_cols, error, match
) -> None:
    """Test that gdf_to_pyg raises errors for invalid feature column types."""
    nodes = sample_hetero_nodes_dict if is_hetero else sample_nodes_gdf
    with pytest.raises(error, match=match):
        gdf_to_pyg(
            nodes,
            node_feature_cols=node_cols,
            node_label_cols=label_cols,
            edge_feature_cols=edge_cols,
        )


# ============================================================================
# nx_to_pyg and pyg_to_nx Tests
# ============================================================================


@pytest.mark.parametrize(
    ("node_feature_cols", "node_label_cols", "edge_feature_cols"),
    [
        (None, None, None),
        (["feature1"], ["label1"], ["edge_feature1"]),
    ],
)
def test_nx_round_trip(
    sample_nx_graph, sample_crs,
    node_feature_cols, node_label_cols, edge_feature_cols,
) -> None:
    """Test nx_to_pyg -> pyg_to_nx round trip."""
    pyg_data = nx_to_pyg(
        sample_nx_graph,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_feature_cols=edge_feature_cols,
    )

    assert isinstance(pyg_data, Data)
    assert pyg_data.crs == sample_crs

    re_nx_graph = pyg_to_nx(pyg_data)

    assert isinstance(re_nx_graph, nx.Graph)
    assert re_nx_graph.graph.get("crs") == sample_crs
    assert re_nx_graph.number_of_nodes() == sample_nx_graph.number_of_nodes()
    assert re_nx_graph.number_of_edges() == sample_nx_graph.number_of_edges()


def test_nx_to_pyg_empty_graph() -> None:
    """Test nx_to_pyg with an empty NetworkX graph."""
    with pytest.raises(ValueError, match="Graph has no nodes"):
        nx_to_pyg(nx.Graph())


def test_pyg_to_nx_hetero_structure(sample_pyg_hetero_data, sample_hetero_nodes_dict, sample_hetero_edges_dict) -> None:
    """Test that pyg_to_nx preserves heterogeneous structure information."""
    nx_graph = pyg_to_nx(sample_pyg_hetero_data)

    assert nx_graph.graph.get("is_hetero") is True
    assert set(nx_graph.graph.get("node_types")) == set(sample_hetero_nodes_dict.keys())
    assert set(nx_graph.graph.get("edge_types")) == set(sample_hetero_edges_dict.keys())

    node_types_in_nx = {data.get("node_type") for _, data in nx_graph.nodes(data=True)}
    assert node_types_in_nx == set(sample_hetero_nodes_dict.keys())

    edge_types_in_nx = {data.get("edge_type") for _, _, data in nx_graph.edges(data=True)}
    expected_rel_types = {rel for _, rel, _ in sample_hetero_edges_dict}
    assert edge_types_in_nx == expected_rel_types


# ============================================================================
# validate_pyg Tests
# ============================================================================


def test_validate_pyg_success(sample_pyg_data) -> None:
    """Test that validate_pyg runs successfully on a valid object."""
    metadata = validate_pyg(sample_pyg_data)
    assert isinstance(metadata, GraphMetadata)


@pytest.mark.parametrize(
    ("obj_fixture", "error", "match"),
    [
        ("not_a_pyg_object", TypeError, "Input must be a PyTorch Geometric"),
    ],
)
def test_validate_pyg_errors(obj_fixture, error, match, request) -> None:
    """Test various validation errors in validate_pyg."""
    obj = request.getfixturevalue(obj_fixture)
    with pytest.raises(error, match=match):
        validate_pyg(obj)

def test_validate_pyg_missing_metadata() -> None:
    """Test validation error when graph_metadata is missing."""
    data = Data()
    with pytest.raises(ValueError, match="missing 'graph_metadata' attribute"):
        validate_pyg(data)

def test_validate_pyg_wrong_metadata_type() -> None:
    """Test validation error when graph_metadata is of the wrong type."""
    data = Data()
    data.graph_metadata = "wrong_type"
    with pytest.raises(ValueError, match="incorrect type"):
        validate_pyg(data)


@pytest.mark.parametrize(
    ("is_hetero_data", "is_hetero_meta", "match"),
    [
        (True, False, "object is HeteroData but metadata.is_hetero is False"),
        (False, True, "object is Data but metadata.is_hetero is True"),
    ],
)
def test_validate_pyg_inconsistencies(is_hetero_data, is_hetero_meta, match) -> None:
    """Test validation of inconsistencies between data type and metadata."""
    data = HeteroData() if is_hetero_data else Data()
    data.graph_metadata = GraphMetadata(is_hetero=is_hetero_meta)
    with pytest.raises(ValueError, match=match):
        validate_pyg(data)


def test_validate_pyg_structure_errors_homo(sample_pyg_data) -> None:
    """Test structural validation errors for homogeneous graphs."""
    # Has node_types
    sample_pyg_data.graph_metadata.node_types = ["some_type"]
    with pytest.raises(ValueError, match="should not have node_types"):
        validate_pyg(sample_pyg_data)
    sample_pyg_data.graph_metadata.node_types = []

    # Has edge_types
    sample_pyg_data.graph_metadata.edge_types = [("a", "b", "c")]
    with pytest.raises(ValueError, match="should not have edge_types"):
        validate_pyg(sample_pyg_data)
    sample_pyg_data.graph_metadata.edge_types = []

    # Missing 'default' key in node_mappings
    sample_pyg_data.graph_metadata.node_mappings = {"other": {}}
    with pytest.raises(ValueError, match="should use 'default' key"):
        validate_pyg(sample_pyg_data)
    sample_pyg_data.graph_metadata.node_mappings = {"default": {}}

    # Wrong type for feature/label cols
    sample_pyg_data.graph_metadata.node_feature_cols = {"a": "b"}
    with pytest.raises(ValueError, match="node_feature_cols as list"):
        validate_pyg(sample_pyg_data)
    sample_pyg_data.graph_metadata.node_feature_cols = []

    sample_pyg_data.graph_metadata.node_label_cols = {"a": "b"}
    with pytest.raises(ValueError, match="node_label_cols as list"):
        validate_pyg(sample_pyg_data)
    sample_pyg_data.graph_metadata.node_label_cols = []

    sample_pyg_data.graph_metadata.edge_feature_cols = {"a": "b"}
    with pytest.raises(ValueError, match="edge_feature_cols as list"):
        validate_pyg(sample_pyg_data)
    sample_pyg_data.graph_metadata.edge_feature_cols = []

    # Tensor shape mismatch
    sample_pyg_data.pos = torch.randn(1, 2)  # Wrong size
    with pytest.raises(ValueError, match="position tensor size"):
        validate_pyg(sample_pyg_data)
    sample_pyg_data.pos = torch.randn(sample_pyg_data.num_nodes, 2)

    sample_pyg_data.y = torch.randn(1, 1)
    with pytest.raises(ValueError, match="label tensor size"):
        validate_pyg(sample_pyg_data)
    sample_pyg_data.y = None

    sample_pyg_data.edge_attr = torch.randn(1, 1)
    with pytest.raises(ValueError, match="Edge attribute tensor size"):
        validate_pyg(sample_pyg_data)
    sample_pyg_data.edge_attr = torch.randn(sample_pyg_data.num_edges, 1)


def test_validate_pyg_structure_errors_hetero(sample_pyg_hetero_data) -> None:
    """Test structural validation errors for heterogeneous graphs."""
    # Node type mismatch
    original_node_types = sample_pyg_hetero_data.graph_metadata.node_types
    sample_pyg_hetero_data.graph_metadata.node_types = ["building"]  # Missing 'road'
    with pytest.raises(ValueError, match="Node types mismatch"):
        validate_pyg(sample_pyg_hetero_data)
    sample_pyg_hetero_data.graph_metadata.node_types = original_node_types

    # Edge type mismatch
    original_edge_types = sample_pyg_hetero_data.graph_metadata.edge_types
    sample_pyg_hetero_data.graph_metadata.edge_types = [original_edge_types[0]]
    with pytest.raises(ValueError, match="Edge types mismatch"):
        validate_pyg(sample_pyg_hetero_data)
    sample_pyg_hetero_data.graph_metadata.edge_types = original_edge_types

    # Tensor shape mismatch
    original_pos = sample_pyg_hetero_data["building"].pos
    sample_pyg_hetero_data["building"].pos = torch.randn(1, 2)  # Wrong size
    with pytest.raises(ValueError, match="position tensor size"):
        validate_pyg(sample_pyg_hetero_data)
    sample_pyg_hetero_data["building"].pos = original_pos

    sample_pyg_hetero_data["building"].y = torch.randn(sample_pyg_hetero_data["building"].num_nodes, 1)
    original_y = sample_pyg_hetero_data["building"].y
    sample_pyg_hetero_data["building"].y = torch.randn(1, 1)
    with pytest.raises(ValueError, match="label tensor size"):
        validate_pyg(sample_pyg_hetero_data)
    sample_pyg_hetero_data["building"].y = original_y

