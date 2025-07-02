"""Tests for the graph module."""

from unittest.mock import patch  # Added import

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import Point  # Added import

from city2graph.graph import _get_device
from city2graph.graph import gdf_to_pyg
from city2graph.graph import is_torch_available
from city2graph.graph import nx_to_pyg
from city2graph.graph import pyg_to_gdf
from city2graph.graph import pyg_to_nx

# Try to import torch, skip tests if not available
try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.data import HeteroData
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Define dummy classes if torch_geometric is not available for type hinting
    class Data:  # noqa: D101
        pass
    class HeteroData:  # noqa: D101
        pass

# Pytest skipif marker for tests requiring torch
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch or PyTorch Geometric is not available.")

@requires_torch
@pytest.mark.parametrize(("node_feature_cols", "node_label_cols", "edge_feature_cols"), [
    (None, None, None),
    (["feature1"], ["label1"], ["edge_feature1"]),
    (["feature1"], None, None),
    (None, ["label1"], None),
    (None, None, ["edge_feature1"]),
    (["non_existent_node_feat"], ["non_existent_node_label"], ["non_existent_edge_feat"]),
])
def test_gdf_to_pyg_homogeneous_round_trip(sample_nodes_gdf, sample_edges_gdf, sample_crs, node_feature_cols, node_label_cols, edge_feature_cols) -> None:
    """Test gdf_to_pyg and pyg_to_gdf for homogeneous graphs with various feature/label configurations."""
    # Convert GDF to PyG
    pyg_data = gdf_to_pyg(
        sample_nodes_gdf,
        sample_edges_gdf,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_feature_cols=edge_feature_cols,
        device="cpu", # Test with CPU
    )

    assert isinstance(pyg_data, Data)
    assert pyg_data.crs == sample_crs

    # Check node features
    if node_feature_cols and "feature1" in node_feature_cols:
        assert hasattr(pyg_data, "x")
        assert pyg_data.x.shape[0] == len(sample_nodes_gdf)
        assert pyg_data.x.shape[1] == 1
    elif node_feature_cols and "non_existent_node_feat" in node_feature_cols:
        assert hasattr(pyg_data, "x")
        assert pyg_data.x.shape[1] == 0 # No features should be created
    else:
        assert hasattr(pyg_data, "x") # x is always created
        assert pyg_data.x.shape[1] == 0 # Empty features

    # Check node labels
    if node_label_cols and "label1" in node_label_cols:
        assert hasattr(pyg_data, "y")
        assert pyg_data.y.shape[0] == len(sample_nodes_gdf)
        assert pyg_data.y.shape[1] == 1
    elif node_label_cols and "non_existent_node_label" in node_label_cols:
        assert hasattr(pyg_data, "y")
        assert pyg_data.y.shape[1] == 0
    else:
        assert not hasattr(pyg_data, "y") or pyg_data.y is None

    # Check edge features
    if edge_feature_cols and "edge_feature1" in edge_feature_cols:
        assert hasattr(pyg_data, "edge_attr")
        assert pyg_data.edge_attr.shape[0] == len(sample_edges_gdf)
        assert pyg_data.edge_attr.shape[1] == 1
    elif edge_feature_cols and "non_existent_edge_feat" in edge_feature_cols:
        assert hasattr(pyg_data, "edge_attr")
        assert pyg_data.edge_attr.shape[1] == 0
    else:
        assert hasattr(pyg_data, "edge_attr")
        assert pyg_data.edge_attr.shape[1] == 0

    # Convert PyG back to GDF
    reconstructed_nodes_gdf, reconstructed_edges_gdf = pyg_to_gdf(pyg_data)

    assert isinstance(reconstructed_nodes_gdf, gpd.GeoDataFrame)

    # Compare only the columns that were actually reconstructed
    reconstructed_cols = reconstructed_nodes_gdf.columns.drop("geometry", errors="ignore")
    # Ensure index is sorted for consistent comparison
    reconstructed_nodes_gdf = reconstructed_nodes_gdf.sort_index()
    original_nodes_to_compare = sample_nodes_gdf.sort_index()

    if reconstructed_cols.empty:
        # If no attribute columns were reconstructed, compare against an empty selection from original
        pd.testing.assert_frame_equal(
            reconstructed_nodes_gdf[reconstructed_cols],
            original_nodes_to_compare[reconstructed_cols], # This will be an empty DataFrame
            check_dtype=False, atol=1e-5,
        )
    else:
        pd.testing.assert_frame_equal(
            reconstructed_nodes_gdf[reconstructed_cols],
            original_nodes_to_compare[reconstructed_cols],
            check_dtype=False, atol=1e-5,
        )

    assert reconstructed_nodes_gdf.crs == sample_crs
    assert reconstructed_nodes_gdf.geom_equals_exact(sample_nodes_gdf.geometry, tolerance=1e-5).all()

    if sample_edges_gdf is not None and not sample_edges_gdf.empty:
        assert isinstance(reconstructed_edges_gdf, gpd.GeoDataFrame)
        # When edge features are not selected, edge_attr is empty, leading to no feature columns in reconstructed_edges_gdf
        # So, we only compare if original edge_feature_cols were specified and valid

        reconstructed_edge_cols = reconstructed_edges_gdf.columns.drop("geometry", errors="ignore")
        # Ensure index is sorted for consistent comparison
        reconstructed_edges_gdf = reconstructed_edges_gdf.sort_index()
        original_edges_to_compare = sample_edges_gdf.sort_index()

        if edge_feature_cols and "edge_feature1" in edge_feature_cols and "edge_feature1" in reconstructed_edge_cols:
             pd.testing.assert_frame_equal(
                 reconstructed_edges_gdf[reconstructed_edge_cols],
                 original_edges_to_compare[reconstructed_edge_cols],
                 check_dtype=False, atol=1e-5,
            )
        elif reconstructed_edge_cols.empty:
             pd.testing.assert_frame_equal(
                 reconstructed_edges_gdf[reconstructed_edge_cols],
                 original_edges_to_compare[reconstructed_edge_cols], # Empty comparison
                 check_dtype=False, atol=1e-5,
            )
        # If there are other columns in reconstructed_edges_gdf (e.g. if edge_feature_cols was None but some default were picked)
        # This part of logic might need refinement based on how pyg_to_gdf handles unspecified edge_feature_cols
        # For now, if specific features were requested and exist, we check them. If reconstructed is empty, we check that.

        assert reconstructed_edges_gdf.crs == sample_crs
        assert reconstructed_edges_gdf.geom_equals_exact(sample_edges_gdf.geometry, tolerance=1e-5).all()
        # Check if index is preserved
        pd.testing.assert_index_equal(reconstructed_edges_gdf.index, sample_edges_gdf.index)
    else:
        assert reconstructed_edges_gdf is None

@requires_torch
@pytest.mark.parametrize(("node_feature_cols", "node_label_cols", "edge_feature_cols"), [
    (None, None, None),
    (None, None, {"connects_to": ["conn_feat1"]}),
    (None, {"road": ["r_label"]}, None),
    ({"building": ["b_feat1"]}, None, None),
    ({"building": ["non_existent_feat"]}, None, {"connects_to": ["non_existent_edge_feat"]}),
    ({"building": ["b_feat1"], "road": ["r_feat1"]}, {"building": ["b_label"], "road": ["r_label"]}, {"connects_to": ["conn_feat1"], "links_to": ["link_feat1"]}),
])
def test_gdf_to_pyg_heterogeneous_round_trip(sample_hetero_nodes_dict, sample_hetero_edges_dict, sample_crs, node_feature_cols, node_label_cols, edge_feature_cols) -> None:
    """Test gdf_to_pyg and pyg_to_gdf for heterogeneous graphs with various feature/label configurations."""
    pyg_data = gdf_to_pyg(
        sample_hetero_nodes_dict,
        sample_hetero_edges_dict,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_feature_cols=edge_feature_cols,
        device="cpu",
    )

    assert isinstance(pyg_data, HeteroData)
    assert pyg_data.crs == sample_crs

    # Check node features and labels for each type
    for node_type, original_gdf in sample_hetero_nodes_dict.items():
        # Features
        if node_feature_cols and node_type in node_feature_cols and pyg_data[node_type].x.numel() > 0:
            assert pyg_data[node_type].x.shape[0] == len(original_gdf)
            if original_gdf.columns.intersection(node_feature_cols[node_type]).any():
                assert pyg_data[node_type].x.shape[1] == len(node_feature_cols[node_type])
            else:
                assert pyg_data[node_type].x.shape[1] == 0  # No valid features
        else:
            assert pyg_data[node_type].x.shape[1] == 0
        # Labels
        if node_label_cols and node_type in node_label_cols and pyg_data[node_type].y.numel() > 0:
            assert pyg_data[node_type].y.shape[0] == len(original_gdf)
            if original_gdf.columns.intersection(node_label_cols[node_type]).any():
                assert pyg_data[node_type].y.shape[1] == len(node_label_cols[node_type])
            else:
                assert pyg_data[node_type].y.shape[1] == 0  # No valid labels
        elif not (hasattr(pyg_data[node_type], "y") and pyg_data[node_type].y is not None and pyg_data[node_type].y.numel() > 0):
            assert True  # y can be None or empty tensor

    # Check edge features for each type
    for edge_type_tuple, original_edge_gdf in sample_hetero_edges_dict.items():
        relation_type = edge_type_tuple[1]
        if edge_feature_cols and relation_type in edge_feature_cols and pyg_data[edge_type_tuple].edge_attr.numel() > 0:
            assert pyg_data[edge_type_tuple].edge_attr.shape[0] == len(original_edge_gdf)
            if original_edge_gdf.columns.intersection(edge_feature_cols[relation_type]).any():
                assert pyg_data[edge_type_tuple].edge_attr.shape[1] == len(edge_feature_cols[relation_type])
            else:
                assert pyg_data[edge_type_tuple].edge_attr.shape[1] == 0  # No valid features
        else:
            assert pyg_data[edge_type_tuple].edge_attr.shape[1] == 0

    reconstructed_nodes_dict, reconstructed_edges_dict = pyg_to_gdf(pyg_data)

    for node_type, original_gdf in sample_hetero_nodes_dict.items():
        assert node_type in reconstructed_nodes_dict
        re_gdf = reconstructed_nodes_dict[node_type].sort_index()
        original_gdf_sorted = original_gdf.sort_index()
        assert isinstance(re_gdf, gpd.GeoDataFrame)

        reconstructed_cols = re_gdf.columns.drop("geometry", errors="ignore")
        if reconstructed_cols.empty:
            pd.testing.assert_frame_equal(
                re_gdf[reconstructed_cols],
                original_gdf_sorted[reconstructed_cols],
                check_dtype=False, atol=1e-5,
            )
        else:
            pd.testing.assert_frame_equal(
                re_gdf[reconstructed_cols],
                original_gdf_sorted[reconstructed_cols],
                check_dtype=False, atol=1e-5,
            )
        assert re_gdf.crs == sample_crs
        assert re_gdf.geom_equals_exact(original_gdf_sorted.geometry, tolerance=1e-5).all()

    for edge_type, original_gdf in sample_hetero_edges_dict.items():
        relation_type = edge_type[1] # This is the key for edge_feature_cols dict
        assert edge_type in reconstructed_edges_dict
        re_gdf = reconstructed_edges_dict[edge_type].sort_index()
        original_gdf_sorted = original_gdf.sort_index()

        assert isinstance(re_gdf, gpd.GeoDataFrame)

        reconstructed_edge_cols = re_gdf.columns.drop("geometry", errors="ignore")

        # Determine expected columns based on what was requested and if it's valid
        expected_edge_cols = []
        if edge_feature_cols and relation_type in edge_feature_cols:
            # Only consider columns that were actually requested for this relation_type
            # and are present in the original GDF.
            requested_and_valid = [col for col in edge_feature_cols[relation_type] if col in original_gdf_sorted.columns]
            if requested_and_valid:  # If any requested columns are valid
                # And these columns are also in the reconstructed GDF
                expected_edge_cols = [col for col in requested_and_valid if col in reconstructed_edge_cols]


        if expected_edge_cols: # If we have specific columns to compare
            pd.testing.assert_frame_equal(
                re_gdf[expected_edge_cols],
                original_gdf_sorted[expected_edge_cols],
                check_dtype=False, atol=1e-5,
            )
        elif reconstructed_edge_cols.empty:  # If no features were reconstructed
            pd.testing.assert_frame_equal(
                re_gdf[reconstructed_edge_cols],
                original_gdf_sorted[reconstructed_edge_cols],  # Empty comparison
                check_dtype=False, atol=1e-5,
            )
        # Add a general check for all reconstructed columns if no specific ones were expected
        # This handles cases where edge_feature_cols might be None but some features are still reconstructed
        elif not reconstructed_edge_cols.empty:
            pd.testing.assert_frame_equal(
                re_gdf[reconstructed_edge_cols],
                original_gdf_sorted[reconstructed_edge_cols],
                check_dtype=False, atol=1e-5,
            )


        assert re_gdf.crs == sample_crs
        assert re_gdf.geom_equals_exact(original_gdf_sorted.geometry, tolerance=1e-5).all()
        pd.testing.assert_index_equal(re_gdf.index, original_gdf_sorted.index)


def test_is_torch_available() -> None:
    """Test the is_torch_available function."""
    # This test's success depends on whether torch was successfully imported at the top
    assert is_torch_available() == TORCH_AVAILABLE

@requires_torch
def test_gdf_to_pyg_device_handling(sample_nodes_gdf, sample_edges_gdf) -> None:
    """Test device handling in gdf_to_pyg."""
    # Test with CPU
    pyg_data_cpu = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, device="cpu")
    assert pyg_data_cpu.x.device.type == "cpu"
    if hasattr(pyg_data_cpu, "pos") and pyg_data_cpu.pos is not None:
        assert pyg_data_cpu.pos.device.type == "cpu"
    assert pyg_data_cpu.edge_index.device.type == "cpu"

    # Test with CUDA if available
    if torch.cuda.is_available():
        try:
            pyg_data_cuda = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, device="cuda")
            assert pyg_data_cuda.x.device.type == "cuda"
            if hasattr(pyg_data_cuda, "pos") and pyg_data_cuda.pos is not None:
                assert pyg_data_cuda.pos.device.type == "cuda"
            assert pyg_data_cuda.edge_index.device.type == "cuda"
        except (RuntimeError, AssertionError) as e: # Catch runtime error or assertion error if CUDA compiled but not usable
            if "Torch not compiled with CUDA enabled" in str(e) or "CUDA driver version is insufficient" in str(e) or "CUDA out of memory" in str(e):
                pytest.skip(f"CUDA environment issue: {e}")
            else:
                raise # Re-raise if it's an unexpected RuntimeError
    else:
        with pytest.raises(ValueError, match="CUDA selected, but not available. Device must be 'cuda', 'cpu', a torch.device object, or None"):
            gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, device="cuda")  # Should raise error if cuda not available but specified

@requires_torch
def test_empty_edges_gdf_to_pyg(sample_nodes_gdf) -> None:
    """Test gdf_to_pyg with an empty edges GeoDataFrame."""
    empty_edges_gdf = gpd.GeoDataFrame(columns=["source_id", "target_id", "geometry"], crs=sample_nodes_gdf.crs)
    empty_edges_gdf = empty_edges_gdf.set_index(["source_id", "target_id"])

    with pytest.raises(ValueError, match="GeoDataFrame cannot be empty"):
        gdf_to_pyg(sample_nodes_gdf, empty_edges_gdf)


@requires_torch
def test_no_edges_gdf_to_pyg(sample_nodes_gdf) -> None:
    """Test gdf_to_pyg with no edges GeoDataFrame provided (edges=None)."""
    pyg_data = gdf_to_pyg(sample_nodes_gdf, edges=None)
    assert isinstance(pyg_data, Data)
    assert pyg_data.edge_index.shape == (2,0)
    assert pyg_data.edge_attr.shape[0] == 0

    _re_nodes, re_edges = pyg_to_gdf(pyg_data) # _re_nodes to avoid unused var
    assert re_edges is None or re_edges.empty # Expect None or an empty GeoDataFrame

@requires_torch
def test_hetero_no_edges_dict(sample_hetero_nodes_dict) -> None:
    """Test gdf_to_pyg for heterogeneous graph with no edges_dict."""
    pyg_data = gdf_to_pyg(sample_hetero_nodes_dict, edges=None)
    assert isinstance(pyg_data, HeteroData)
    for edge_type in pyg_data.edge_types:
        assert pyg_data[edge_type].edge_index.shape == (2,0)
        assert pyg_data[edge_type].edge_attr.shape[0] == 0

    re_nodes_dict, re_edges_dict = pyg_to_gdf(pyg_data)
    assert not re_edges_dict # Should be an empty dictionary

@requires_torch
def test_hetero_empty_edges_in_dict(sample_hetero_nodes_dict, sample_crs) -> None:
    """Test gdf_to_pyg for heterogeneous graph with an edge type having an empty GeoDataFrame."""
    empty_conn_gdf = gpd.GeoDataFrame(columns=["b_id", "r_id", "geometry"], crs=sample_crs)
    empty_conn_gdf = empty_conn_gdf.set_index(["b_id", "r_id"])
    edges_dict_with_empty = {("building", "connects", "road"): empty_conn_gdf}

    # No longer raises ValueError, should create empty tensors for the edge type
    pyg_data = gdf_to_pyg(sample_hetero_nodes_dict, edges_dict_with_empty)

    assert isinstance(pyg_data, HeteroData)
    edge_type = ("building", "connects", "road")
    assert edge_type in pyg_data.edge_types
    assert pyg_data[edge_type].edge_index.shape == (2, 0)
    assert pyg_data[edge_type].edge_attr.shape == (0, 0)


@requires_torch
def test_pyg_to_gdf_specific_types(sample_hetero_nodes_dict, sample_hetero_edges_dict) -> None:
    """Test pyg_to_gdf with specific node_types and edge_types."""
    pyg_data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict)

    # Test reconstructing only 'building' nodes
    nodes_dict_specific, _edges_dict_specific = pyg_to_gdf(pyg_data, node_types=["building"])
    assert "building" in nodes_dict_specific
    assert "road" not in nodes_dict_specific
    assert len(nodes_dict_specific) == 1

    # Test reconstructing only ('building', 'connects_to', 'road') edges
    _nodes_dict_specific, edges_dict_specific = pyg_to_gdf(pyg_data,
                                                           edge_types=[("building","connects_to", "road")])
    assert ("building", "connects_to", "road") in edges_dict_specific
    assert ("road", "links_to", "road") not in edges_dict_specific
    assert len(edges_dict_specific) == 1

    # Test reconstructing specific node and edge types together
    nodes_dict_combo, edges_dict_combo = pyg_to_gdf(pyg_data,
                                                    node_types=["road"],
                                                    edge_types=[("road", "links_to", "road")])
    assert "road" in nodes_dict_combo
    assert "building" not in nodes_dict_combo
    assert ("road", "links_to", "road") in edges_dict_combo
    assert ("building", "connects_to", "road") not in edges_dict_combo

@requires_torch
def test_nx_to_pyg_empty_graph() -> None:
    """Test nx_to_pyg with an empty NetworkX graph."""
    empty_nx = nx.Graph()
    with pytest.raises(ValueError, match="Graph has no nodes"):
        nx_to_pyg(empty_nx)

@requires_torch
def test_pyg_to_nx_hetero_structure(sample_hetero_nodes_dict, sample_hetero_edges_dict) -> None:
    """Test that pyg_to_nx preserves heterogeneous structure information."""
    pyg_data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict)
    nx_graph = pyg_to_nx(pyg_data)

    assert nx_graph.graph.get("is_hetero") is True
    assert set(nx_graph.graph.get("node_types")) == set(sample_hetero_nodes_dict.keys())
    assert set(nx_graph.graph.get("edge_types")) == set(sample_hetero_edges_dict.keys())

    # Check node type attributes
    node_types_in_nx = {data.get("node_type") for _, data in nx_graph.nodes(data=True)}
    assert node_types_in_nx == set(sample_hetero_nodes_dict.keys())

    # Check edge type attributes
    edge_types_in_nx = {data.get("edge_type") for _, _, data in nx_graph.edges(data=True)}
    expected_rel_types = {rel for _, rel, _ in sample_hetero_edges_dict}
    assert edge_types_in_nx == expected_rel_types

# Tests for when TORCH_AVAILABLE is False (simulated via mocking)

def test_gdf_to_pyg_torch_unavailable(sample_nodes_gdf) -> None:
    """Test gdf_to_pyg raises ImportError if torch is unavailable."""
    with patch("city2graph.graph.TORCH_AVAILABLE", False):
        with pytest.raises(ImportError, match="PyTorch required."):
            gdf_to_pyg(sample_nodes_gdf)

def test_pyg_to_gdf_torch_unavailable() -> None:
    """Test pyg_to_gdf raises ImportError if torch is unavailable."""
    # Create a dummy Data object for testing, as it's needed by the function signature
    # This object won't be deeply inspected if TORCH_AVAILABLE is false.
    dummy_pyg_data = Data() if TORCH_AVAILABLE else object() # Use object() if Data itself is not defined
    # The _validate_pyg function, called by pyg_to_gdf, checks TORCH_AVAILABLE first.
    with patch("city2graph.graph.TORCH_AVAILABLE", False):
        with pytest.raises(ImportError, match="PyTorch required."):
            pyg_to_gdf(dummy_pyg_data)

def test_nx_to_pyg_torch_unavailable(sample_nx_graph) -> None:
    """Test nx_to_pyg raises ImportError if torch is unavailable."""
    with patch("city2graph.graph.TORCH_AVAILABLE", False):
        with pytest.raises(ImportError, match="PyTorch required."):
            nx_to_pyg(sample_nx_graph)

def test_pyg_to_nx_torch_unavailable() -> None:
    """Test pyg_to_nx raises ImportError if torch is unavailable."""
    dummy_pyg_data = Data() if TORCH_AVAILABLE else object()
    with patch("city2graph.graph.TORCH_AVAILABLE", False):
        with pytest.raises(ImportError, match="PyTorch required."):
            pyg_to_nx(dummy_pyg_data)

@requires_torch
@pytest.mark.parametrize(("node_feature_cols", "node_label_cols", "edge_feature_cols", "generic_test_type"), [
    (None, None, None, None),
    (["feature1"], ["label1"], ["edge_feature1"], None),
    (["feature1"], None, None, "node_x_generic"), # New case for generic node feature names
    (None, ["label1"], None, "node_y_generic"),   # New case for generic node label names
    (None, None, ["edge_feature1"], "edge_attr_generic"), # New case for generic edge feature names
])
def test_nx_to_pyg_round_trip(sample_nx_graph, sample_crs, node_feature_cols, node_label_cols, edge_feature_cols, generic_test_type) -> None:
    """Test nx_to_pyg and pyg_to_nx for round trip conversion."""
    # Convert NX to PyG
    pyg_data = nx_to_pyg(
        sample_nx_graph,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_feature_cols=edge_feature_cols,
        device="cpu",
    )

    assert isinstance(pyg_data, Data)
    assert pyg_data.crs == sample_crs

    # Modify pyg_data metadata if testing for generic attribute names
    if generic_test_type == "node_x_generic":
        if hasattr(pyg_data, "x") and pyg_data.x.shape[1] > 0:
            pyg_data.graph_metadata.node_feature_cols = []
    elif generic_test_type == "node_y_generic":
        if hasattr(pyg_data, "y") and pyg_data.y is not None and pyg_data.y.shape[1] > 0:
            pyg_data.graph_metadata.node_label_cols = []
    elif generic_test_type == "edge_attr_generic":
        if hasattr(pyg_data, "edge_attr") and pyg_data.edge_attr.shape[1] > 0:
            pyg_data.graph_metadata.edge_feature_cols = []


    # Check node features tensor based on node_feature_cols
    if node_feature_cols:
        assert hasattr(pyg_data, "x")
        assert pyg_data.x.shape[0] == sample_nx_graph.number_of_nodes()
        # Count actual features extracted, not just len(node_feature_cols)
        # as some input features might not exist in the graph.
        # For sample_nx_graph, "feature1" exists.
        if "feature1" in node_feature_cols:
            assert pyg_data.x.shape[1] >= 1  # Should be 1 if only "feature1"
        else:
            assert pyg_data.x.shape[1] == 0  # If node_feature_cols don't exist
    else:  # If node_feature_cols is None
        assert hasattr(pyg_data, "x")
        assert pyg_data.x.shape[1] == 0

    # Check node labels tensor
    if node_label_cols:
        assert hasattr(pyg_data, "y")
        assert pyg_data.y.shape[0] == sample_nx_graph.number_of_nodes()
        if "label1" in node_label_cols:
            assert pyg_data.y.shape[1] >= 1
        else:
            assert pyg_data.y.shape[1] == 0
    elif generic_test_type != "node_y_generic":  # if not specifically testing generic y, and node_label_cols is None
        assert not hasattr(pyg_data, "y") or pyg_data.y is None

    # Check edge features tensor
    if edge_feature_cols:
        assert hasattr(pyg_data, "edge_attr")
        assert pyg_data.edge_attr.shape[0] == sample_nx_graph.number_of_edges()
        if "edge_feature1" in edge_feature_cols:
            assert pyg_data.edge_attr.shape[1] >= 1
        else:
            assert pyg_data.edge_attr.shape[1] == 0
    else:  # If edge_feature_cols is None
        assert hasattr(pyg_data, "edge_attr")
        assert pyg_data.edge_attr.shape[1] == 0

    # Convert PyG back to NX
    reconstructed_nx_graph = pyg_to_nx(pyg_data)

    assert isinstance(reconstructed_nx_graph, nx.Graph)
    assert reconstructed_nx_graph.graph.get("crs") == sample_crs
    assert reconstructed_nx_graph.number_of_nodes() == sample_nx_graph.number_of_nodes()
    assert reconstructed_nx_graph.number_of_edges() == sample_nx_graph.number_of_edges()

    original_ids_map = {}
    if hasattr(pyg_data, "graph_metadata") and pyg_data.graph_metadata.node_mappings:
        id_mapping_inv = {v: k for k, v in pyg_data.graph_metadata.node_mappings["default"]["mapping"].items()}
        for i in range(reconstructed_nx_graph.number_of_nodes()):
            original_ids_map[i] = id_mapping_inv.get(i)
    else:
        # Fallback for sample_nx_graph if metadata somehow missing (should not happen)
        for i in range(reconstructed_nx_graph.number_of_nodes()):
            original_ids_map[i] = i + 1


    for reconstructed_node_id in reconstructed_nx_graph.nodes():
        original_node_id = original_ids_map.get(reconstructed_node_id)
        assert original_node_id is not None, f"Could not map reconstructed node ID {reconstructed_node_id}"
        assert original_node_id in sample_nx_graph.nodes, f"Original node ID {original_node_id} not in sample_nx_graph"

        original_attrs = sample_nx_graph.nodes[original_node_id]
        reconstructed_attrs = reconstructed_nx_graph.nodes[reconstructed_node_id]

        # Node features assertions
        if generic_test_type == "node_x_generic":
            # Expect "feat_0" if node_feature_cols was ["feature1"]
            if "feature1" in original_attrs and pyg_data.x.shape[1] > 0:  # Ensure original feature existed and was extracted
                assert "feat_0" in reconstructed_attrs, f"Generic attribute feat_0 missing in node {reconstructed_node_id}"
                assert abs(original_attrs["feature1"] - reconstructed_attrs["feat_0"]) < 1e-5, f"Value mismatch for feat_0 in node {reconstructed_node_id}"
        elif node_feature_cols:
            for feat in node_feature_cols:
                if feat in original_attrs:  # Only assert if the feature was in the original graph
                    assert feat in reconstructed_attrs, f"Feature {feat} missing in reconstructed node {reconstructed_node_id}"
                    assert abs(original_attrs.get(feat, 0) - reconstructed_attrs.get(feat, 0)) < 1e-5

        # Node labels assertions
        if generic_test_type == "node_y_generic":
            # Expect "label_0" if node_label_cols was ["label1"]
            if "label1" in original_attrs and pyg_data.y is not None and pyg_data.y.shape[1] > 0:
                assert "label_0" in reconstructed_attrs, f"Generic attribute label_0 missing in node {reconstructed_node_id}"
                assert abs(original_attrs["label1"] - reconstructed_attrs["label_0"]) < 1e-5, f"Value mismatch for label_0 in node {reconstructed_node_id}"
        elif node_label_cols:
            for label in node_label_cols:
                if label in original_attrs:
                    assert label in reconstructed_attrs, f"Label {label} missing in reconstructed node {reconstructed_node_id}"
                    assert abs(original_attrs.get(label, 0) - reconstructed_attrs.get(label, 0)) < 1e-5

        if "geometry" in original_attrs:
            # For PyG round trip, we expect 'pos' but geometry may or may not be preserved
            # depending on the conversion implementation. Check for position instead.
            assert "pos" in reconstructed_attrs
            original_point = original_attrs["geometry"]
            reconstructed_point_coords = reconstructed_attrs["pos"]
            assert isinstance(original_point, Point)
            assert isinstance(reconstructed_point_coords, tuple)
            assert len(reconstructed_point_coords) == 2
            assert abs(original_point.x - reconstructed_point_coords[0]) < 1e-5
            assert abs(original_point.y - reconstructed_point_coords[1]) < 1e-5

    # Compare edge attributes
    for u_re, v_re, reconstructed_edge_attrs in reconstructed_nx_graph.edges(data=True):
        u_orig = original_ids_map.get(u_re)
        v_orig = original_ids_map.get(v_re)
        assert u_orig is not None, f"Could not map edge node {u_re} to original ID"
        assert v_orig is not None, f"Could not map edge node {v_re} to original ID"

        original_edge_attrs = None
        if sample_nx_graph.has_edge(u_orig, v_orig):
            original_edge_attrs = sample_nx_graph.edges[u_orig, v_orig]
        elif sample_nx_graph.has_edge(v_orig, u_orig):
            original_edge_attrs = sample_nx_graph.edges[v_orig, u_orig]
        assert original_edge_attrs is not None, f"Edge ({u_orig}, {v_orig}) not found in original graph"

        if generic_test_type == "edge_attr_generic":
            # Expect "edge_feat_0" if edge_feature_cols was ["edge_feature1"]
            if "edge_feature1" in original_edge_attrs and pyg_data.edge_attr.shape[1] > 0:
                assert "edge_feat_0" in reconstructed_edge_attrs
                assert abs(original_edge_attrs["edge_feature1"] - reconstructed_edge_attrs["edge_feat_0"]) < 1e-5
        elif edge_feature_cols:
            for feat in edge_feature_cols:
                if feat in original_edge_attrs:
                    assert feat in reconstructed_edge_attrs
                    assert abs(original_edge_attrs.get(feat, 0) - reconstructed_edge_attrs.get(feat, 0)) < 1e-5

@requires_torch
def test_get_device() -> None:
    """Test the _get_device utility function for device handling."""
    # 1. Test with device=None
    with patch("torch.cuda.is_available", return_value=True):
        assert _get_device(None) == torch.device("cuda")
    with patch("torch.cuda.is_available", return_value=False):
        assert _get_device(None) == torch.device("cpu")

    # 2. Test with device as a string
    assert _get_device("cpu") == torch.device("cpu")
    assert _get_device("CPU") == torch.device("cpu")

    with patch("torch.cuda.is_available", return_value=True):
        assert _get_device("cuda") == torch.device("cuda")
        assert _get_device("CUDA") == torch.device("cuda")

    with patch("torch.cuda.is_available", return_value=False), \
         pytest.raises(ValueError, match="CUDA selected, but not available"):
        _get_device("cuda")

    with pytest.raises(ValueError, match="Device must be 'cuda', 'cpu'"):
        _get_device("tpu")

    # 3. Test with device as a torch.device object
    assert _get_device(torch.device("cpu")) == torch.device("cpu")

    with patch("torch.cuda.is_available", return_value=True):
        assert _get_device(torch.device("cuda")) == torch.device("cuda")

    with patch("torch.cuda.is_available", return_value=False), \
         pytest.raises(ValueError, match="CUDA selected, but not available"):
        _get_device(torch.device("cuda"))

    # 4. Test with invalid device type
    with pytest.raises(TypeError, match="Device must be 'cuda', 'cpu'"):
        _get_device(123)
    with pytest.raises(TypeError, match="Device must be 'cuda', 'cpu'"):
        _get_device([torch.device("cpu")])
    with pytest.raises(ValueError, match="CUDA selected, but not available"):
        _get_device(torch.device("cuda"))

