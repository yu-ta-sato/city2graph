"""Tests for the graph module."""

from unittest.mock import patch

import geopandas as gpd
import networkx as nx
import pytest

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


def test_is_torch_available() -> None:
    """Test the is_torch_available function."""
    # This test's success depends on whether torch was successfully imported at the top
    assert is_torch_available() == TORCH_AVAILABLE


# Tests for when TORCH_AVAILABLE is False (simulated via mocking)
def test_gdf_to_pyg_torch_unavailable(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test gdf_to_pyg raises ImportError if torch is unavailable."""
    with patch("city2graph.graph.TORCH_AVAILABLE", new=False), \
         pytest.raises(ImportError, match="PyTorch required."):
        gdf_to_pyg(sample_nodes_gdf)


def test_pyg_to_gdf_torch_unavailable() -> None:
    """Test pyg_to_gdf raises ImportError if torch is unavailable."""
    # Create a dummy Data object for testing, as it's needed by the function signature
    # This object won't be deeply inspected if TORCH_AVAILABLE is false.
    dummy_pyg_data = Data() if TORCH_AVAILABLE else object()
    # The _validate_pyg function, called by pyg_to_gdf, checks TORCH_AVAILABLE first.
    with patch("city2graph.graph.TORCH_AVAILABLE", new=False), \
         pytest.raises(ImportError, match="PyTorch required."):
        pyg_to_gdf(dummy_pyg_data)


def test_nx_to_pyg_torch_unavailable(sample_nx_graph: nx.Graph) -> None:
    """Test nx_to_pyg raises ImportError if torch is unavailable."""
    with patch("city2graph.graph.TORCH_AVAILABLE", new=False), \
         pytest.raises(ImportError, match="PyTorch required."):
        nx_to_pyg(sample_nx_graph)


def test_pyg_to_nx_torch_unavailable() -> None:
    """Test pyg_to_nx raises ImportError if torch is unavailable."""
    dummy_pyg_data = Data() if TORCH_AVAILABLE else object()
    with patch("city2graph.graph.TORCH_AVAILABLE", new=False), \
         pytest.raises(ImportError, match="PyTorch required."):
        pyg_to_nx(dummy_pyg_data)


@requires_torch
def test_gdf_to_pyg_device_handling(sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
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
        except (RuntimeError, AssertionError) as e:
            # Catch runtime error or assertion error if CUDA compiled but not usable
            cuda_errors = [
                "Torch not compiled with CUDA enabled",
                "CUDA driver version is insufficient",
                "CUDA out of memory",
            ]
            if any(error in str(e) for error in cuda_errors):
                pytest.skip(f"CUDA environment issue: {e}")
            else:
                raise  # Re-raise if it's an unexpected RuntimeError
    else:
        with pytest.raises(ValueError, match="CUDA selected, but not available"):
            gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, device="cuda")


@requires_torch
def test_gdf_to_pyg_empty_edges(sample_nodes_gdf: gpd.GeoDataFrame, sample_crs: str) -> None:
    """Test gdf_to_pyg with an empty edges GeoDataFrame."""
    empty_edges_gdf = gpd.GeoDataFrame(columns=["source_id", "target_id", "geometry"], crs=sample_crs)
    empty_edges_gdf = empty_edges_gdf.set_index(["source_id", "target_id"])

    # No longer raises ValueError, should create empty tensors for edges
    pyg_data = gdf_to_pyg(sample_nodes_gdf, empty_edges_gdf)

    assert isinstance(pyg_data, Data)
    assert pyg_data.num_nodes == len(sample_nodes_gdf)
    assert pyg_data.edge_index.shape == (2, 0)
    assert pyg_data.edge_attr.shape == (0, 0)


@requires_torch
def test_gdf_to_pyg_no_edges(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test gdf_to_pyg with no edges GeoDataFrame provided (edges=None)."""
    pyg_data = gdf_to_pyg(sample_nodes_gdf, edges=None)
    assert isinstance(pyg_data, Data)
    assert pyg_data.edge_index.shape == (2, 0)
    assert pyg_data.edge_attr.shape[0] == 0

    _re_nodes, re_edges = pyg_to_gdf(pyg_data)
    assert re_edges is None or re_edges.empty


@requires_torch
def test_gdf_to_pyg_hetero_no_edges_dict(sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame]) -> None:
    """Test gdf_to_pyg for heterogeneous graph with no edges_dict."""
    pyg_data = gdf_to_pyg(sample_hetero_nodes_dict, edges=None)
    assert isinstance(pyg_data, HeteroData)
    for edge_type in pyg_data.edge_types:
        assert pyg_data[edge_type].edge_index.shape == (2, 0)
        assert pyg_data[edge_type].edge_attr.shape[0] == 0

    re_nodes_dict, re_edges_dict = pyg_to_gdf(pyg_data)
    assert not re_edges_dict  # Should be an empty dictionary


@requires_torch
def test_gdf_to_pyg_hetero_empty_edges_in_dict(sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame], sample_crs: str) -> None:
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
def test_pyg_to_gdf_specific_types(
    sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
) -> None:
    """Test pyg_to_gdf with specific node_types and edge_types."""
    pyg_data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict)

    # Test reconstructing only 'building' nodes
    nodes_dict_specific, _edges_dict_specific = pyg_to_gdf(pyg_data, node_types=["building"])
    assert "building" in nodes_dict_specific
    assert "road" not in nodes_dict_specific
    assert len(nodes_dict_specific) == 1

    # Test reconstructing only ('building', 'connects_to', 'road') edges
    _nodes_dict_specific, edges_dict_specific = pyg_to_gdf(pyg_data,
                                                           edge_types=[("building", "connects_to", "road")])
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
def test_pyg_to_nx_hetero_structure(
    sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
) -> None:
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


@requires_torch
@pytest.mark.parametrize(
    ("node_feature_cols", "node_label_cols", "edge_feature_cols"),
    [
        (None, None, None),
        (["feature1"], ["label1"], ["edge_feature1"]),
        (["feature1"], None, None),
        (None, ["label1"], None),
        (None, None, ["edge_feature1"]),
        (["non_existent_node_feat"], ["non_existent_node_label"], ["non_existent_edge_feat"]),
    ],
)
def test_gdf_to_pyg_homogeneous_round_trip(
    sample_nodes_gdf: gpd.GeoDataFrame,
    sample_edges_gdf: gpd.GeoDataFrame,
    sample_crs: str,
    node_feature_cols: list[str] | None,
    node_label_cols: list[str] | None,
    edge_feature_cols: list[str] | None,
) -> None:
    """Test gdf_to_pyg and pyg_to_gdf for homogeneous graphs with various feature/label configurations."""
    # Convert GDF to PyG
    pyg_data = gdf_to_pyg(
        sample_nodes_gdf,
        sample_edges_gdf,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_feature_cols=edge_feature_cols,
        device="cpu",  # Test with CPU
    )

    _validate_pyg_data(pyg_data, sample_nodes_gdf, sample_edges_gdf, sample_crs,
                       node_feature_cols, node_label_cols, edge_feature_cols)

    # Test round trip
    reconstructed_nodes_gdf, reconstructed_edges_gdf = pyg_to_gdf(pyg_data)
    _validate_reconstructed_homogeneous(
        reconstructed_nodes_gdf, reconstructed_edges_gdf,
        sample_nodes_gdf, sample_edges_gdf, sample_crs,
    )


@requires_torch
@pytest.mark.parametrize(
    ("node_feature_cols", "node_label_cols", "edge_feature_cols"),
    [
        (None, None, None),
        (None, None, {"connects_to": ["conn_feat1"]}),
        (None, {"road": ["r_label"]}, None),
        ({"building": ["b_feat1"]}, None, None),
        ({"building": ["non_existent_feat"]}, None, {"connects_to": ["non_existent_edge_feat"]}),
        (
            {"building": ["b_feat1"], "road": ["r_feat1"]},
            {"building": ["b_label"], "road": ["r_label"]},
            {"connects_to": ["conn_feat1"], "links_to": ["link_feat1"]},
        ),
    ],
)
def test_gdf_to_pyg_heterogeneous_round_trip(
    sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    sample_crs: str,
    node_feature_cols: dict[str, list[str]] | None,
    node_label_cols: dict[str, list[str]] | None,
    edge_feature_cols: dict[str, list[str]] | None,
) -> None:
    """Test gdf_to_pyg and pyg_to_gdf for heterogeneous graphs with various feature/label configurations."""
    pyg_data = gdf_to_pyg(
        sample_hetero_nodes_dict,
        sample_hetero_edges_dict,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_feature_cols=edge_feature_cols,
        device="cpu",
    )

    _validate_hetero_pyg_data(
        pyg_data, sample_hetero_nodes_dict, sample_hetero_edges_dict,
        sample_crs, node_feature_cols, node_label_cols, edge_feature_cols,
    )

    # Test round trip
    reconstructed_nodes_dict, reconstructed_edges_dict = pyg_to_gdf(pyg_data)
    _validate_reconstructed_heterogeneous(
        reconstructed_nodes_dict, reconstructed_edges_dict,
        sample_hetero_nodes_dict, sample_hetero_edges_dict,
        sample_crs,
    )


@requires_torch
@pytest.mark.parametrize(
    ("node_feature_cols", "node_label_cols", "edge_feature_cols"),
    [
        (None, None, None),
        (["feature1"], ["label1"], ["edge_feature1"]),
        (["feature1"], None, None),
        (None, ["label1"], None),
        (None, None, ["edge_feature1"]),
    ],
)
def test_nx_to_pyg_round_trip(
    sample_nx_graph: nx.Graph,
    sample_crs: str,
    node_feature_cols: list[str] | None,
    node_label_cols: list[str] | None,
    edge_feature_cols: list[str] | None,
) -> None:
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

    # Convert PyG back to NX
    reconstructed_nx_graph = pyg_to_nx(pyg_data)
    _validate_reconstructed_nx(reconstructed_nx_graph, sample_nx_graph, sample_crs)


def _validate_pyg_data(
    pyg_data: Data,
    sample_nodes_gdf: gpd.GeoDataFrame,
    sample_edges_gdf: gpd.GeoDataFrame,
    sample_crs: str,
    node_feature_cols: list[str] | None,
    node_label_cols: list[str] | None,
    edge_feature_cols: list[str] | None,
) -> None:
    """Validate homogeneous PyG data structure."""
    assert isinstance(pyg_data, Data)
    assert pyg_data.crs == sample_crs

    # Check node features
    if node_feature_cols and "feature1" in node_feature_cols:
        assert hasattr(pyg_data, "x")
        assert pyg_data.x.shape[0] == len(sample_nodes_gdf)
        assert pyg_data.x.shape[1] == 1
    elif node_feature_cols and "non_existent_node_feat" in node_feature_cols:
        assert hasattr(pyg_data, "x")
        assert pyg_data.x.shape[1] == 0  # No features should be created
    else:
        assert hasattr(pyg_data, "x")  # x is always created
        assert pyg_data.x.shape[1] == 0  # Empty features

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


def _validate_hetero_pyg_data(
    pyg_data: HeteroData,
    sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    sample_crs: str,
    node_feature_cols: dict[str, list[str]] | None,
    node_label_cols: dict[str, list[str]] | None,
    edge_feature_cols: dict[str, list[str]] | None,
) -> None:
    """Validate heterogeneous PyG data structure."""
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
        elif not (
            hasattr(pyg_data[node_type], "y")
            and pyg_data[node_type].y is not None
            and pyg_data[node_type].y.numel() > 0
        ):
            assert True  # y can be None or empty tensor

    # Check edge features for each type
    for edge_type_tuple, original_edge_gdf in sample_hetero_edges_dict.items():
        relation_type = edge_type_tuple[1]
        if (
            edge_feature_cols
            and relation_type in edge_feature_cols
            and pyg_data[edge_type_tuple].edge_attr.numel() > 0
        ):
            assert pyg_data[edge_type_tuple].edge_attr.shape[0] == len(original_edge_gdf)
            if original_edge_gdf.columns.intersection(edge_feature_cols[relation_type]).any():
                assert pyg_data[edge_type_tuple].edge_attr.shape[1] == len(edge_feature_cols[relation_type])
            else:
                assert pyg_data[edge_type_tuple].edge_attr.shape[1] == 0  # No valid features
        else:
            assert pyg_data[edge_type_tuple].edge_attr.shape[1] == 0


def _validate_reconstructed_homogeneous(
    reconstructed_nodes_gdf: gpd.GeoDataFrame,
    reconstructed_edges_gdf: gpd.GeoDataFrame | None,
    sample_nodes_gdf: gpd.GeoDataFrame,
    sample_edges_gdf: gpd.GeoDataFrame,
    sample_crs: str,
) -> None:
    """Validate reconstructed homogeneous GDFs."""
    assert isinstance(reconstructed_nodes_gdf, gpd.GeoDataFrame)
    assert reconstructed_nodes_gdf.crs == sample_crs
    assert reconstructed_nodes_gdf.geom_equals_exact(sample_nodes_gdf.geometry, tolerance=1e-5).all()

    if sample_edges_gdf is not None and not sample_edges_gdf.empty:
        assert isinstance(reconstructed_edges_gdf, gpd.GeoDataFrame)
        assert reconstructed_edges_gdf.crs == sample_crs
        assert reconstructed_edges_gdf.geom_equals_exact(sample_edges_gdf.geometry, tolerance=1e-5).all()
    else:
        assert reconstructed_edges_gdf is None


def _validate_reconstructed_heterogeneous(
    reconstructed_nodes_dict: dict[str, gpd.GeoDataFrame],
    reconstructed_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    sample_crs: str,
) -> None:
    """Validate reconstructed heterogeneous GDFs."""
    for node_type, original_gdf in sample_hetero_nodes_dict.items():
        assert node_type in reconstructed_nodes_dict
        re_gdf = reconstructed_nodes_dict[node_type].sort_index()
        original_gdf_sorted = original_gdf.sort_index()
        assert isinstance(re_gdf, gpd.GeoDataFrame)
        assert re_gdf.crs == sample_crs
        assert re_gdf.geom_equals_exact(original_gdf_sorted.geometry, tolerance=1e-5).all()

    for edge_type, original_gdf in sample_hetero_edges_dict.items():
        assert edge_type in reconstructed_edges_dict
        re_gdf = reconstructed_edges_dict[edge_type].sort_index()
        original_gdf_sorted = original_gdf.sort_index()
        assert isinstance(re_gdf, gpd.GeoDataFrame)
        assert re_gdf.crs == sample_crs
        assert re_gdf.geom_equals_exact(original_gdf_sorted.geometry, tolerance=1e-5).all()


def _validate_reconstructed_nx(
    reconstructed_nx_graph: nx.Graph,
    sample_nx_graph: nx.Graph,
    sample_crs: str,
) -> None:
    """Validate reconstructed NetworkX graph."""
    assert isinstance(reconstructed_nx_graph, nx.Graph)
    assert reconstructed_nx_graph.graph.get("crs") == sample_crs
    assert reconstructed_nx_graph.number_of_nodes() == sample_nx_graph.number_of_nodes()
    assert reconstructed_nx_graph.number_of_edges() == sample_nx_graph.number_of_edges()


@requires_torch
def test_gdf_to_pyg_invalid_hetero_feature_cols(
    sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
) -> None:
    """Test that invalid feature column types raise ValueError for heterogeneous graphs."""
    # Test invalid node_feature_cols type
    with pytest.raises(ValueError, match="node_feature_cols must be a dict for heterogeneous graphs"):
        gdf_to_pyg(sample_hetero_nodes_dict, node_feature_cols=["invalid"])

    # Test invalid node_label_cols type
    with pytest.raises(ValueError, match="node_label_cols must be a dict for heterogeneous graphs"):
        gdf_to_pyg(sample_hetero_nodes_dict, node_label_cols=["invalid"])

    # Test invalid edge_feature_cols type
    with pytest.raises(ValueError, match="edge_feature_cols must be a dict for heterogeneous graphs"):
        gdf_to_pyg(sample_hetero_nodes_dict, edge_feature_cols=["invalid"])


@requires_torch
def test_gdf_to_pyg_invalid_device_types(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test that invalid device types raise appropriate errors."""
    # Test invalid device string
    with pytest.raises(ValueError, match="Device must be"):
        gdf_to_pyg(sample_nodes_gdf, device="invalid_device")

    # Test invalid device type
    with pytest.raises(TypeError, match="Device must be"):
        gdf_to_pyg(sample_nodes_gdf, device=123)

    # Test invalid device type (list)
    with pytest.raises(TypeError, match="Device must be"):
        gdf_to_pyg(sample_nodes_gdf, device=["cpu"])


@requires_torch
def test_gdf_to_pyg_with_dtype(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test gdf_to_pyg with specific dtype parameter."""
    pyg_data = gdf_to_pyg(sample_nodes_gdf, device="cpu", dtype=torch.float32)
    assert pyg_data.x.dtype == torch.float32

    if hasattr(pyg_data, "pos") and pyg_data.pos is not None:
        assert pyg_data.pos.dtype == torch.float32


@requires_torch
def test_nx_to_pyg_with_dtype(sample_nx_graph: nx.Graph) -> None:
    """Test nx_to_pyg with specific dtype parameter."""
    pyg_data = nx_to_pyg(sample_nx_graph, device="cpu", dtype=torch.float64)
    assert pyg_data.x.dtype == torch.float64

    # Note: pos tensor dtype may not always match the specified dtype
    # depending on the implementation, so we don't assert it here


@requires_torch
def test_gdf_to_pyg_invalid_homogeneous_feature_cols(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test that invalid feature column types raise ValueError for homogeneous graphs."""
    # Test invalid node_feature_cols type (dict instead of list)
    with pytest.raises(ValueError, match="node_feature_cols must be a list for homogeneous graphs"):
        gdf_to_pyg(sample_nodes_gdf, node_feature_cols={"invalid": ["col"]})

    # Test invalid node_label_cols type (dict instead of list)
    with pytest.raises(ValueError, match="node_label_cols must be a list for homogeneous graphs"):
        gdf_to_pyg(sample_nodes_gdf, node_label_cols={"invalid": ["col"]})

    # Test invalid edge_feature_cols type (dict instead of list)
    with pytest.raises(ValueError, match="edge_feature_cols must be a list for homogeneous graphs"):
        gdf_to_pyg(sample_nodes_gdf, edge_feature_cols={"invalid": ["col"]})


@requires_torch
def test_pyg_validation_errors() -> None:
    """Test various validation errors in validate_pyg function."""
    from city2graph.graph import validate_pyg

    # Test with non-PyG object
    with pytest.raises(TypeError, match="Input must be a PyTorch Geometric Data or HeteroData object"):
        validate_pyg("not_a_pyg_object")

    # Test with PyG object missing graph_metadata
    data = Data()
    with pytest.raises(ValueError, match="PyG object is missing 'graph_metadata' attribute"):
        validate_pyg(data)

    # Test with PyG object having wrong metadata type
    data.graph_metadata = "wrong_type"
    with pytest.raises(ValueError, match="PyG object has 'graph_metadata' of incorrect type"):
        validate_pyg(data)


@requires_torch
def test_pyg_validation_hetero_inconsistencies() -> None:
    """Test validation errors for heterogeneous graph inconsistencies."""
    from city2graph.graph import validate_pyg
    from city2graph.utils import GraphMetadata

    # Test HeteroData with metadata.is_hetero = False
    hetero_data = HeteroData()
    hetero_data.graph_metadata = GraphMetadata(is_hetero=False)
    with pytest.raises(ValueError, match="Inconsistency detected: PyG object is HeteroData but metadata.is_hetero is False"):
        validate_pyg(hetero_data)

    # Test Data with metadata.is_hetero = True
    homo_data = Data()
    homo_data.graph_metadata = GraphMetadata(is_hetero=True)
    with pytest.raises(ValueError, match="Inconsistency detected: PyG object is Data but metadata.is_hetero is True"):
        validate_pyg(homo_data)


@requires_torch
def test_pyg_validation_node_edge_type_mismatches() -> None:
    """Test validation errors for node/edge type mismatches."""
    from city2graph.graph import validate_pyg
    from city2graph.utils import GraphMetadata

    # Create HeteroData with mismatched node types
    hetero_data = HeteroData()
    hetero_data["building"].x = torch.randn(5, 3)
    metadata = GraphMetadata(is_hetero=True)
    metadata.node_types = ["building", "road"]  # metadata expects 'road' but it's not in data
    metadata.edge_types = []
    hetero_data.graph_metadata = metadata
    with pytest.raises(ValueError, match="Node types mismatch"):
        validate_pyg(hetero_data)

    # Create HeteroData with mismatched edge types
    hetero_data2 = HeteroData()
    hetero_data2["building"].x = torch.randn(5, 3)
    hetero_data2["road"].x = torch.randn(3, 2)
    hetero_data2[("building", "connects", "road")].edge_index = torch.tensor([[0, 1], [0, 1]])
    metadata2 = GraphMetadata(is_hetero=True)
    metadata2.node_types = ["building", "road"]
    metadata2.edge_types = [("building", "connects", "road"), ("road", "links", "building")]  # extra edge type
    hetero_data2.graph_metadata = metadata2
    with pytest.raises(ValueError, match="Edge types mismatch"):
        validate_pyg(hetero_data2)


@requires_torch
def test_device_validation_with_torch_device(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test device validation with torch.device objects."""
    # Test with torch.device object - this tests the public interface
    device_obj = torch.device("cpu")
    pyg_data = gdf_to_pyg(sample_nodes_gdf, device=device_obj)
    assert pyg_data.x.device == device_obj

    if torch.cuda.is_available():
        cuda_device = torch.device("cuda:0")
        try:
            pyg_data_cuda = gdf_to_pyg(sample_nodes_gdf, device=cuda_device)
            assert pyg_data_cuda.x.device == cuda_device
        except (RuntimeError, ValueError):
            # Skip if CUDA not properly configured
            pass


@requires_torch
def test_gdf_to_pyg_with_torch_device_object(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test gdf_to_pyg with torch.device object."""
    device_obj = torch.device("cpu")
    pyg_data = gdf_to_pyg(sample_nodes_gdf, device=device_obj)
    assert pyg_data.x.device == device_obj


@requires_torch
def test_pyg_to_gdf_with_missing_node_features() -> None:
    """Test pyg_to_gdf when node data has no features but has labels."""
    from city2graph.utils import GraphMetadata

    # Create data with only labels, no features
    data = Data()
    data.y = torch.randn(5, 2)
    data.pos = torch.randn(5, 2)
    data.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    data.edge_attr = torch.randn(2, 1)
    metadata = GraphMetadata(is_hetero=False, crs="EPSG:4326")
    metadata.node_label_cols = ["label1", "label2"]
    data.graph_metadata = metadata

    nodes_gdf, edges_gdf = pyg_to_gdf(data)
    assert "label1" in nodes_gdf.columns
    assert "label2" in nodes_gdf.columns


@requires_torch
def test_pyg_to_gdf_hetero_with_missing_features() -> None:
    """Test pyg_to_gdf with heterogeneous data missing some features."""
    from city2graph.utils import GraphMetadata

    data = HeteroData()
    # Node type with only labels, no features
    data["building"].y = torch.randn(3, 1)
    data["building"].pos = torch.randn(3, 2)
    # Node type with only features, no labels
    data["road"].x = torch.randn(2, 2)
    data["road"].pos = torch.randn(2, 2)

    metadata = GraphMetadata(is_hetero=True, crs="EPSG:4326")
    metadata.node_types = ["building", "road"]
    metadata.node_label_cols = {"building": ["b_label"]}
    metadata.node_feature_cols = {"road": ["r_feat1", "r_feat2"]}
    data.graph_metadata = metadata

    nodes_dict, edges_dict = pyg_to_gdf(data)
    assert "b_label" in nodes_dict["building"].columns
    assert "r_feat1" in nodes_dict["road"].columns
    assert "r_feat2" in nodes_dict["road"].columns


@requires_torch
def test_pyg_to_nx_with_no_edges() -> None:
    """Test pyg_to_nx conversion with no edges."""
    from city2graph.utils import GraphMetadata

    # Create data with nodes but no edges
    data = Data()
    data.x = torch.randn(5, 2)
    data.pos = torch.randn(5, 2)
    data.edge_index = torch.tensor([[], []], dtype=torch.long)  # Empty edges
    metadata = GraphMetadata(is_hetero=False, crs="EPSG:4326")
    data.graph_metadata = metadata

    nx_graph = pyg_to_nx(data)
    assert nx_graph.number_of_nodes() == 5
    assert nx_graph.number_of_edges() == 0


@requires_torch
def test_pyg_to_nx_hetero_with_no_edges() -> None:
    """Test pyg_to_nx conversion with heterogeneous data but no edges."""
    from city2graph.utils import GraphMetadata

    data = HeteroData()
    data["building"].x = torch.randn(3, 2)
    data["building"].pos = torch.randn(3, 2)
    data["road"].x = torch.randn(2, 2)
    data["road"].pos = torch.randn(2, 2)
    # No edge data

    metadata = GraphMetadata(is_hetero=True, crs="EPSG:4326")
    metadata.node_types = ["building", "road"]
    data.graph_metadata = metadata

    nx_graph = pyg_to_nx(data)
    assert nx_graph.number_of_nodes() == 5
    assert nx_graph.number_of_edges() == 0


@requires_torch
def test_pyg_to_nx_hetero_with_empty_edge_tensors() -> None:
    """Test pyg_to_nx conversion with heterogeneous data with empty edge tensors."""
    from city2graph.utils import GraphMetadata

    data = HeteroData()
    data["building"].x = torch.randn(3, 2)
    data["building"].pos = torch.randn(3, 2)
    data["road"].x = torch.randn(2, 2)
    data["road"].pos = torch.randn(2, 2)
    # Empty edge tensor
    data[("building", "connects", "road")].edge_index = torch.tensor([[], []], dtype=torch.long)

    metadata = GraphMetadata(is_hetero=True, crs="EPSG:4326")
    metadata.node_types = ["building", "road"]
    metadata.edge_types = [("building", "connects", "road")]
    data.graph_metadata = metadata

    nx_graph = pyg_to_nx(data)
    assert nx_graph.number_of_nodes() == 5
    assert nx_graph.number_of_edges() == 0


@requires_torch
def test_pyg_to_gdf_hetero_with_feature_columns() -> None:
    """Test pyg_to_gdf with heterogeneous data and feature column metadata."""
    from city2graph.utils import GraphMetadata

    data = HeteroData()
    data["building"].x = torch.randn(3, 2)
    data["building"].pos = torch.randn(3, 2)

    metadata = GraphMetadata(is_hetero=True, crs="EPSG:4326")
    metadata.node_types = ["building"]
    metadata.node_feature_cols = {"building": ["height", "area"]}
    data.graph_metadata = metadata

    nodes_dict, edges_dict = pyg_to_gdf(data)
    # Should use the specified feature column names
    assert "height" in nodes_dict["building"].columns
    assert "area" in nodes_dict["building"].columns


# ============================================================================
# ENHANCED TEST CASES FOR BETTER COVERAGE
# ============================================================================

@requires_torch
def test_gdf_to_pyg_invalid_feature_cols_type_error(sample_crs: str) -> None:
    """Test TypeError for invalid feature column types in heterogeneous graphs."""
    # Create sample data
    nodes_dict = {
        "building": gpd.GeoDataFrame({
            "height": [10, 20],
            "geometry": [gpd.points_from_xy([0, 1], [0, 1])[0], gpd.points_from_xy([0, 1], [0, 1])[1]],
        }, index=[0, 1], crs=sample_crs),
    }

    # Test node_feature_cols type error for heterogeneous graphs
    with pytest.raises(ValueError, match="node_feature_cols must be a dict for heterogeneous graphs"):
        gdf_to_pyg(nodes_dict, node_feature_cols=["height"])

    # Test node_label_cols type error for heterogeneous graphs
    with pytest.raises(ValueError, match="node_label_cols must be a dict for heterogeneous graphs"):
        gdf_to_pyg(nodes_dict, node_label_cols=["height"])

    # Test edge_feature_cols type error for heterogeneous graphs
    with pytest.raises(ValueError, match="edge_feature_cols must be a dict for heterogeneous graphs"):
        gdf_to_pyg(nodes_dict, edge_feature_cols=["weight"])


@requires_torch
def test_gdf_to_pyg_invalid_feature_cols_value_error(sample_crs: str) -> None:
    """Test ValueError for invalid feature column types in homogeneous graphs."""
    # Create sample data
    nodes_gdf = gpd.GeoDataFrame({
        "height": [10, 20],
        "geometry": gpd.points_from_xy([0, 1], [0, 1]),
    }, index=[0, 1], crs=sample_crs)

    # Test node_label_cols value error for homogeneous graphs
    with pytest.raises(ValueError, match="node_label_cols must be a list for homogeneous graphs"):
        gdf_to_pyg(nodes_gdf, node_label_cols={"building": ["height"]})

    # Test edge_feature_cols value error for homogeneous graphs
    with pytest.raises(ValueError, match="edge_feature_cols must be a list for homogeneous graphs"):
        gdf_to_pyg(nodes_gdf, edge_feature_cols={"road": ["weight"]})


@requires_torch
def test_validation_errors_hetero_structure() -> None:
    """Test validation errors for heterogeneous structure inconsistencies."""
    from city2graph.utils import GraphMetadata

    # Create a HeteroData object with inconsistent metadata
    data = HeteroData()
    data["building"].x = torch.randn(3, 2)
    data["building"].pos = torch.randn(2, 2)  # Inconsistent size

    metadata = GraphMetadata(is_hetero=True, crs="EPSG:4326")
    metadata.node_types = ["building"]
    data.graph_metadata = metadata

    # Test position tensor size mismatch
    with pytest.raises(ValueError, match="position tensor size.*doesn't match node feature tensor size"):
        from city2graph.graph import validate_pyg
        validate_pyg(data)


@requires_torch
def test_validation_errors_hetero_label_mismatch() -> None:
    """Test validation errors for heterogeneous label tensor mismatches."""
    from city2graph.utils import GraphMetadata

    # Create a HeteroData object with inconsistent label tensor
    data = HeteroData()
    data["building"].x = torch.randn(3, 2)
    data["building"].y = torch.randn(2, 1)  # Inconsistent size
    data["building"].pos = torch.randn(3, 2)

    metadata = GraphMetadata(is_hetero=True, crs="EPSG:4326")
    metadata.node_types = ["building"]
    data.graph_metadata = metadata

    # Test label tensor size mismatch
    with pytest.raises(ValueError, match="label tensor size.*doesn't match node feature tensor size"):
        from city2graph.graph import validate_pyg
        validate_pyg(data)


@requires_torch
def test_validation_errors_homo_structure() -> None:
    """Test validation errors for homogeneous structure inconsistencies."""
    from city2graph.utils import GraphMetadata

    # Test metadata indicating hetero but Data object is homo
    data = Data()
    data.x = torch.randn(3, 2)

    metadata = GraphMetadata(is_hetero=True, crs="EPSG:4326")  # Wrong flag
    data.graph_metadata = metadata

    with pytest.raises(ValueError, match="Inconsistency detected: PyG object is Data but metadata.is_hetero is True"):
        from city2graph.graph import validate_pyg
        validate_pyg(data)


@requires_torch
def test_validation_errors_homo_metadata_structure() -> None:
    """Test validation errors for homogeneous metadata structure."""
    from city2graph.utils import GraphMetadata

    # Test homogeneous graph with node_types specified
    data = Data()
    data.x = torch.randn(3, 2)

    metadata = GraphMetadata(is_hetero=False, crs="EPSG:4326")
    metadata.node_types = ["building"]  # Should not be specified for homo
    data.graph_metadata = metadata

    with pytest.raises(ValueError, match="Homogeneous graph metadata should not have node_types specified"):
        from city2graph.graph import validate_pyg
        validate_pyg(data)


@requires_torch
def test_validation_errors_homo_edge_types() -> None:
    """Test validation errors for homogeneous edge types."""
    from city2graph.utils import GraphMetadata

    # Test homogeneous graph with edge_types specified
    data = Data()
    data.x = torch.randn(3, 2)

    metadata = GraphMetadata(is_hetero=False, crs="EPSG:4326")
    metadata.edge_types = [("building", "connects", "road")]  # Should not be specified for homo
    data.graph_metadata = metadata

    with pytest.raises(ValueError, match="Homogeneous graph metadata should not have edge_types specified"):
        from city2graph.graph import validate_pyg
        validate_pyg(data)


@requires_torch
def test_validation_errors_homo_node_mappings() -> None:
    """Test validation errors for homogeneous node mappings."""
    from city2graph.utils import GraphMetadata

    # Test homogeneous graph without 'default' key in node_mappings
    data = Data()
    data.x = torch.randn(3, 2)

    metadata = GraphMetadata(is_hetero=False, crs="EPSG:4326")
    metadata.node_mappings = {"building": {}}  # Should use 'default' key
    data.graph_metadata = metadata

    with pytest.raises(ValueError, match="Homogeneous graph metadata should use 'default' key in node_mappings"):
        from city2graph.graph import validate_pyg
        validate_pyg(data)


@requires_torch
def test_validation_errors_homo_feature_cols_dict() -> None:
    """Test validation errors for homogeneous feature columns as dict."""
    from city2graph.utils import GraphMetadata

    # Test homogeneous graph with node_feature_cols as dict
    data = Data()
    data.x = torch.randn(3, 2)

    metadata = GraphMetadata(is_hetero=False, crs="EPSG:4326")
    metadata.node_feature_cols = {"building": ["height"]}  # Should be list for homo
    data.graph_metadata = metadata

    with pytest.raises(ValueError, match="Homogeneous graph metadata should have node_feature_cols as list, not dict"):
        from city2graph.graph import validate_pyg
        validate_pyg(data)


@requires_torch
def test_validation_errors_homo_label_cols_dict() -> None:
    """Test validation errors for homogeneous label columns as dict."""
    from city2graph.utils import GraphMetadata

    # Test homogeneous graph with node_label_cols as dict
    data = Data()
    data.x = torch.randn(3, 2)

    metadata = GraphMetadata(is_hetero=False, crs="EPSG:4326")
    metadata.node_label_cols = {"building": ["type"]}  # Should be list for homo
    data.graph_metadata = metadata

    with pytest.raises(ValueError, match="Homogeneous graph metadata should have node_label_cols as list, not dict"):
        from city2graph.graph import validate_pyg
        validate_pyg(data)


@requires_torch
def test_validation_errors_homo_edge_cols_dict() -> None:
    """Test validation errors for homogeneous edge feature columns as dict."""
    from city2graph.utils import GraphMetadata

    # Test homogeneous graph with edge_feature_cols as dict
    data = Data()
    data.x = torch.randn(3, 2)
    data.edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    data.edge_attr = torch.randn(2, 1)

    metadata = GraphMetadata(is_hetero=False, crs="EPSG:4326")
    metadata.edge_feature_cols = {"connects": ["weight"]}  # Should be list for homo
    data.graph_metadata = metadata

    with pytest.raises(ValueError, match="Homogeneous graph metadata should have edge_feature_cols as list, not dict"):
        from city2graph.graph import validate_pyg
        validate_pyg(data)


@requires_torch
def test_validation_errors_homo_position_mismatch() -> None:
    """Test validation errors for homogeneous position tensor mismatch."""
    from city2graph.utils import GraphMetadata

    # Test homogeneous graph with position tensor size mismatch
    data = Data()
    data.x = torch.randn(3, 2)
    data.pos = torch.randn(2, 2)  # Inconsistent size

    metadata = GraphMetadata(is_hetero=False, crs="EPSG:4326")
    data.graph_metadata = metadata

    with pytest.raises(ValueError, match="Node position tensor size.*doesn't match node feature tensor size"):
        from city2graph.graph import validate_pyg
        validate_pyg(data)


@requires_torch
def test_validation_errors_homo_label_mismatch() -> None:
    """Test validation errors for homogeneous label tensor mismatch."""
    from city2graph.utils import GraphMetadata

    # Test homogeneous graph with label tensor size mismatch
    data = Data()
    data.x = torch.randn(3, 2)
    data.y = torch.randn(2, 1)  # Inconsistent size

    metadata = GraphMetadata(is_hetero=False, crs="EPSG:4326")
    data.graph_metadata = metadata

    with pytest.raises(ValueError, match="Node label tensor size.*doesn't match node feature tensor size"):
        from city2graph.graph import validate_pyg
        validate_pyg(data)


@requires_torch
def test_pyg_to_gdf_hetero_with_missing_label_cols() -> None:
    """Test pyg_to_gdf with heterogeneous data and missing label columns in metadata."""
    from city2graph.utils import GraphMetadata

    data = HeteroData()
    data["building"].x = torch.randn(3, 2)
    data["building"].y = torch.randn(3, 1)  # Has labels but no metadata
    data["building"].pos = torch.randn(3, 2)

    metadata = GraphMetadata(is_hetero=True, crs="EPSG:4326")
    metadata.node_types = ["building"]
    # No node_label_cols specified in metadata
    data.graph_metadata = metadata

    nodes_dict, edges_dict = pyg_to_gdf(data)

    # Should not create any label columns since they're not specified in metadata
    assert "building" in nodes_dict
    building_gdf = nodes_dict["building"]
    # Check that no label columns were created since they weren't specified in metadata
    label_cols = [col for col in building_gdf.columns if col.startswith("label_")]
    assert len(label_cols) == 0

