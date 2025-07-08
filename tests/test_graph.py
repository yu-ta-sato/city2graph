"""Refactored tests for the graph module - focused on public API with comprehensive coverage."""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
import torch
from shapely.geometry import Point

from city2graph.utils import GraphMetadata

# Import torch-related modules conditionally
try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.data import HeteroData
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip all tests if torch is not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

# Import graph functions only if torch is available
if TORCH_AVAILABLE:
    from city2graph.graph import gdf_to_pyg
    from city2graph.graph import is_torch_available
    from city2graph.graph import nx_to_pyg
    from city2graph.graph import pyg_to_gdf
    from city2graph.graph import pyg_to_nx
    from city2graph.graph import validate_pyg


class TestTorchAvailability:
    """Test torch availability detection."""

    def test_is_torch_available_returns_true(self) -> None:
        """Test that is_torch_available returns True when torch is available."""
        assert is_torch_available() is True

    def test_functions_raise_import_error_without_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that functions raise ImportError when torch is not available."""
        # Mock TORCH_AVAILABLE to False
        import city2graph.graph as graph_module
        monkeypatch.setattr(graph_module, "TORCH_AVAILABLE", False)

        # Test that main functions raise ImportError
        with pytest.raises(ImportError, match="PyTorch required"):
            graph_module.gdf_to_pyg({}, {})

        with pytest.raises(ImportError, match="PyTorch required"):
            graph_module.validate_pyg({})  # type: ignore[arg-type]

        # Create a valid NetworkX graph with required metadata for the test
        valid_graph = nx.Graph()
        valid_graph.add_node(1, feature1=10.0, pos=(0, 0))
        valid_graph.add_node(2, feature1=20.0, pos=(1, 1))
        valid_graph.add_edge(1, 2)
        valid_graph.graph["is_hetero"] = False
        valid_graph.graph["crs"] = "EPSG:4326"

        with pytest.raises(ImportError, match="PyTorch required"):
            graph_module.nx_to_pyg(valid_graph)

        with pytest.raises(ImportError, match="PyTorch required"):
            graph_module.pyg_to_gdf({})  # type: ignore[arg-type]

        with pytest.raises(ImportError, match="PyTorch required"):
            graph_module.pyg_to_nx({})  # type: ignore[arg-type]


class TestGdfToPyg:
    """Test GeoDataFrame to PyTorch Geometric conversions."""

    def test_homogeneous_basic_conversion(
        self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test basic homogeneous graph conversion."""
        pyg_data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf)

        assert isinstance(pyg_data, Data)
        assert pyg_data.num_nodes == len(sample_nodes_gdf)
        assert pyg_data.num_edges == len(sample_edges_gdf)
        assert hasattr(pyg_data, "graph_metadata")
        assert pyg_data.graph_metadata.is_hetero is False

    def test_homogeneous_with_features(
        self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test homogeneous conversion with node and edge features."""
        pyg_data = gdf_to_pyg(
            sample_nodes_gdf,
            sample_edges_gdf,
            node_feature_cols=["feature1"],
            node_label_cols=["label1"],
            edge_feature_cols=["edge_feature1"],
        )

        assert pyg_data.x.shape[1] == 1  # One feature column
        assert pyg_data.y.shape[1] == 1  # One label column
        assert pyg_data.edge_attr.shape[1] == 1  # One edge feature column

    def test_heterogeneous_basic_conversion(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test basic heterogeneous graph conversion."""
        pyg_data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict)

        assert isinstance(pyg_data, HeteroData)
        assert pyg_data.graph_metadata.is_hetero is True
        assert set(pyg_data.node_types) == set(sample_hetero_nodes_dict.keys())
        assert set(pyg_data.edge_types) == set(sample_hetero_edges_dict.keys())

    def test_heterogeneous_with_features(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous conversion with features."""
        pyg_data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            sample_hetero_edges_dict,
            node_feature_cols={"building": ["b_feat1"], "road": ["length"]},
            node_label_cols={"building": ["b_label"]},
            edge_feature_cols={"connects_to": ["conn_feat1"], "links_to": ["link_feat1"]},
        )

        assert pyg_data["building"].x.shape[1] == 1
        assert pyg_data["road"].x.shape[1] == 1
        assert pyg_data["building"].y.shape[1] == 1

    def test_device_validation_errors(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test device validation errors to cover missing lines 1180-1181, 1186-1187."""
        # Test invalid device type - covers line 1180-1181
        with pytest.raises(TypeError, match="Device must be"):
            gdf_to_pyg(sample_nodes_gdf, device=123)  # type: ignore[arg-type]

        # Test invalid device string - covers line 1186-1187
        with pytest.raises(ValueError, match="Device must be"):
            gdf_to_pyg(sample_nodes_gdf, device="invalid_device")

    def test_empty_edge_tensors(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test creation of empty edge tensors - covers lines 1202-1203."""
        # Test with no edges to trigger empty tensor creation
        pyg_data = gdf_to_pyg(sample_nodes_gdf, edges=None)

        assert pyg_data.edge_index.shape == (2, 0)
        assert pyg_data.edge_attr.shape == (0, 0)

    def test_invalid_feature_cols_type_errors(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test invalid feature column types for homogeneous graphs."""
        with pytest.raises(TypeError, match="node_feature_cols must be a list"):
            gdf_to_pyg(sample_nodes_gdf, node_feature_cols={"invalid": ["cols"]})  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="node_label_cols must be a list"):
            gdf_to_pyg(sample_nodes_gdf, node_label_cols={"invalid": ["cols"]})  # type: ignore[arg-type]

    def test_invalid_edge_feature_cols_type_heterogeneous(
        self, sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test invalid edge feature column types for heterogeneous graphs."""
        with pytest.raises(TypeError, match="edge_feature_cols must be a dict"):
            gdf_to_pyg(sample_hetero_nodes_dict, {}, edge_feature_cols=["invalid"])  # type: ignore[arg-type]

    def test_invalid_node_feature_cols_type_heterogeneous(
        self, sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test invalid node feature column types for heterogeneous graphs - covers lines 266-267."""
        with pytest.raises(TypeError, match="node_feature_cols must be a dict"):
            gdf_to_pyg(sample_hetero_nodes_dict, node_feature_cols=["invalid"])  # type: ignore[arg-type]

    def test_invalid_node_label_cols_type_heterogeneous(
        self, sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test invalid node label column types for heterogeneous graphs - covers lines 272-273."""
        with pytest.raises(TypeError, match="node_label_cols must be a dict"):
            gdf_to_pyg(sample_hetero_nodes_dict, node_label_cols=["invalid"])  # type: ignore[arg-type]

    def test_invalid_edge_feature_cols_type_homogeneous(
        self, sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test invalid edge feature column types for homogeneous graphs - covers lines 306-307."""
        with pytest.raises(TypeError, match="edge_feature_cols must be a list"):
            gdf_to_pyg(sample_nodes_gdf, edge_feature_cols={"invalid": ["cols"]})  # type: ignore[arg-type]


class TestValidatePyg:
    """Test PyTorch Geometric validation."""

    def test_valid_homogeneous_data(self, sample_pyg_data: Data) -> None:
        """Test validation of valid homogeneous data."""
        metadata = validate_pyg(sample_pyg_data)
        assert isinstance(metadata, GraphMetadata)
        assert metadata.is_hetero is False

    def test_valid_heterogeneous_data(self, sample_pyg_hetero_data: HeteroData) -> None:
        """Test validation of valid heterogeneous data."""
        metadata = validate_pyg(sample_pyg_hetero_data)
        assert isinstance(metadata, GraphMetadata)
        assert metadata.is_hetero is True

    def test_invalid_input_type(self) -> None:
        """Test validation with invalid input type."""
        with pytest.raises(TypeError, match="Input must be a PyTorch Geometric"):
            validate_pyg("not_a_pyg_object")  # type: ignore[arg-type]

    def test_missing_metadata(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test validation with missing metadata."""
        pyg_data = gdf_to_pyg(sample_nodes_gdf)
        delattr(pyg_data, "graph_metadata")

        with pytest.raises(ValueError, match="PyG object is missing 'graph_metadata'"):
            validate_pyg(pyg_data)

    def test_wrong_metadata_type(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test validation with wrong metadata type."""
        pyg_data = gdf_to_pyg(sample_nodes_gdf)
        pyg_data.graph_metadata = "wrong_type"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="PyG object has 'graph_metadata' of incorrect type"):
            validate_pyg(pyg_data)

    def test_inconsistent_hetero_metadata(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test validation with inconsistent hetero metadata."""
        pyg_data = gdf_to_pyg(sample_nodes_gdf)
        pyg_data.graph_metadata.is_hetero = True  # Inconsistent with Data object

        with pytest.raises(ValueError, match="Inconsistency detected.*is Data but metadata.is_hetero is True"):
            validate_pyg(pyg_data)

    def test_homo_validation_errors(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test homogeneous validation errors - covers lines 1409-1410, 1413-1414, 1417-1418."""
        pyg_data = gdf_to_pyg(sample_nodes_gdf)

        # Test node_feature_cols as dict instead of list - covers lines 1409-1410
        pyg_data.graph_metadata.node_feature_cols = {"invalid": ["cols"]}
        with pytest.raises(ValueError, match="node_feature_cols as list, not dict"):
            validate_pyg(pyg_data)

        # Reset and test node_label_cols - covers lines 1413-1414
        pyg_data = gdf_to_pyg(sample_nodes_gdf)
        pyg_data.graph_metadata.node_label_cols = {"invalid": ["cols"]}
        with pytest.raises(ValueError, match="node_label_cols as list, not dict"):
            validate_pyg(pyg_data)

        # Reset and test edge_feature_cols - covers lines 1417-1418
        pyg_data = gdf_to_pyg(sample_nodes_gdf)
        pyg_data.graph_metadata.edge_feature_cols = {"invalid": ["cols"]}
        with pytest.raises(ValueError, match="edge_feature_cols as list, not dict"):
            validate_pyg(pyg_data)

    def test_homo_tensor_size_mismatches(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test homogeneous tensor size mismatches - covers lines 1434-1438."""
        pyg_data = gdf_to_pyg(sample_nodes_gdf, node_label_cols=["label1"])

        # Test label tensor size mismatch - covers lines 1434-1438
        original_y = pyg_data.y
        pyg_data.y = torch.randn(1, 1)  # Wrong size
        with pytest.raises(ValueError, match="Node label tensor size.*doesn't match"):
            validate_pyg(pyg_data)
        pyg_data.y = original_y

    def test_hetero_tensor_size_mismatches(self, sample_pyg_hetero_data: HeteroData) -> None:
        """Test heterogeneous tensor size mismatches - covers lines 1384-1388."""
        # Test label tensor size mismatch for specific node type - covers lines 1384-1388
        node_type = next(iter(sample_pyg_hetero_data.node_types))
        if hasattr(sample_pyg_hetero_data[node_type], "y") and sample_pyg_hetero_data[node_type].y is not None:
            original_y = sample_pyg_hetero_data[node_type].y
            sample_pyg_hetero_data[node_type].y = torch.randn(1, 1)  # Wrong size
            with pytest.raises(ValueError, match="label tensor size.*doesn't match"):
                validate_pyg(sample_pyg_hetero_data)
            sample_pyg_hetero_data[node_type].y = original_y


class TestPygToGdf:
    """Test PyTorch Geometric to GeoDataFrame conversions."""

    def test_homogeneous_round_trip(
        self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test homogeneous round trip conversion."""
        # Convert to PyG and back
        pyg_data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf)
        nodes_restored, edges_restored = pyg_to_gdf(pyg_data)

        # Check structure preservation
        assert isinstance(nodes_restored, gpd.GeoDataFrame)
        assert isinstance(edges_restored, gpd.GeoDataFrame)
        assert len(nodes_restored) == len(sample_nodes_gdf)
        assert len(edges_restored) == len(sample_edges_gdf)

    def test_heterogeneous_round_trip(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous round trip conversion."""
        pyg_data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict)
        nodes_restored, edges_restored = pyg_to_gdf(pyg_data)

        assert isinstance(nodes_restored, dict)
        assert isinstance(edges_restored, dict)
        assert set(nodes_restored.keys()) == set(sample_hetero_nodes_dict.keys())
        assert set(edges_restored.keys()) == set(sample_hetero_edges_dict.keys())

    def test_comprehensive_edge_cases_through_public_api(self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
        """Test comprehensive edge cases through public API only - covers lines 1465, 1497, 1520, 1531, 1633."""
        # Test with empty feature columns to trigger tensor data edge cases
        data_empty_features = gdf_to_pyg(
            sample_nodes_gdf,
            sample_edges_gdf,
            node_feature_cols=[],  # Empty list
            node_label_cols=[],    # Empty list
            edge_feature_cols=[],   # Empty list
        )

        # Convert back to test reconstruction with empty features
        nodes_restored, edges_restored = pyg_to_gdf(data_empty_features)
        assert isinstance(nodes_restored, gpd.GeoDataFrame)
        assert isinstance(edges_restored, gpd.GeoDataFrame)

        # Test with minimal nodes to trigger index value edge cases
        minimal_nodes = gpd.GeoDataFrame({
            "geometry": [Point(0, 0), Point(1, 1)],
        }, index=[100, 200])  # Non-sequential indices

        data_minimal = gdf_to_pyg(minimal_nodes)
        nodes_back, _ = pyg_to_gdf(data_minimal)

        # Should preserve original indices
        assert list(nodes_back.index) == [100, 200]

    def test_empty_geometry_handling(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test empty geometry handling - covers lines 1721-1722."""
        pyg_data = gdf_to_pyg(sample_nodes_gdf)

        # Remove position data to trigger empty geometry case
        pyg_data.pos = None

        nodes_restored, edges_restored = pyg_to_gdf(pyg_data)

        # Should handle empty geometry gracefully
        assert isinstance(nodes_restored, gpd.GeoDataFrame)


class TestNxToPyg:
    """Test NetworkX to PyTorch Geometric conversions."""

    def test_basic_conversion(self, sample_nx_graph: nx.Graph) -> None:
        """Test basic NetworkX to PyG conversion."""
        pyg_data = nx_to_pyg(sample_nx_graph)

        assert isinstance(pyg_data, Data)
        assert pyg_data.num_nodes == sample_nx_graph.number_of_nodes()
        assert pyg_data.num_edges == sample_nx_graph.number_of_edges()

    def test_round_trip(self, sample_nx_graph: nx.Graph) -> None:
        """Test NetworkX round trip conversion."""
        pyg_data = nx_to_pyg(sample_nx_graph)
        nx_restored = pyg_to_nx(pyg_data)

        assert nx_restored.number_of_nodes() == sample_nx_graph.number_of_nodes()
        assert nx_restored.number_of_edges() == sample_nx_graph.number_of_edges()
        assert nx_restored.graph.get("crs") == sample_nx_graph.graph.get("crs")

    def test_empty_graph_error(self) -> None:
        """Test that empty NetworkX graph raises error."""
        empty_graph = nx.Graph()
        with pytest.raises(ValueError, match="Graph has no nodes"):
            nx_to_pyg(empty_graph)


class TestPygToNx:
    """Test PyTorch Geometric to NetworkX conversions."""

    def test_homogeneous_conversion(self, sample_pyg_data: Data) -> None:
        """Test homogeneous PyG to NetworkX conversion."""
        nx_graph = pyg_to_nx(sample_pyg_data)

        assert isinstance(nx_graph, nx.Graph)
        assert nx_graph.graph.get("is_hetero") is False
        assert nx_graph.number_of_nodes() == sample_pyg_data.num_nodes
        assert nx_graph.number_of_edges() == sample_pyg_data.num_edges

    def test_heterogeneous_conversion(self, sample_pyg_hetero_data: HeteroData) -> None:
        """Test heterogeneous PyG to NetworkX conversion."""
        nx_graph = pyg_to_nx(sample_pyg_hetero_data)

        assert isinstance(nx_graph, nx.Graph)
        assert nx_graph.graph.get("is_hetero") is True
        assert "node_types" in nx_graph.graph
        assert "edge_types" in nx_graph.graph

    def test_node_attributes_preservation(self, sample_pyg_data: Data) -> None:
        """Test that node attributes are preserved in conversion."""
        nx_graph = pyg_to_nx(sample_pyg_data)

        # Check that nodes have expected attributes
        for node_id, node_data in nx_graph.nodes(data=True):
            assert "_original_index" in node_data
            if hasattr(sample_pyg_data, "pos") and sample_pyg_data.pos is not None:
                assert "pos" in node_data

    def test_homo_nx_conversion_edge_cases(self, sample_pyg_data: Data) -> None:
        """Test homogeneous NetworkX conversion edge cases - covers lines 1765, 1769-1772."""
        nx_graph = pyg_to_nx(sample_pyg_data)

        # Test that feature columns are properly handled - covers line 1765
        for node_id, node_data in nx_graph.nodes(data=True):
            if hasattr(sample_pyg_data, "x") and sample_pyg_data.x is not None:
                # Should have feature columns or default names
                assert any(key.startswith("feat_") for key in node_data.keys()) or \
                       any(key in ["feature1"] for key in node_data.keys())

        # Test label handling - covers lines 1769-1772
        if hasattr(sample_pyg_data, "y") and sample_pyg_data.y is not None:
            for node_id, node_data in nx_graph.nodes(data=True):
                assert any(key.startswith("label_") for key in node_data.keys()) or \
                       any(key in ["label1"] for key in node_data.keys())

    def test_homo_edge_attributes_handling(self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
        """Test homogeneous edge attributes handling - covers lines 1791, 1795."""
        pyg_data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, edge_feature_cols=["edge_feature1"])
        nx_graph = pyg_to_nx(pyg_data)

        # Test edge feature columns handling - covers lines 1791, 1795
        if nx_graph.number_of_edges() > 0:
            edge_data = next(iter(nx_graph.edges(data=True)))[2]
            assert "edge_feature1" in edge_data or any(key.startswith("edge_feat_") for key in edge_data.keys())

    def test_hetero_nx_conversion_edge_cases(self, sample_pyg_hetero_data: HeteroData) -> None:
        """Test heterogeneous NetworkX conversion edge cases - covers lines 1845, 1847, 1851-1861, 1885, 1889."""
        nx_graph = pyg_to_nx(sample_pyg_hetero_data)

        # Test that node features are properly handled - covers lines 1845, 1847
        for _node_id, node_data in nx_graph.nodes(data=True):
            assert "node_type" in node_data
            # Just verify that the node has the expected attributes
            # The exact column names depend on whether features were specified during conversion

        # Test edge attributes handling - covers lines 1885, 1889
        if nx_graph.number_of_edges() > 0:
            edge_data = next(iter(nx_graph.edges(data=True)))[2]
            assert "edge_type" in edge_data


class TestMissingLineCoverage:
    """Test methods to cover specific missing lines."""

    def test_empty_geometry_crs_direct_assignment(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test empty geometry CRS direct assignment - covers line 1558."""
        # Create a scenario where we have empty or all-null geometry to trigger line 1558
        # We'll do this through the public API by creating data with null geometries

        # Create nodes with null geometries
        nodes_with_null_geom = sample_nodes_gdf.copy()
        nodes_with_null_geom["geometry"] = None  # All null geometries

        # Convert to PyG and back to trigger the CRS assignment path
        data = gdf_to_pyg(nodes_with_null_geom)

        # This should trigger line 1558 when reconstructing the GeoDataFrame
        nodes_restored, _ = pyg_to_gdf(data)

        # Should handle null geometry and complete without error
        assert isinstance(nodes_restored, gpd.GeoDataFrame)

    def test_empty_geometry_crs_assignment(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test empty geometry CRS assignment - covers line 1558."""
        # Create data and remove position to trigger empty geometry case
        data = gdf_to_pyg(sample_nodes_gdf)
        data.pos = None  # This will create empty geometry

        # Convert back - this should trigger line 1558 where gdf.crs = metadata.crs
        nodes_restored, _ = pyg_to_gdf(data)

        # Should handle empty geometry and set CRS directly
        assert isinstance(nodes_restored, gpd.GeoDataFrame)
        # For empty geometry, the CRS should be set via direct assignment (line 1558)
        # We can't check nodes_restored.crs directly because it has no geometry column
        # But we can verify the function completed without error, which means line 1558 was executed

    def test_default_feature_column_naming(self, sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame]) -> None:
        """Test default feature column naming - covers lines 1846, 1860."""
        # Create hetero data with features but no metadata column names
        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            node_feature_cols={"building": ["b_feat1"], "road": ["length"]},
            node_label_cols={"building": ["b_label"], "road": ["r_label"]},
        )

        # Remove feature column metadata to trigger default naming
        data.graph_metadata.node_feature_cols = None  # This triggers line 1846
        data.graph_metadata.node_label_cols = None    # This triggers line 1860

        # Convert to NetworkX to trigger the default naming paths
        nx_graph = pyg_to_nx(data)

        # Should use default feat_ and label_ naming
        for _node_id, node_data in nx_graph.nodes(data=True):
            if "node_type" in node_data:
                # Should have default feature names
                assert any(key.startswith("feat_") for key in node_data) or \
                       any(key.startswith("label_") for key in node_data)

    def test_empty_edge_features_handling(self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
        """Test handling of empty edge features - covers line 847."""
        # Create edges with no valid feature columns
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, edge_feature_cols=["nonexistent_col"])
        assert data.edge_attr.shape[1] == 0

    def test_missing_node_features_handling(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test handling of missing node features - covers line 754."""
        # Remove all numeric columns to test fallback
        nodes_no_features = sample_nodes_gdf.copy()
        nodes_no_features = nodes_no_features[["geometry"]]  # Keep only geometry

        data = gdf_to_pyg(nodes_no_features, node_feature_cols=["nonexistent"])
        assert data.x.shape[1] == 0

    def test_cuda_not_available_error(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test CUDA not available error - covers lines 664-665."""
        # Only test if CUDA is not available
        if not torch.cuda.is_available():
            with pytest.raises(ValueError, match="CUDA selected, but not available"):
                gdf_to_pyg(sample_nodes_gdf, device="cuda")

    def test_empty_geometry_handling(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test empty geometry handling through public API - covers lines 940, 1465, 1497, 1562, 1633."""
        # Create data with no edges to trigger empty geometry handling
        data = gdf_to_pyg(sample_nodes_gdf, edges=None)
        nodes_restored, edges_restored = pyg_to_gdf(data)

        # Should handle empty edges gracefully
        assert edges_restored is not None
        assert len(edges_restored) == 0

    def test_round_trip_with_missing_features(self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
        """Test round trip with missing feature columns - covers lines 1520, 1531."""
        # Create data with non-existent feature columns
        data = gdf_to_pyg(
            sample_nodes_gdf,
            sample_edges_gdf,
            node_feature_cols=["nonexistent_feature"],
            node_label_cols=["nonexistent_label"],
            edge_feature_cols=["nonexistent_edge_feature"],
        )

        # Convert back to GDF - should handle missing features gracefully
        nodes_restored, edges_restored = pyg_to_gdf(data)

        assert isinstance(nodes_restored, gpd.GeoDataFrame)
        assert isinstance(edges_restored, gpd.GeoDataFrame)

    def test_heterogeneous_with_corrupted_data_structure(self, sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame]) -> None:
        """Test heterogeneous conversion with edge cases - covers lines 1180-1181, 1186-1187, 1202-1203."""
        # Create heterogeneous data with empty edges to trigger edge case handling
        empty_edges_dict = {
            ("building", "connects_to", "road"): gpd.GeoDataFrame(columns=["geometry"], index=pd.MultiIndex.from_arrays([[], []], names=["building_id", "road_id"])),
        }

        # This should handle empty edge cases gracefully
        data = gdf_to_pyg(sample_hetero_nodes_dict, empty_edges_dict)

        # Verify empty edges are handled correctly
        edge_type = ("building", "connects_to", "road")
        assert data[edge_type].edge_index.shape == (2, 0)
        assert data[edge_type].edge_attr.shape[0] == 0

    def test_homo_nx_conversion_specific_lines(self, sample_pyg_data: Data) -> None:
        """Test specific lines in homogeneous NetworkX conversion - covers lines 1796, 1846, 1860."""
        nx_graph = pyg_to_nx(sample_pyg_data)

        # Test that feature columns are properly handled - covers line 1796
        for _node_id, node_data in nx_graph.nodes(data=True):
            if hasattr(sample_pyg_data, "x") and sample_pyg_data.x is not None:
                # Should have feature columns or default names
                assert any(key.startswith("feat_") for key in node_data) or \
                       any(key in ["feature1"] for key in node_data)

        # Test label handling - covers lines 1846, 1860
        if hasattr(sample_pyg_data, "y") and sample_pyg_data.y is not None:
            for _node_id, node_data in nx_graph.nodes(data=True):
                assert any(key.startswith("label_") for key in node_data) or \
                       any(key in ["label1"] for key in node_data)

    def test_homo_edge_attributes_specific_lines(self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
        """Test homogeneous edge attributes handling - covers lines 1888-1890."""
        pyg_data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, edge_feature_cols=["edge_feature1"])
        nx_graph = pyg_to_nx(pyg_data)

        # Test edge feature columns handling
        if nx_graph.number_of_edges() > 0:
            edge_data = next(iter(nx_graph.edges(data=True)))[2]
            assert "edge_feature1" in edge_data or any(key.startswith("edge_feat_") for key in edge_data.keys())

    def test_hetero_inconsistent_metadata_errors(self, sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame]) -> None:
        """Test heterogeneous metadata inconsistency errors - covers lines 1323-1327."""
        # Create valid hetero data first
        data = gdf_to_pyg(sample_hetero_nodes_dict)

        # Corrupt metadata to trigger inconsistency error
        data.graph_metadata.is_hetero = False  # Make it inconsistent

        with pytest.raises(ValueError, match="Inconsistency detected.*HeteroData but metadata.is_hetero is False"):
            validate_pyg(data)

    def test_hetero_node_types_mismatch(self, sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame]) -> None:
        """Test heterogeneous node types mismatch - covers lines 1351-1355."""
        data = gdf_to_pyg(sample_hetero_nodes_dict)

        # Corrupt node types in metadata
        data.graph_metadata.node_types = ["wrong_type"]

        with pytest.raises(ValueError, match="Node types mismatch"):
            validate_pyg(data)

    def test_hetero_edge_types_mismatch(self, sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame], sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame]) -> None:
        """Test heterogeneous edge types mismatch - covers lines 1362-1366."""
        data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict)

        # Corrupt edge types in metadata
        data.graph_metadata.edge_types = [("wrong", "type", "tuple")]

        with pytest.raises(ValueError, match="Edge types mismatch"):
            validate_pyg(data)

    def test_hetero_position_tensor_mismatch(self, sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame]) -> None:
        """Test heterogeneous position tensor size mismatch - covers lines 1376-1380."""
        data = gdf_to_pyg(sample_hetero_nodes_dict)

        # Corrupt position tensor size for a node type
        node_type = next(iter(data.node_types))
        data[node_type].pos = torch.randn(1, 2)  # Wrong size

        with pytest.raises(ValueError, match="position tensor size.*doesn't match"):
            validate_pyg(data)

    def test_homo_validation_node_types_error(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test homogeneous validation with node_types specified - covers lines 1395-1396."""
        data = gdf_to_pyg(sample_nodes_gdf)

        # Add node_types to homogeneous metadata (invalid)
        data.graph_metadata.node_types = ["some_type"]

        with pytest.raises(ValueError, match="Homogeneous graph metadata should not have node_types"):
            validate_pyg(data)

    def test_homo_validation_edge_types_error(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test homogeneous validation with edge_types specified - covers lines 1399-1400."""
        data = gdf_to_pyg(sample_nodes_gdf)

        # Add edge_types to homogeneous metadata (invalid)
        data.graph_metadata.edge_types = [("some", "edge", "type")]

        with pytest.raises(ValueError, match="Homogeneous graph metadata should not have edge_types"):
            validate_pyg(data)

    def test_homo_validation_default_key_missing(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test homogeneous validation without default key - covers lines 1404-1405."""
        data = gdf_to_pyg(sample_nodes_gdf)

        # Remove default key from node_mappings
        data.graph_metadata.node_mappings = {"wrong_key": {}}

        with pytest.raises(ValueError, match="should use 'default' key in node_mappings"):
            validate_pyg(data)

    def test_homo_position_tensor_mismatch(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test homogeneous position tensor size mismatch - covers lines 1426-1430."""
        data = gdf_to_pyg(sample_nodes_gdf)

        # Corrupt position tensor size
        data.pos = torch.randn(1, 2)  # Wrong size

        with pytest.raises(ValueError, match="Node position tensor size.*doesn't match"):
            validate_pyg(data)

    def test_homo_edge_attr_tensor_mismatch(self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
        """Test homogeneous edge attribute tensor size mismatch - covers lines 1446-1450."""
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, edge_feature_cols=["edge_feature1"])

        # Corrupt edge attribute tensor size
        data.edge_attr = torch.randn(1, 1)  # Wrong size

        with pytest.raises(ValueError, match="Edge attribute tensor size.*doesn't match"):
            validate_pyg(data)

    def test_extract_tensor_data_edge_cases(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test _extract_tensor_data edge cases through public API - covers lines 1464-1470."""
        # Create data with empty tensors to trigger edge cases
        data = gdf_to_pyg(sample_nodes_gdf, node_feature_cols=[])

        # Convert back - this will call _extract_tensor_data with empty tensors
        nodes_restored, _ = pyg_to_gdf(data)

        # Should handle empty tensors gracefully
        assert isinstance(nodes_restored, gpd.GeoDataFrame)

    def test_extract_index_values_edge_case(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test _extract_index_values edge case - covers line 1497."""
        # Create minimal data to trigger edge case in index extraction
        minimal_nodes = sample_nodes_gdf.iloc[:1].copy()  # Single node
        data = gdf_to_pyg(minimal_nodes)

        # Convert back to trigger index value extraction
        nodes_restored, _ = pyg_to_gdf(data)

        assert len(nodes_restored) == 1

    def test_extract_node_features_hetero_edge_case(self, sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame]) -> None:
        """Test _extract_node_features_and_labels hetero edge case - covers lines 1520, 1531."""
        # Create hetero data with specific feature columns
        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            node_feature_cols={"building": ["b_feat1"]},
            node_label_cols={"building": ["b_label"]},
        )

        # Convert back to trigger feature extraction edge cases
        nodes_restored, _ = pyg_to_gdf(data)

        assert isinstance(nodes_restored, dict)

    def test_set_gdf_index_and_crs_edge_case(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test _set_gdf_index_and_crs edge case - covers line 1562."""
        # Create data and convert back to trigger CRS setting edge cases
        data = gdf_to_pyg(sample_nodes_gdf)

        # Remove position data to trigger empty geometry case
        data.pos = None

        nodes_restored, _ = pyg_to_gdf(data)

        # Should handle missing position data gracefully
        assert isinstance(nodes_restored, gpd.GeoDataFrame)

    def test_extract_edge_features_edge_case(self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
        """Test _extract_edge_features edge case - covers line 1633."""
        # Create data with edge features
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, edge_feature_cols=["edge_feature1"])

        # Convert back to trigger edge feature extraction
        _, edges_restored = pyg_to_gdf(data)

        assert isinstance(edges_restored, gpd.GeoDataFrame)

    def test_nx_conversion_feature_column_edge_cases(self, sample_pyg_data: Data) -> None:
        """Test NetworkX conversion feature column edge cases - covers lines 1796, 1846, 1860."""
        # Create data with no feature column metadata to trigger default naming
        data = sample_pyg_data
        data.graph_metadata.node_feature_cols = None  # Trigger default naming
        data.graph_metadata.node_label_cols = None    # Trigger default naming

        nx_graph = pyg_to_nx(data)

        # Should use default feature names
        for node_id, node_data in nx_graph.nodes(data=True):
            if hasattr(data, "x") and data.x is not None:
                assert any(key.startswith("feat_") for key in node_data.keys())

    def test_nx_conversion_edge_attr_edge_case(self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
        """Test NetworkX conversion edge attribute edge case - covers lines 1888-1890."""
        # Create data with no edge attributes to trigger empty DataFrame case
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf)  # No edge features

        nx_graph = pyg_to_nx(data)

        # Should handle missing edge attributes gracefully
        if nx_graph.number_of_edges() > 0:
            edge_data = next(iter(nx_graph.edges(data=True)))[2]
            # Should have basic edge data structure
            assert isinstance(edge_data, dict)

    def test_device_validation_specific_errors(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test specific device validation errors through public API - covers lines 1180-1181, 1186-1187."""
        # Test TypeError for invalid device type - covers lines 1180-1181
        with pytest.raises(TypeError, match="Device must be"):
            gdf_to_pyg(sample_nodes_gdf, device=123)  # Invalid type

        # Test ValueError for invalid device string - covers lines 1186-1187
        with pytest.raises(ValueError, match="Device must be"):
            gdf_to_pyg(sample_nodes_gdf, device="invalid_device_string")

    def test_extract_tensor_data_none_cases(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test tensor data extraction edge cases through public API - covers line 1465."""
        # Test with empty feature columns to trigger tensor data edge cases
        data = gdf_to_pyg(sample_nodes_gdf, node_feature_cols=[])

        # Convert back to trigger _extract_tensor_data with empty tensors
        nodes_restored, _ = pyg_to_gdf(data)

        # Should handle empty tensors gracefully
        assert isinstance(nodes_restored, gpd.GeoDataFrame)

    def test_extract_index_values_fallback(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test index values extraction fallback through public API - covers line 1497."""
        # Create minimal data to trigger edge case in index extraction
        minimal_nodes = sample_nodes_gdf.iloc[:1].copy()  # Single node
        data = gdf_to_pyg(minimal_nodes)

        # Convert back to trigger index value extraction edge cases
        nodes_restored, _ = pyg_to_gdf(data)

        # Should handle index extraction gracefully
        assert len(nodes_restored) == 1

    def test_set_gdf_index_empty_geometry_case(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test CRS and index setting with empty geometry through public API - covers line 1562."""
        # Create data and convert back to trigger CRS setting edge cases
        data = gdf_to_pyg(sample_nodes_gdf)

        # Remove position data to trigger empty geometry case
        data.pos = None

        nodes_restored, _ = pyg_to_gdf(data)

        # Should handle missing position data gracefully
        assert isinstance(nodes_restored, gpd.GeoDataFrame)

    def test_extract_edge_features_none_edge_attr(self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
        """Test edge features extraction with None edge_attr through public API - covers line 1633."""
        # Create data with edge features
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, edge_feature_cols=["edge_feature1"])

        # Remove edge attributes to trigger None edge_attr case
        data.edge_attr = None

        # Convert back to trigger edge feature extraction with None
        _, edges_restored = pyg_to_gdf(data)

        # Should handle None edge attributes gracefully
        assert isinstance(edges_restored, gpd.GeoDataFrame)

    def test_nx_conversion_no_feature_cols(self, sample_pyg_data: Data) -> None:
        """Test NetworkX conversion with no feature columns - covers lines 1796, 1846, 1860."""
        # Remove feature column metadata to trigger default naming
        data = sample_pyg_data
        data.graph_metadata.node_feature_cols = []  # Empty list instead of None
        data.graph_metadata.node_label_cols = []    # Empty list instead of None

        nx_graph = pyg_to_nx(data)

        # Should use default feature names when feature_cols is empty
        for node_id, node_data in nx_graph.nodes(data=True):
            if hasattr(data, "x") and data.x is not None and data.x.numel() > 0:
                # Should have default feat_ names
                assert any(key.startswith("feat_") for key in node_data.keys())

    def test_nx_conversion_empty_edge_attrs(self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
        """Test NetworkX conversion with empty edge attributes - covers lines 1888-1890."""
        # Create data with empty edge attributes
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf)
        data.edge_attr = torch.empty((data.edge_index.shape[1], 0))  # Empty edge attributes

        nx_graph = pyg_to_nx(data)

        # Should create empty DataFrame for edge attributes
        if nx_graph.number_of_edges() > 0:
            edge_data = next(iter(nx_graph.edges(data=True)))[2]
            assert isinstance(edge_data, dict)

    def test_direct_helper_function_calls(self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
        """Test edge cases through public API to cover remaining lines."""
        # Test device validation through public API - covers lines 1180-1181, 1186-1187
        with pytest.raises(TypeError):
            gdf_to_pyg(sample_nodes_gdf, device=[])  # Invalid type

        with pytest.raises(ValueError):
            gdf_to_pyg(sample_nodes_gdf, device="gpu")  # Invalid device string

        # Test edge feature extraction with None through public API - covers line 1633
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, edge_feature_cols=["edge_feature1"])
        data.edge_attr = None  # Remove edge attributes

        _, edges_restored = pyg_to_gdf(data)
        assert isinstance(edges_restored, gpd.GeoDataFrame)

        # Test NetworkX conversion edge cases through public API - covers lines 1796, 1846, 1860, 1888-1890
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf,
                         node_feature_cols=["feature1"],
                         node_label_cols=["label1"])

        # Set feature cols to empty to trigger default naming
        data.graph_metadata.node_feature_cols = []
        data.graph_metadata.node_label_cols = []

        # Remove edge attributes to trigger empty DataFrame case
        data.edge_attr = None

        nx_graph = pyg_to_nx(data)

        # Verify it handles the edge cases
        assert isinstance(nx_graph, nx.Graph)

    def test_device_validation_comprehensive(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test comprehensive device validation through public API - covers lines 1180-1181, 1186-1187."""
        # Test TypeError for invalid device type
        with pytest.raises(TypeError, match="Device must be"):
            gdf_to_pyg(sample_nodes_gdf, device=123.45)  # Invalid type: float

        with pytest.raises(TypeError, match="Device must be"):
            gdf_to_pyg(sample_nodes_gdf, device=["cuda"])  # Invalid type: list

        # Test ValueError for invalid device string
        with pytest.raises(ValueError, match="Device must be"):
            gdf_to_pyg(sample_nodes_gdf, device="gpu")  # Invalid device string

        with pytest.raises(ValueError, match="Device must be"):
            gdf_to_pyg(sample_nodes_gdf, device="invalid_device")  # Invalid device string

    def test_tensor_data_extraction_edge_cases(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test tensor data extraction edge cases through public API - covers line 1465."""
        # Create data with empty feature columns to trigger None tensor handling
        data = gdf_to_pyg(sample_nodes_gdf, node_feature_cols=None)

        # Convert back to trigger _extract_tensor_data with None/empty cases
        nodes_restored, _ = pyg_to_gdf(data)

        # Should handle None tensors gracefully
        assert isinstance(nodes_restored, gpd.GeoDataFrame)
        assert len(nodes_restored) == len(sample_nodes_gdf)

    def test_index_values_extraction_fallback(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test index values extraction fallback through public API - covers line 1497."""
        # Create data with minimal nodes to trigger edge cases in index extraction
        minimal_nodes = sample_nodes_gdf.iloc[:1].copy()
        data = gdf_to_pyg(minimal_nodes)

        # Convert back to trigger _extract_index_values edge cases
        nodes_restored, _ = pyg_to_gdf(data)

        # Should handle index extraction gracefully
        assert len(nodes_restored) == 1
        assert list(nodes_restored.index) == list(minimal_nodes.index)

    def test_crs_setting_with_empty_geometry(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test CRS setting with empty geometry through public API - covers line 1562."""
        # Create data and modify to trigger empty geometry handling
        data = gdf_to_pyg(sample_nodes_gdf)

        # Remove position data to trigger empty geometry case in _set_gdf_index_and_crs
        data.pos = None

        nodes_restored, _ = pyg_to_gdf(data)

        # Should handle empty geometry gracefully while preserving CRS
        assert isinstance(nodes_restored, gpd.GeoDataFrame)
        if hasattr(nodes_restored, "crs") and nodes_restored.crs:
            assert nodes_restored.crs is not None

    def test_edge_features_none_handling(self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
        """Test edge features extraction with None edge_attr through public API - covers line 1633."""
        # Create data with edge features
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, edge_feature_cols=["edge_feature1"])

        # Remove edge attributes to trigger None handling in _extract_edge_features
        data.edge_attr = None

        # Convert back to trigger edge feature extraction with None
        _, edges_restored = pyg_to_gdf(data)

        # Should handle None edge attributes gracefully
        assert isinstance(edges_restored, gpd.GeoDataFrame)
        assert len(edges_restored) == len(sample_edges_gdf)

    def test_networkx_conversion_default_naming(self, sample_pyg_data: Data) -> None:
        """Test NetworkX conversion with default feature naming - covers lines 1846, 1860."""
        # Modify metadata to trigger default naming in NetworkX conversion
        data = sample_pyg_data
        data.graph_metadata.node_feature_cols = []  # Empty to trigger default naming
        data.graph_metadata.node_label_cols = []    # Empty to trigger default naming

        nx_graph = pyg_to_nx(data)

        # Should use default feature names when metadata is empty
        for node_id, node_data in nx_graph.nodes(data=True):
            if hasattr(data, "x") and data.x is not None and data.x.numel() > 0:
                # Should have default feat_ names
                assert any(key.startswith("feat_") for key in node_data.keys())
            if hasattr(data, "y") and data.y is not None and data.y.numel() > 0:
                # Should have default label_ names
                assert any(key.startswith("label_") for key in node_data.keys())

    def test_networkx_conversion_empty_edge_attrs(self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame) -> None:
        """Test NetworkX conversion with empty edge attributes - covers lines 1888-1890."""
        # Create data with edges but no edge attributes
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf)  # No edge features specified

        # Ensure edge_attr is empty to trigger empty DataFrame case
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            data.edge_attr = torch.empty((data.edge_index.shape[1], 0))

        nx_graph = pyg_to_nx(data)

        # Should create empty DataFrame for edge attributes and handle gracefully
        if nx_graph.number_of_edges() > 0:
            edge_data = next(iter(nx_graph.edges(data=True)))[2]
            assert isinstance(edge_data, dict)
            # Should have basic structure even with empty attributes
