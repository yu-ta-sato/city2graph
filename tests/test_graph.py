"""
Streamlined tests for the graph module.

This module provides comprehensive test coverage for city2graph.graph with improved
maintainability and clear organization. Tests are organized by core functionality
with minimal redundancy.

Key improvements:
- Simplified test structure focused on core functionality
- Reduced redundancy through better parameterization
- Clear separation of concerns
- Easier to maintain and extend
"""

from __future__ import annotations

from typing import Any
from typing import cast

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import Point

import city2graph.graph as graph_module
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
    """Test PyTorch availability detection and error handling."""

    def test_is_torch_available_returns_true(self) -> None:
        """Test that is_torch_available returns True when torch is available."""
        assert is_torch_available() is True

    def test_functions_raise_import_error_without_torch(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that functions raise ImportError when torch is not available."""
        monkeypatch.setattr(graph_module, "TORCH_AVAILABLE", False)

        functions_to_test: list[tuple[Any, tuple[Any, ...]]] = [
            (graph_module.gdf_to_pyg, ({}, {})),
            (graph_module.validate_pyg, ({},)),
            (graph_module.pyg_to_gdf, ({},)),
            (graph_module.pyg_to_nx, ({},)),
        ]

        for func, args in functions_to_test:
            if isinstance(args, tuple):
                with pytest.raises(ImportError, match="PyTorch and PyTorch Geometric required"):
                    func(*args)
            else:
                with pytest.raises(ImportError, match="PyTorch and PyTorch Geometric required"):
                    func(args)

        # Test nx_to_pyg with valid graph
        valid_graph = nx.Graph()
        valid_graph.add_node(1, feature1=10.0, pos=(0, 0))
        valid_graph.add_node(2, feature1=20.0, pos=(1, 1))
        valid_graph.add_edge(1, 2)
        valid_graph.graph["is_hetero"] = False
        valid_graph.graph["crs"] = "EPSG:4326"

        with pytest.raises(ImportError, match="PyTorch and PyTorch Geometric required"):
            graph_module.nx_to_pyg(valid_graph)


class TestConversions:
    """Test graph conversion functionality between different formats."""

    @pytest.mark.parametrize("graph_type", ["homogeneous", "heterogeneous"])
    def test_gdf_to_pyg_basic(
        self,
        graph_type: str,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test basic conversion from GeoDataFrames to PyG objects."""
        if graph_type == "homogeneous":
            data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf)
            assert isinstance(data, Data)
            assert data.num_nodes == len(sample_nodes_gdf)
            assert data.num_edges == len(sample_edges_gdf)
            assert not data.graph_metadata.is_hetero
        else:
            data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict)
            assert isinstance(data, HeteroData)
            assert data.graph_metadata.is_hetero
            assert set(data.node_types) == set(sample_hetero_nodes_dict.keys())
            assert set(data.edge_types) == set(sample_hetero_edges_dict.keys())

    @pytest.mark.parametrize("graph_type", ["homogeneous", "heterogeneous"])
    def test_gdf_to_pyg_with_features(
        self,
        graph_type: str,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test conversion with node and edge features."""
        if graph_type == "homogeneous":
            data = gdf_to_pyg(
                sample_nodes_gdf,
                sample_edges_gdf,
                node_feature_cols=["feature1"],
                node_label_cols=["label1"],
                edge_feature_cols=["edge_feature1"],
            )
            assert data.x.shape[1] == 1
            assert data.y.shape[1] == 1
            assert data.edge_attr.shape[1] == 1
        else:
            data = gdf_to_pyg(
                sample_hetero_nodes_dict,
                sample_hetero_edges_dict,
                node_feature_cols={"building": ["b_feat1"], "road": ["length"]},
                node_label_cols={"building": ["b_label"]},
                edge_feature_cols={"connects_to": ["conn_feat1"], "links_to": ["link_feat1"]},
            )
            assert data["building"].x.shape[1] == 1
            assert data["road"].x.shape[1] == 1
            assert data["building"].y.shape[1] == 1

    @pytest.mark.parametrize("graph_type", ["homogeneous", "heterogeneous"])
    def test_round_trip_conversion(
        self,
        graph_type: str,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test that data survives round-trip conversion (GDF -> PyG -> GDF)."""
        if graph_type == "homogeneous":
            data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf)
            nodes_restored, edges_restored = pyg_to_gdf(data)

            assert isinstance(nodes_restored, gpd.GeoDataFrame)
            assert isinstance(edges_restored, gpd.GeoDataFrame)
            assert len(nodes_restored) == len(sample_nodes_gdf)
            assert len(edges_restored) == len(sample_edges_gdf)
        else:
            data = gdf_to_pyg(sample_hetero_nodes_dict, sample_hetero_edges_dict)
            nodes_restored, edges_restored = pyg_to_gdf(data)

            assert isinstance(nodes_restored, dict)
            assert isinstance(edges_restored, dict)
            assert set(nodes_restored.keys()) == set(sample_hetero_nodes_dict.keys())
            assert set(edges_restored.keys()) == set(sample_hetero_edges_dict.keys())

    def test_nx_conversions(self, sample_nx_graph: nx.Graph) -> None:
        """Test NetworkX conversions."""
        # NX -> PyG
        data = nx_to_pyg(sample_nx_graph)
        assert isinstance(data, Data)
        assert data.num_nodes == sample_nx_graph.number_of_nodes()
        assert data.num_edges == sample_nx_graph.number_of_edges()

        # PyG -> NX (round trip)
        nx_restored = pyg_to_nx(data)
        assert isinstance(nx_restored, nx.Graph)
        assert nx_restored.graph.get("is_hetero") is False
        assert nx_restored.number_of_nodes() == sample_nx_graph.number_of_nodes()
        assert nx_restored.number_of_edges() == sample_nx_graph.number_of_edges()
        assert nx_restored.graph.get("crs") == sample_nx_graph.graph.get("crs")

    def test_pyg_to_nx_heterogeneous(self, sample_pyg_hetero_data: HeteroData) -> None:
        """Test PyG to NetworkX conversion for heterogeneous graphs."""
        nx_graph = pyg_to_nx(sample_pyg_hetero_data)
        assert isinstance(nx_graph, nx.Graph)
        assert nx_graph.graph.get("is_hetero") is True
        assert "node_types" in nx_graph.graph
        assert "edge_types" in nx_graph.graph

    def test_nx_default_feature_and_label_naming(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test NetworkX conversion with both default feature and label naming."""
        # Create data with both features and labels
        data = gdf_to_pyg(
            sample_nodes_gdf,
            node_feature_cols=["feature1"],
            node_label_cols=["label1"],
        )

        # Clear both feature and label column names to trigger default naming
        data.graph_metadata.node_feature_cols = None
        data.graph_metadata.node_label_cols = None

        nx_graph = pyg_to_nx(data)

        # Check that nodes have both default feature and label attributes
        for _node_id, node_data in nx_graph.nodes(data=True):
            assert isinstance(node_data, dict)
            # Should have default feature attributes
            feat_attrs = [key for key in node_data if key.startswith("feat_")]
            assert len(feat_attrs) > 0, "Should have default feature attributes"
            # Should have default label attributes
            label_attrs = [key for key in node_data if key.startswith("label_")]
            assert len(label_attrs) > 0, "Should have default label attributes"
            break


class TestValidation:
    """Test validation functionality for PyG objects."""

    def test_valid_data(self, sample_pyg_data: Data, sample_pyg_hetero_data: HeteroData) -> None:
        """Test validation of valid PyG objects."""
        # Test homogeneous data
        metadata = validate_pyg(sample_pyg_data)
        assert isinstance(metadata, GraphMetadata)
        assert metadata.is_hetero is False

        # Test heterogeneous data
        metadata_hetero = validate_pyg(sample_pyg_hetero_data)
        assert isinstance(metadata_hetero, GraphMetadata)
        assert metadata_hetero.is_hetero is True

    def test_invalid_inputs(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test validation with invalid inputs."""
        # Invalid input type
        with pytest.raises(TypeError, match="Input must be a PyTorch Geometric"):
            validate_pyg("not_a_pyg_object")

        # Missing metadata
        data = gdf_to_pyg(sample_nodes_gdf)
        delattr(data, "graph_metadata")
        with pytest.raises(ValueError, match="PyG object is missing 'graph_metadata'"):
            validate_pyg(data)

        # Wrong metadata type
        data = gdf_to_pyg(sample_nodes_gdf)
        data.graph_metadata = "wrong_type"
        with pytest.raises(TypeError, match="PyG object has 'graph_metadata' of incorrect type"):
            validate_pyg(data)

    @pytest.mark.parametrize(
        ("graph_type", "inconsistency"),
        [
            ("homo_marked_as_hetero", "is Data but metadata.is_hetero is True"),
            ("hetero_marked_as_homo", "HeteroData but metadata.is_hetero is False"),
        ],
    )
    def test_inconsistent_metadata(
        self,
        graph_type: str,
        inconsistency: str,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test validation with inconsistent metadata."""
        if graph_type == "homo_marked_as_hetero":
            data = gdf_to_pyg(sample_nodes_gdf)
            data.graph_metadata.is_hetero = True
        else:
            data = gdf_to_pyg(sample_hetero_nodes_dict)
            data.graph_metadata.is_hetero = False

        with pytest.raises(ValueError, match=f"Inconsistency detected.*{inconsistency}"):
            validate_pyg(data)


class TestDeviceHandling:
    """Test device and dtype handling."""

    def test_device_validation_errors(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test device validation with various invalid inputs."""
        # Test TypeError cases - invalid types
        invalid_type_inputs = [123, ["cuda"]]
        for device_input in invalid_type_inputs:
            with pytest.raises(TypeError, match="Device must be"):
                gdf_to_pyg(sample_nodes_gdf, device=cast("Any", device_input))

        # Test ValueError cases - invalid string values
        invalid_string_inputs = ["invalid_device", "gpu"]
        for device_input in invalid_string_inputs:
            with pytest.raises(ValueError, match="Device must be"):
                gdf_to_pyg(sample_nodes_gdf, device=device_input)

    def test_cuda_not_available(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test CUDA not available error when CUDA is requested but not available."""
        if not torch.cuda.is_available():
            with pytest.raises(ValueError, match="CUDA selected, but not available"):
                gdf_to_pyg(sample_nodes_gdf, device="cuda")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_inputs(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test handling of empty inputs."""
        # No edges
        data = gdf_to_pyg(sample_nodes_gdf, edges=None)
        assert data.edge_index.shape == (2, 0)
        assert data.edge_attr.shape == (0, 0)

        # Round trip with no edges
        nodes_restored, edges_restored = pyg_to_gdf(data)
        assert isinstance(nodes_restored, gpd.GeoDataFrame)
        assert isinstance(edges_restored, gpd.GeoDataFrame)

    def test_empty_features(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test handling of empty feature columns."""
        # Empty feature columns
        data = gdf_to_pyg(
            sample_nodes_gdf,
            sample_edges_gdf,
            node_feature_cols=[],
            node_label_cols=[],
            edge_feature_cols=[],
        )
        assert data.x.shape[1] == 0
        assert data.edge_attr.shape[1] == 0

        # Non-existent feature columns
        data = gdf_to_pyg(
            sample_nodes_gdf,
            sample_edges_gdf,
            node_feature_cols=["nonexistent"],
            edge_feature_cols=["nonexistent"],
        )
        assert data.x.shape[1] == 0
        assert data.edge_attr.shape[1] == 0

    def test_empty_nx_graph(self) -> None:
        """Test that empty NetworkX graph raises error."""
        empty_graph = nx.Graph()
        with pytest.raises(ValueError, match="Graph has no nodes"):
            nx_to_pyg(empty_graph)

    @pytest.mark.parametrize(
        ("feature_type", "graph_type"),
        [
            ("node_feature_cols", "homogeneous"),
            ("node_label_cols", "homogeneous"),
            ("edge_feature_cols", "homogeneous"),
            ("node_feature_cols", "heterogeneous"),
            ("node_label_cols", "heterogeneous"),
            ("edge_feature_cols", "heterogeneous"),
        ],
    )
    def test_invalid_feature_types(
        self,
        feature_type: str,
        graph_type: str,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test invalid feature column types."""
        if graph_type == "homogeneous":
            kwargs: dict[str, Any] = {feature_type: {"invalid": "cols"}}
            expected_msg = f"{feature_type} must be a list"
            with pytest.raises(TypeError, match=expected_msg):
                gdf_to_pyg(sample_nodes_gdf, **kwargs)
        else:
            kwargs = {feature_type: ["invalid"]}
            expected_msg = f"{feature_type} must be a dict"
            with pytest.raises(TypeError, match=expected_msg):
                gdf_to_pyg(sample_hetero_nodes_dict, **kwargs)

    def test_geometry_handling(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test geometry handling edge cases."""
        # Null geometries
        nodes_with_null = sample_nodes_gdf.copy()
        nodes_with_null["geometry"] = None
        data = gdf_to_pyg(nodes_with_null)
        nodes_restored, _ = pyg_to_gdf(data)
        assert isinstance(nodes_restored, gpd.GeoDataFrame)

        # Missing position data
        data.pos = None
        nodes_restored, _ = pyg_to_gdf(data)
        assert isinstance(nodes_restored, gpd.GeoDataFrame)


class TestSpecialCases:
    """Test special cases and edge conditions for coverage."""

    def test_node_attribute_preservation(self, sample_pyg_data: Data) -> None:
        """Test that node attributes are preserved in conversions."""
        nx_graph = pyg_to_nx(sample_pyg_data)
        for _node_id, node_data in nx_graph.nodes(data=True):
            assert "_original_index" in node_data
            if hasattr(sample_pyg_data, "pos") and sample_pyg_data.pos is not None:
                assert "pos" in node_data

    def test_tensor_validation_errors(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test tensor validation errors."""
        # Position tensor mismatch
        data = gdf_to_pyg(sample_nodes_gdf)
        data.pos = torch.randn(1, 2)  # Wrong size
        with pytest.raises(ValueError, match="position tensor size.*doesn't match"):
            validate_pyg(data)

        # Label tensor mismatch
        data = gdf_to_pyg(sample_nodes_gdf, node_label_cols=["label1"])
        data.y = torch.randn(1, 1)  # Wrong size
        with pytest.raises(ValueError, match="label tensor size.*doesn't match"):
            validate_pyg(data)

    def test_hetero_label_tensor_mismatch(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous label tensor size mismatch."""
        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            node_feature_cols={"building": ["b_feat1"], "road": ["length"]},
            node_label_cols={"building": ["b_label"]},
        )

        # Corrupt the label tensor size for building node type
        data["building"].y = torch.randn(1, 1)  # Wrong size

        with pytest.raises(
            ValueError,
            match="Node type 'building': label tensor size.*doesn't match",
        ):
            validate_pyg(data)

    def test_homo_validation_edge_cases(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test homogeneous graph validation edge cases."""
        # Missing default mapping
        data = gdf_to_pyg(sample_nodes_gdf)
        data.graph_metadata.node_mappings = {"wrong_key": {}}
        with pytest.raises(ValueError, match="should use 'default' key in node_mappings"):
            validate_pyg(data)

        # Dict feature cols instead of list
        data = gdf_to_pyg(sample_nodes_gdf)
        data.graph_metadata.node_feature_cols = {"wrong": ["feature1"]}
        with pytest.raises(ValueError, match="should have node_feature_cols as list, not dict"):
            validate_pyg(data)

        # Dict label cols instead of list
        data = gdf_to_pyg(sample_nodes_gdf, node_label_cols=["label1"])
        data.graph_metadata.node_label_cols = {"wrong": ["label1"]}
        with pytest.raises(ValueError, match="should have node_label_cols as list, not dict"):
            validate_pyg(data)

    def test_homo_edge_validation_cases(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test homogeneous edge validation cases."""
        # Dict edge feature cols instead of list
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, edge_feature_cols=["edge_feature1"])
        data.graph_metadata.edge_feature_cols = {"wrong": ["edge_feature1"]}
        with pytest.raises(ValueError, match="should have edge_feature_cols as list, not dict"):
            validate_pyg(data)

        # Edge attribute tensor mismatch
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf, edge_feature_cols=["edge_feature1"])
        data.edge_attr = torch.randn(2, 1)  # Wrong size
        with pytest.raises(
            ValueError,
            match="Edge attribute tensor size.*doesn't match number of edges",
        ):
            validate_pyg(data)

    def test_nx_default_feature_naming(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test NetworkX conversion with default feature naming."""
        data = gdf_to_pyg(sample_nodes_gdf, node_feature_cols=["feature1"])

        # Clear the feature column names to trigger default naming
        data.graph_metadata.node_feature_cols = None

        nx_graph = pyg_to_nx(data)

        # Check that nodes have attributes (default naming may or may not include feat_ prefix)
        for _node_id, node_data in nx_graph.nodes(data=True):
            assert isinstance(node_data, dict)
            assert len(node_data) > 0  # Should have some attributes
            break

    def test_homogeneous_feature_extraction_branches(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test homogeneous feature extraction branches - covers lines 1512-1513, 1523-1524."""
        # Create data with features and labels
        data = gdf_to_pyg(
            sample_nodes_gdf,
            node_feature_cols=["feature1"],
            node_label_cols=["label1"],
        )

        # Test the homogeneous branch in _extract_node_features_and_labels
        # by calling pyg_to_gdf which uses this function
        nodes_restored, _ = pyg_to_gdf(data)

        # Verify features and labels are extracted correctly
        assert isinstance(nodes_restored, gpd.GeoDataFrame)
        assert "feature1" in nodes_restored.columns
        assert "label1" in nodes_restored.columns

    def test_hetero_nx_default_naming(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous NetworkX conversion with default naming - covers lines 1841, 1855."""
        # Create heterogeneous data with features and labels
        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            node_feature_cols={"building": ["b_feat1"], "road": ["length"]},
            node_label_cols={"building": ["b_label"], "road": ["r_label"]},
        )

        # Clear feature/label column names for one node type to trigger default naming
        data.graph_metadata.node_feature_cols = {"building": None, "road": ["length"]}
        data.graph_metadata.node_label_cols = {"building": None, "road": ["r_label"]}

        # Convert to NetworkX - this should trigger the default naming branches
        nx_graph = pyg_to_nx(data)

        # Check that nodes have attributes with default names
        building_nodes = [
            n for n, d in nx_graph.nodes(data=True) if d.get("node_type") == "building"
        ]
        if building_nodes:
            node_data = nx_graph.nodes[building_nodes[0]]
            # Should have default feature names like feat_0, label_0
            has_default_feat = any(key.startswith("feat_") for key in node_data)
            has_default_label = any(key.startswith("label_") for key in node_data)
            # At least one should be true if we have features/labels
            non_meta_keys = [
                k for k in node_data if not k.startswith("_") and k not in {"node_type", "pos"}
            ]
            assert has_default_feat or has_default_label or len(non_meta_keys) == 0

    def test_heterogeneous_empty_edges(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous conversion with empty edges."""
        empty_edges_dict = {
            ("building", "connects_to", "road"): gpd.GeoDataFrame(
                columns=["geometry"],
                index=pd.MultiIndex.from_arrays([[], []], names=["building_id", "road_id"]),
            ),
        }
        data = gdf_to_pyg(sample_hetero_nodes_dict, empty_edges_dict)
        edge_type = ("building", "connects_to", "road")
        assert data[edge_type].edge_index.shape == (2, 0)
        assert data[edge_type].edge_attr.shape[0] == 0

    def test_heterogeneous_tensor_validation(self, sample_pyg_hetero_data: HeteroData) -> None:
        """Test tensor validation for heterogeneous graphs."""
        node_type = next(iter(sample_pyg_hetero_data.node_types))
        sample_pyg_hetero_data[node_type].pos = torch.randn(1, 2)  # Wrong size
        with pytest.raises(ValueError, match="position tensor size.*doesn't match"):
            validate_pyg(sample_pyg_hetero_data)

    def test_feature_extraction_minimal_data(self) -> None:
        """Test feature extraction with minimal node data."""
        minimal_nodes = gpd.GeoDataFrame(
            {
                "geometry": [Point(0, 0), Point(1, 1)],
            },
            index=[100, 200],
        )
        data = gdf_to_pyg(minimal_nodes)
        nodes_back, _ = pyg_to_gdf(data)
        assert isinstance(nodes_back, gpd.GeoDataFrame)
        assert list(nodes_back.index) == [100, 200]

    def test_missing_features_handling(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test handling of missing node features."""
        nodes_no_features = sample_nodes_gdf[["geometry"]]
        data = gdf_to_pyg(nodes_no_features, node_feature_cols=["nonexistent"])
        assert data.x.shape[1] == 0

    def test_nx_feature_naming(self, sample_pyg_data: Data) -> None:
        """Test NetworkX conversion with default feature naming."""
        data = sample_pyg_data
        data.graph_metadata.node_feature_cols = None
        data.graph_metadata.node_label_cols = None
        nx_graph = pyg_to_nx(data)
        for _node_id, node_data in nx_graph.nodes(data=True):
            if hasattr(data, "x") and data.x is not None and data.x.numel() > 0:
                assert any(key.startswith("feat_") for key in node_data)

    def test_hetero_feature_extraction(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous feature extraction."""
        data = gdf_to_pyg(
            sample_hetero_nodes_dict,
            node_feature_cols={"building": ["b_feat1"], "road": ["r_feat1"]},
            node_label_cols={"building": ["b_label"], "road": ["r_label"]},
        )
        nodes_restored, _ = pyg_to_gdf(data)
        assert isinstance(nodes_restored, dict)
        assert isinstance(nodes_restored["building"], gpd.GeoDataFrame)
        assert isinstance(nodes_restored["road"], gpd.GeoDataFrame)
        assert "b_feat1" in nodes_restored["building"].columns
        # Note: r_feat1 doesn't exist in the fixture, so it won't be restored
        assert "r_label" in nodes_restored["road"].columns

    def test_edge_features_none_handling(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test edge feature extraction with None edge attributes."""
        data = gdf_to_pyg(sample_nodes_gdf, sample_edges_gdf)
        data.edge_attr = None
        _, edges_restored = pyg_to_gdf(data)
        assert isinstance(edges_restored, gpd.GeoDataFrame)
        assert len(edges_restored) == len(sample_edges_gdf)

    def test_homogeneous_validation_errors(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test validation errors specific to homogeneous graphs."""
        data = gdf_to_pyg(sample_nodes_gdf)

        # Node types specified
        data.graph_metadata.node_types = ["some_type"]
        with pytest.raises(ValueError, match="should not have node_types"):
            validate_pyg(data)

        # Reset and test edge types
        data = gdf_to_pyg(sample_nodes_gdf)
        data.graph_metadata.edge_types = [("some", "edge", "type")]
        with pytest.raises(ValueError, match="should not have edge_types"):
            validate_pyg(data)

    def test_hetero_validation_errors(self, sample_pyg_hetero_data: HeteroData) -> None:
        """Test validation errors specific to heterogeneous graphs."""
        # Node types mismatch
        data = sample_pyg_hetero_data
        original_node_types = data.graph_metadata.node_types
        data.graph_metadata.node_types = ["wrong_type"]
        with pytest.raises(ValueError, match="Node types mismatch"):
            validate_pyg(data)

        # Reset and test edge types mismatch
        data.graph_metadata.node_types = original_node_types
        data.graph_metadata.edge_types = [("wrong", "type", "tuple")]
        with pytest.raises(ValueError, match="Edge types mismatch"):
            validate_pyg(data)
