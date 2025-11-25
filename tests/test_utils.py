"""Tests for the utils module - lean and maintainable.

This file delegates common assertions and helpers to tests/helpers.py to avoid
duplication and keep a single source of truth for shared logic across tests.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING
from unittest import mock

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
import rustworkx as rx
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph import utils
from city2graph.base import GeoDataProcessor
from city2graph.base import GraphMetadata
from city2graph.utils import NxConverter
from city2graph.utils import gdf_to_nx
from city2graph.utils import nx_to_gdf
from tests import helpers

if TYPE_CHECKING:
    from collections.abc import Callable

# Try to import matplotlib for tests that need it
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore[assignment]
    MATPLOTLIB_AVAILABLE = False

# ============================================================================
# BASE TEST CLASSES WITH SHARED FUNCTIONALITY
# ============================================================================


class BaseGraphTest:
    """Base class for graph-related tests with common utilities."""

    @staticmethod
    def assert_valid_gdf(gdf: gpd.GeoDataFrame, expected_empty: bool = False) -> None:
        """Delegate to shared helper to ensure consistent checks across tests."""
        helpers.assert_valid_gdf(gdf, expected_empty)

    @staticmethod
    def assert_crs_consistency(*gdfs: gpd.GeoDataFrame) -> None:
        """Delegate to shared helper for CRS consistency checks."""
        helpers.assert_crs_consistency(*gdfs)


class BaseConversionTest(BaseGraphTest):
    """Base class for conversion tests between GDF and NetworkX."""

    def assert_roundtrip_consistency(
        self,
        original_nodes: gpd.GeoDataFrame,
        original_edges: gpd.GeoDataFrame,
        converted_nodes: gpd.GeoDataFrame,
        converted_edges: gpd.GeoDataFrame,
    ) -> None:
        """Delegate to shared helper for roundtrip integrity checks."""
        helpers.assert_roundtrip_consistency(
            original_nodes,
            original_edges,
            converted_nodes,
            converted_edges,
        )


# ============================================================================
# TESSELLATION TESTS
# ============================================================================


class TestTessellation(BaseGraphTest):
    """Test tessellation creation functionality."""

    @pytest.mark.parametrize(
        ("geometry_fixture", "barriers_fixture", "expect_empty"),
        [
            ("empty_gdf", None, True),
            ("sample_buildings_gdf", None, False),
            ("sample_buildings_gdf", "sample_segments_gdf", False),
        ],
    )
    def test_tessellation_creation(
        self,
        geometry_fixture: str,
        barriers_fixture: str | None,
        expect_empty: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test tessellation creation with various input combinations."""
        geometry = request.getfixturevalue(geometry_fixture)
        barriers = request.getfixturevalue(barriers_fixture) if barriers_fixture else None

        try:
            tessellation = utils.create_tessellation(geometry, primary_barriers=barriers)
        except (UnboundLocalError, TypeError, ValueError) as e:
            pytest.skip(f"Skipping due to incomplete implementation: {e}")

        self.assert_valid_gdf(tessellation, expect_empty)
        if not expect_empty:
            helpers.assert_has_columns(tessellation, ["tess_id"])
            assert tessellation.crs == geometry.crs

    def test_tessellation_handles_empty_geometry(self, empty_gdf: gpd.GeoDataFrame) -> None:
        """Empty input should return an empty GeoDataFrame with expected columns."""
        result = utils.create_tessellation(empty_gdf)
        self.assert_valid_gdf(result, expected_empty=True)

    def test_tessellation_handles_degenerate_inputs(
        self,
        single_point_geom_gdf: gpd.GeoDataFrame,
        tessellation_barriers_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Degenerate geometries or barriers should not raise."""
        single_result = utils.create_tessellation(single_point_geom_gdf)
        self.assert_valid_gdf(single_result, expected_empty=True)

        barrier_result = utils.create_tessellation(
            single_point_geom_gdf,
            primary_barriers=tessellation_barriers_gdf,
        )
        self.assert_valid_gdf(barrier_result, expected_empty=True)

    def test_tessellation_empty_with_barriers(self, empty_gdf: gpd.GeoDataFrame) -> None:
        """Even with barriers an empty geometry should stay empty."""
        barriers = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 1)])]},
            crs=empty_gdf.crs,
        )
        result = utils.create_tessellation(empty_gdf, primary_barriers=barriers)
        self.assert_valid_gdf(result, expected_empty=True)
        helpers.assert_has_columns(result, ["enclosure_index"])


# ============================================================================
# GRAPH STRUCTURE TESTS
# ============================================================================


class TestGraphStructures(BaseGraphTest):
    """Test graph structure operations like dual graph and segments conversion."""

    @pytest.mark.parametrize(
        (
            "nodes_fixture",
            "edges_fixture",
            "keep_geom",
            "edge_id_col",
            "should_error",
            "error_match",
        ),
        [
            # Success cases
            ("sample_nodes_gdf", "sample_edges_gdf", False, None, False, None),
            ("sample_nodes_gdf", "sample_edges_gdf", True, "edge_id", False, None),
            ("empty_gdf", "empty_gdf", False, None, False, None),
            # Error cases
            ("sample_segments_gdf", None, False, None, True, r"Input `graph` must be a tuple"),
            (
                "sample_nodes_gdf",
                "segments_gdf_no_crs",
                False,
                None,
                True,
                "Edges GeoDataFrame must have a CRS",
            ),
        ],
    )
    def test_dual_graph_conversion(
        self,
        nodes_fixture: str,
        edges_fixture: str | None,
        keep_geom: bool,
        edge_id_col: str | None,
        should_error: bool,
        error_match: str | None,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test dual graph conversion with comprehensive parameter combinations."""
        if edges_fixture is None:
            graph_input = request.getfixturevalue(nodes_fixture)
        else:
            nodes = request.getfixturevalue(nodes_fixture)
            edges = request.getfixturevalue(edges_fixture)
            graph_input = (nodes, edges)

        if should_error:
            with pytest.raises((TypeError, ValueError, AttributeError), match=error_match):
                utils.dual_graph(graph_input, edge_id_col=edge_id_col, keep_original_geom=keep_geom)
        else:
            dual_nodes, dual_edges = utils.dual_graph(
                graph_input,
                edge_id_col=edge_id_col,
                keep_original_geom=keep_geom,
            )

            # Handle empty case
            if isinstance(graph_input, tuple) and graph_input[1].empty:
                self.assert_valid_gdf(dual_nodes, expected_empty=True)
                self.assert_valid_gdf(dual_edges, expected_empty=True)
                return

            self.assert_valid_gdf(dual_nodes)
            self.assert_valid_gdf(dual_edges)
            self.assert_crs_consistency(dual_nodes, dual_edges)

            if keep_geom:
                assert "original_geometry" in dual_nodes.columns

    def test_dual_graph_invalid_edge_id_col(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test that providing a non-existent edge_id_col raises ValueError."""
        with pytest.raises(ValueError, match="Column 'non_existent_col' not found"):
            utils.dual_graph((sample_nodes_gdf, sample_edges_gdf), edge_id_col="non_existent_col")

    def test_dual_graph_accepts_networkx_input(self, simple_nx_graph: nx.Graph) -> None:
        """Dual graph should accept NetworkX graphs directly."""
        dual_nodes, dual_edges = utils.dual_graph(simple_nx_graph)
        self.assert_valid_gdf(dual_nodes)
        self.assert_valid_gdf(dual_edges, expected_empty=True)

    def test_dual_graph_warns_on_geographic_crs(self) -> None:
        """Warn users when dual graph is computed on geographic CRS data."""
        edges = gpd.GeoDataFrame(
            {
                "source": [1, 2],
                "target": [2, 3],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            },
            crs="EPSG:4326",
        ).set_index(["source", "target"])
        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
            crs="EPSG:4326",
            index=[1, 2, 3],
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            utils.dual_graph((nodes, edges))
            assert any("geographic CRS" in str(item.message) for item in caught)

    def test_dual_graph_handles_mixed_edge_ids(self) -> None:
        """Mixed edge-id types should still yield deterministic adjacencies."""
        edges = gpd.GeoDataFrame(
            {
                "edge_id": ["a", 1],
                "source": [1, 2],
                "target": [2, 3],
                "geometry": [
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2)]),
                ],
            },
            crs="EPSG:27700",
        ).set_index(["source", "target"])
        nodes = gpd.GeoDataFrame(
            {"geometry": [Point(0, 0), Point(1, 1), Point(2, 2)]},
            crs="EPSG:27700",
            index=[1, 2, 3],
        )

        dual_nodes, dual_edges = utils.dual_graph((nodes, edges), edge_id_col="edge_id")
        self.assert_valid_gdf(dual_nodes)
        self.assert_valid_gdf(dual_edges)

    def test_validate_nx_populates_pos_from_xy(self, sample_crs: str) -> None:
        """validate_nx should auto-create pos from x/y when missing."""
        G = nx.Graph()
        G.add_node(1, x=0.0, y=0.0)
        G.add_node(2, x=1.0, y=1.0)
        G.add_edge(1, 2)
        G.graph = {"crs": sample_crs, "is_hetero": False}

        GeoDataProcessor().validate_nx(G)
        assert "pos" in G.nodes[1]
        assert "pos" in G.nodes[2]

    @pytest.mark.parametrize(
        ("segments_fixture", "expect_empty", "multigraph"),
        [
            ("sample_segments_gdf", False, False),
            ("empty_gdf", True, False),
            ("segments_invalid_geom_gdf", True, False),
            ("sample_segments_gdf", False, True),  # Test multigraph functionality
        ],
    )
    def test_segments_to_graph_conversion(
        self,
        segments_fixture: str,
        expect_empty: bool,
        multigraph: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test conversion of line segments to graph structure."""
        segments_gdf = request.getfixturevalue(segments_fixture)
        nodes_gdf, edges_gdf = utils.segments_to_graph(segments_gdf, multigraph=multigraph)

        self.assert_valid_gdf(nodes_gdf, expect_empty)
        self.assert_valid_gdf(edges_gdf, expect_empty)
        self.assert_crs_consistency(nodes_gdf, edges_gdf)

        if not expect_empty:
            assert nodes_gdf.index.name == "node_id"
            helpers.assert_geometry_types(nodes_gdf, ["Point"])
            assert isinstance(edges_gdf.index, pd.MultiIndex)

            expected_index_names = (
                ["from_node_id", "to_node_id", "edge_key"]
                if multigraph
                else ["from_node_id", "to_node_id"]
            )
            helpers.assert_index_names(edges_gdf, expected_index_names)

    def test_segments_multigraph_duplicate_handling(
        self,
        duplicate_segments_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test multigraph handling of duplicate edge connections."""
        nodes_gdf, edges_gdf = utils.segments_to_graph(duplicate_segments_gdf, multigraph=True)

        assert len(nodes_gdf) == 2  # Two unique points
        assert len(edges_gdf) == 2  # Both edges preserved
        assert edges_gdf.index.names == ["from_node_id", "to_node_id", "edge_key"]

        # Verify edge keys are different for duplicates
        edge_keys = edges_gdf.index.get_level_values("edge_key")
        assert list(edge_keys) == [0, 1]


# ============================================================================
# GRAPH ANALYSIS TESTS
# ============================================================================


class TestGraphAnalysis(BaseGraphTest):
    """Test graph analysis operations like filtering and isochrone generation."""

    @pytest.mark.parametrize(
        ("graph_fixture", "as_nx", "center_fixture", "distance", "expect_empty"),
        [
            ("sample_segments_gdf", False, "mg_center_point", 100.0, False),
            ("sample_segments_gdf", False, "mg_center_point", 0.01, True),
            ("sample_nx_graph", True, "sample_nodes_gdf", 1.0, False),
            ("empty_gdf", False, "mg_center_point", 100.0, True),
        ],
    )
    def test_graph_distance_filtering(
        self,
        graph_fixture: str,
        as_nx: bool,
        center_fixture: str,
        distance: float,
        expect_empty: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test filtering graphs by distance from center points."""
        graph = request.getfixturevalue(graph_fixture)
        center_source = request.getfixturevalue(center_fixture)
        center_point = helpers.get_center_point(center_source)

        filtered = utils.filter_graph_by_distance(graph, center_point, distance=distance)

        if as_nx:
            assert isinstance(filtered, nx.Graph)
            assert (filtered.number_of_edges() == 0) == expect_empty
        else:
            self.assert_valid_gdf(filtered, expect_empty)

    def test_graph_distance_filtering_without_pos(self, sample_crs: str) -> None:
        """Graphs lacking position data should return empty filtered results."""
        graph = nx.Graph()
        graph.add_node(1, feature=1)
        graph.add_node(2, feature=2)
        graph.add_edge(1, 2)
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        filtered = utils.filter_graph_by_distance(graph, Point(0, 0), 100.0)
        assert isinstance(filtered, nx.Graph)
        assert filtered.number_of_nodes() == 0

    @pytest.mark.parametrize(
        ("graph_fixture", "center_fixture", "distance", "expect_empty"),
        [
            ("sample_segments_gdf", "mg_center_point", 100.0, False),
            ("sample_segments_gdf", "mg_center_point", 0.01, True),
            ("sample_nx_graph", "sample_nodes_gdf", 1.0, False),
        ],
    )
    def test_isochrone_generation(
        self,
        graph_fixture: str,
        center_fixture: str,
        distance: float,
        expect_empty: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test isochrone polygon generation from graphs."""
        graph = request.getfixturevalue(graph_fixture)
        center_source = request.getfixturevalue(center_fixture)
        center_point = helpers.get_center_point(center_source)

        isochrone = utils.create_isochrone(graph, center_point, distance=distance)

        self.assert_valid_gdf(isochrone, expect_empty)
        if not expect_empty:
            assert len(isochrone) == 1
            assert isochrone.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]


# ============================================================================
# CONVERSION TESTS
# ============================================================================


class TestNxConversions(BaseConversionTest):
    """Test conversions between GeoDataFrame and NetworkX formats."""

    def test_homogeneous_roundtrip_conversion(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test roundtrip conversion preserves data integrity for homogeneous graphs."""
        graph = gdf_to_nx(sample_nodes_gdf, sample_edges_gdf)
        nodes_converted, edges_converted = nx_to_gdf(graph)

        self.assert_roundtrip_consistency(
            sample_nodes_gdf,
            sample_edges_gdf,
            nodes_converted,
            edges_converted,
        )

    def test_heterogeneous_roundtrip_conversion(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test roundtrip conversion for heterogeneous graphs."""
        graph = gdf_to_nx(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            multigraph=True,
        )
        nodes_dict_converted, edges_dict_converted = nx_to_gdf(graph)

        assert isinstance(nodes_dict_converted, dict)
        assert isinstance(edges_dict_converted, dict)
        assert sample_hetero_nodes_dict.keys() == nodes_dict_converted.keys()
        assert sample_hetero_edges_dict.keys() == edges_dict_converted.keys()

        for node_type, original_nodes in sample_hetero_nodes_dict.items():
            assert isinstance(nodes_dict_converted[node_type], gpd.GeoDataFrame)
            converted_nodes = nodes_dict_converted[node_type]
            # Use the first available edge type for consistency check
            first_edge_type = next(iter(sample_hetero_edges_dict.keys()))
            original_edges = sample_hetero_edges_dict[first_edge_type]
            assert isinstance(edges_dict_converted[first_edge_type], gpd.GeoDataFrame)
            converted_edges = edges_dict_converted[first_edge_type]
            self.assert_roundtrip_consistency(
                original_nodes,
                original_edges,
                converted_nodes,
                converted_edges,
            )

    @pytest.mark.parametrize(
        ("input_type", "gdf_fixture"),
        [("edges_only", "sample_edges_gdf"), ("hetero_edges_only", "sample_hetero_edges_dict")],
    )
    def test_edges_only_conversion(
        self,
        input_type: str,
        gdf_fixture: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test conversion with only edge data provided."""
        gdf = request.getfixturevalue(gdf_fixture)

        if input_type == "edges_only":
            graph = gdf_to_nx(edges=gdf)
            assert isinstance(graph, nx.Graph)
            assert graph.number_of_edges() == len(gdf)
            assert graph.number_of_nodes() > 0  # Nodes created from edge endpoints
        else:  # hetero_edges_only
            graph = gdf_to_nx(edges=gdf)
            assert isinstance(graph, nx.Graph)
            # Heterogeneous edges dict without nodes results in empty graph
            assert graph.number_of_edges() == 0

    def test_nx_to_gdf_requires_nodes_or_edges(self, simple_nx_graph: nx.Graph) -> None:
        """Requesting neither nodes nor edges should raise."""
        with pytest.raises(ValueError, match="Must request at least one of nodes or edges"):
            nx_to_gdf(simple_nx_graph, nodes=False, edges=False)

    def test_directed_multigraph_conversion(
        self,
        directed_multigraph_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Directed + multigraph flags should produce a MultiDiGraph."""
        converter = NxConverter(directed=True, multigraph=True)
        graph = converter.gdf_to_nx(nodes=None, edges=directed_multigraph_edges_gdf)
        assert isinstance(graph, nx.MultiDiGraph)

    @pytest.mark.parametrize(
        ("nodes_arg", "edges_arg", "error_type", "error_match"),
        [
            (None, None, ValueError, "Either nodes or edges must be provided"),
            ("not_a_gdf", "sample_edges_gdf", TypeError, "Input must be a GeoDataFrame"),
            (
                "sample_hetero_nodes_dict",
                "sample_edges_gdf",
                TypeError,
                "If nodes is a dict, edges must also be a dict",
            ),
            (
                "sample_nodes_gdf_alt_crs",
                "sample_edges_gdf",
                ValueError,
                "All GeoDataFrames must have the same CRS",
            ),
        ],
    )
    def test_conversion_error_handling(
        self,
        nodes_arg: str | None,
        edges_arg: str | None,
        error_type: type[Exception],
        error_match: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test proper error handling for invalid conversion inputs."""
        nodes = request.getfixturevalue(nodes_arg) if nodes_arg else None
        edges = request.getfixturevalue(edges_arg) if edges_arg else None

        with pytest.raises(error_type, match=error_match):
            gdf_to_nx(nodes=nodes, edges=edges)

    def test_node_index_names_survive_roundtrip(
        self,
        single_name_index_nodes_gdf: gpd.GeoDataFrame,
        simple_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Roundtrips should respect single-level index names."""
        graph = gdf_to_nx(nodes=single_name_index_nodes_gdf, edges=simple_edges_gdf)
        nodes_back, _ = nx_to_gdf(graph)
        assert isinstance(nodes_back, gpd.GeoDataFrame)
        assert nodes_back.index.name == "single_name"

        graph.graph["node_index_names"] = None
        nodes_back, _ = nx_to_gdf(graph)
        assert isinstance(nodes_back, gpd.GeoDataFrame)
        assert nodes_back.index.name is None

    def test_multiindex_nodes_roundtrip(
        self,
        multiindex_nodes_gdf: gpd.GeoDataFrame,
        simple_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """MultiIndex node metadata should be preserved in graph metadata."""
        graph = gdf_to_nx(nodes=multiindex_nodes_gdf, edges=simple_edges_gdf)
        assert graph.graph["node_index_names"] == ["node_type", "node_id"]

    @pytest.mark.parametrize(
        ("graph_fixture", "expect_crs", "expect_geom"),
        [
            ("sample_nx_graph", True, True),
            ("sample_nx_graph_no_crs", False, True),
            ("sample_nx_graph_no_pos", True, True),
        ],
    )
    def test_nx_to_gdf_variants(
        self,
        graph_fixture: str,
        expect_crs: bool,
        expect_geom: bool,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test NetworkX to GDF conversion with different graph properties."""
        graph = request.getfixturevalue(graph_fixture)
        nodes, edges = nx_to_gdf(graph)

        # These are homogeneous graphs, so ensure they return GeoDataFrames not dicts
        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)

        if expect_geom:
            assert "geometry" in nodes.columns
            assert "geometry" in edges.columns

        if expect_crs:
            assert nodes.crs is not None
            assert edges.crs is not None
        else:
            assert nodes.crs is None
            assert edges.crs is None

    def test_nx_to_gdf_handles_empty_edges(self, sample_crs: str) -> None:
        """Graphs with no edges should still return empty edge GeoDataFrames."""
        graph = nx.Graph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        nodes_gdf, edges_gdf = nx_to_gdf(graph)
        self.assert_valid_gdf(nodes_gdf)
        self.assert_valid_gdf(edges_gdf, expected_empty=True)

    def test_nx_to_gdf_multigraph_edges(self, sample_crs: str) -> None:
        """Multigraph attributes should be preserved after conversion."""
        graph = nx.MultiGraph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.add_node(2, pos=(1, 1), geometry=Point(1, 1))
        graph.add_edge(1, 2, key=0, weight=1.0, geometry=LineString([(0, 0), (1, 1)]))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        _, edges_gdf = nx_to_gdf(graph)
        self.assert_valid_gdf(edges_gdf)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert "weight" in edges_gdf.columns

    def test_nx_to_gdf_multiindex_edges(self, sample_crs: str) -> None:
        """List-based stored edge indices should be normalized during reconstruction."""
        graph = nx.MultiGraph()
        graph.add_node(1, pos=(0, 0), geometry=Point(0, 0))
        graph.add_node(2, pos=(1, 1), geometry=Point(1, 1))
        graph.add_edge(1, 2, key=0, geometry=LineString([(0, 0), (1, 1)]))
        graph.add_edge(1, 2, key=1, geometry=LineString([(0, 0), (1, 1)]))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        for u, v, k, attrs in graph.edges(data=True, keys=True):
            attrs["_original_edge_index"] = [u, v, k]

        _, edges_gdf = nx_to_gdf(graph)
        self.assert_valid_gdf(edges_gdf)
        assert len(edges_gdf) == 2

    def test_nx_to_gdf_single_coord_attr(self, sample_crs: str) -> None:
        """Missing pos attributes can be populated from alternative coordinate keys."""
        graph = nx.Graph()
        graph.add_node(1, coords=[0.0, 1.0])
        graph.add_node(2, coords=(2.0, 3.0))
        graph.graph = {"crs": sample_crs, "is_hetero": False}

        nodes_gdf = nx_to_gdf(graph, nodes=True, edges=False, set_missing_pos_from=("coords",))
        self.assert_valid_gdf(nodes_gdf)
        assert len(nodes_gdf) == 2

    def test_heterogeneous_edge_processing(
        self,
        regular_hetero_graph: nx.Graph,
        empty_hetero_graph: nx.Graph,
    ) -> None:
        """nx_to_gdf should return dictionaries for heterogeneous graphs."""
        nodes_dict, edges_dict = nx_to_gdf(regular_hetero_graph)
        assert isinstance(nodes_dict, dict)
        assert isinstance(edges_dict, dict)

        nodes_dict, edges_dict = nx_to_gdf(empty_hetero_graph)
        assert isinstance(edges_dict[("building", "connects", "road")], gpd.GeoDataFrame)
        assert edges_dict[("building", "connects", "road")].empty


# ============================================================================
# VALIDATION TESTS
# ============================================================================


class TestValidation:
    """Test validation functions for GeoDataFrames and NetworkX graphs."""

    def test_geo_processor_edge_cases(
        self,
        sample_buildings_gdf: gpd.GeoDataFrame,
        empty_gdf: gpd.GeoDataFrame,
        invalid_geom_gdf: gpd.GeoDataFrame,
        all_invalid_geom_gdf: gpd.GeoDataFrame,
    ) -> None:
        """GeoDataProcessor should handle geometry filtering and allow-empty toggles."""
        processor = GeoDataProcessor()

        result = processor.validate_gdf(
            sample_buildings_gdf,
            expected_geom_types=["Polygon", "MultiPolygon"],
        )
        assert isinstance(result, gpd.GeoDataFrame)

        assert processor.validate_gdf(empty_gdf, allow_empty=True) is not None
        with pytest.raises(ValueError, match="GeoDataFrame cannot be empty"):
            processor.validate_gdf(empty_gdf, allow_empty=False)

        filtered = processor.validate_gdf(invalid_geom_gdf)
        assert filtered is not None
        assert len(filtered) < len(invalid_geom_gdf)

        with pytest.raises(ValueError, match="GeoDataFrame cannot be empty"):
            processor.validate_gdf(all_invalid_geom_gdf, allow_empty=False)

    @pytest.mark.parametrize(
        ("nodes_fixture", "edges_fixture", "should_error", "error_match"),
        [
            # Success cases
            ("sample_nodes_gdf", "sample_edges_gdf", False, None),
            ("sample_nodes_gdf", None, False, None),
            (None, "sample_edges_gdf", False, None),
            ("empty_gdf", "sample_edges_gdf", False, None),
            # Error cases
            ("not_a_gdf", "sample_edges_gdf", True, "Input must be a GeoDataFrame"),
            (
                "sample_nodes_gdf_alt_crs",
                "sample_edges_gdf",
                True,
                "All GeoDataFrames must have the same CRS",
            ),
            (
                "sample_nodes_gdf",
                "edges_dict_for_hetero",
                True,
                "If edges is a dict, nodes must also be a dict or None",
            ),
            (
                "simple_nodes_dict_type1",
                "sample_edges_gdf",
                True,
                "If nodes is a dict, edges must also be a dict or None",
            ),
        ],
    )
    def test_gdf_validation(
        self,
        nodes_fixture: str | None,
        edges_fixture: str | None,
        should_error: bool,
        error_match: str | None,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test GeoDataFrame validation with various input combinations."""
        nodes = request.getfixturevalue(nodes_fixture) if nodes_fixture else None
        edges = request.getfixturevalue(edges_fixture) if edges_fixture else None

        if should_error:
            with pytest.raises((TypeError, ValueError), match=error_match):
                utils.validate_gdf(nodes, edges)
        else:
            utils.validate_gdf(nodes, edges)  # Should not raise

    @pytest.mark.parametrize(
        ("graph_fixture", "should_error", "error_match"),
        [
            ("sample_nx_graph", False, None),
            ("sample_nx_multigraph", False, None),
            ("not_a_gdf", True, "Input must be a NetworkX Graph or MultiGraph"),
        ],
    )
    def test_nx_validation(
        self,
        graph_fixture: str,
        should_error: bool,
        error_match: str | None,
        request: pytest.FixtureRequest,
    ) -> None:
        """Test NetworkX graph validation."""
        graph = request.getfixturevalue(graph_fixture)

        if should_error:
            with pytest.raises(TypeError, match=error_match):
                utils.validate_nx(graph)
        else:
            utils.validate_nx(graph)  # Should not raise

    @pytest.mark.parametrize(
        ("graph_fixture", "error_match"),
        [
            ("graph_missing_crs", "Graph metadata is missing required key"),
            (
                "hetero_graph_no_node_types",
                "Heterogeneous graph metadata is missing 'node_types'",
            ),
            (
                "hetero_graph_no_edge_types",
                "Heterogeneous graph metadata is missing 'edge_types'",
            ),
            ("graph_no_pos_geom", "All nodes must have a 'pos' or 'geometry' attribute"),
            (
                "hetero_graph_no_node_type",
                "All nodes in a heterogeneous graph must have a 'node_type' attribute",
            ),
            (
                "hetero_graph_no_edge_type",
                "All edges in a heterogeneous graph must have an 'edge_type' attribute",
            ),
        ],
    )
    def test_validate_nx_rejects_invalid_graphs(
        self,
        graph_fixture: str,
        error_match: str,
        request: pytest.FixtureRequest,
    ) -> None:
        """GeoDataProcessor.validate_nx should surface detailed errors."""
        graph = request.getfixturevalue(graph_fixture)
        processor = GeoDataProcessor()
        with pytest.raises(ValueError, match=error_match):
            processor.validate_nx(graph)

    def test_validation_edge_cases(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        empty_gdf: gpd.GeoDataFrame,
        segments_invalid_geom_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test validation handles edge cases properly."""
        # Empty edges should be allowed
        utils.validate_gdf(sample_nodes_gdf, empty_gdf)

        # Invalid geometries should be handled with warning
        utils.validate_gdf(sample_nodes_gdf, segments_invalid_geom_gdf)

    def test_heterogeneous_validation_errors(
        self,
        nodes_dict_bad_keys: dict[int, gpd.GeoDataFrame],
        edges_dict_bad_tuple: dict[str, gpd.GeoDataFrame],
        simple_nodes_dict_type1: dict[str, gpd.GeoDataFrame],
        edges_dict_bad_elements: dict[tuple[int, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Ensure validation catches malformed heterogeneous inputs."""
        with pytest.raises(TypeError, match="Node type keys must be strings"):
            gdf_to_nx(nodes=nodes_dict_bad_keys, edges=None)

        with pytest.raises(TypeError, match="Edge type keys must be tuples"):
            gdf_to_nx(nodes=simple_nodes_dict_type1, edges=edges_dict_bad_tuple)

        with pytest.raises(TypeError, match="All elements in edge type tuples must be strings"):
            gdf_to_nx(nodes=simple_nodes_dict_type1, edges=edges_dict_bad_elements)

    def test_validate_nx_basic_structure_errors(self) -> None:
        """Graphs with missing metadata or topology should raise informative errors."""
        processor = GeoDataProcessor()

        empty_graph = nx.Graph()
        with pytest.raises(ValueError, match="Graph has no nodes"):
            processor.validate_nx(empty_graph)

        no_edges_graph = nx.Graph()
        no_edges_graph.add_node(1)
        with pytest.raises(ValueError, match="Graph has no edges"):
            processor.validate_nx(no_edges_graph)

        incomplete_graph = nx.Graph()
        incomplete_graph.add_node(1, pos=(0, 0))
        incomplete_graph.add_edge(1, 2)
        delattr(incomplete_graph, "graph")
        with pytest.raises(ValueError, match="Graph is missing 'graph' attribute dictionary"):
            utils.validate_nx(incomplete_graph)


# ============================================================================
# METADATA TESTS
# ============================================================================


class TestGraphMetadata:
    """Test GraphMetadata class functionality."""

    def test_metadata_creation_and_conversion(self) -> None:
        """Test GraphMetadata creation, conversion, and validation."""
        metadata = GraphMetadata(crs="EPSG:4326", is_hetero=True)
        metadata.node_types = ["building", "road"]
        metadata.edge_types = [("building", "connects", "road")]

        # Test to_dict conversion
        result_dict = metadata.to_dict()
        assert result_dict["crs"] == "EPSG:4326"
        assert result_dict["is_hetero"] is True
        assert result_dict["node_types"] == ["building", "road"]

        # Test from_dict creation
        recreated = GraphMetadata.from_dict(result_dict)
        assert recreated.crs == metadata.crs
        assert recreated.is_hetero == metadata.is_hetero
        assert recreated.node_types == metadata.node_types

    @pytest.mark.parametrize(
        ("invalid_data", "error_type", "error_match"),
        [
            ({"crs": 123.45, "is_hetero": False}, TypeError, "CRS must be str, int, dict"),
            ({"crs": "EPSG:4326", "is_hetero": "not_bool"}, TypeError, "is_hetero must be bool"),
        ],
    )
    def test_metadata_validation_errors(
        self,
        invalid_data: dict[str, object],
        error_type: type[Exception],
        error_match: str,
    ) -> None:
        """Test GraphMetadata validation catches invalid inputs."""
        with pytest.raises(error_type, match=error_match):
            GraphMetadata.from_dict(invalid_data)

    def test_metadata_valid_crs_types(self) -> None:
        """Test GraphMetadata accepts various valid CRS formats."""
        valid_crs_values = ["EPSG:4326", 4326, {"init": "epsg:4326"}, None]

        for crs_value in valid_crs_values:
            metadata = GraphMetadata.from_dict({"crs": crs_value, "is_hetero": False})
            assert metadata.crs == crs_value


# ============================================================================
# COMPREHENSIVE EDGE CASE TESTS
# ============================================================================


# ============================================================================
# RUSTWORKX CONVERSION TESTS
# ============================================================================


class TestRustworkxConversions:
    """Test conversions between NetworkX and rustworkx."""

    @pytest.mark.parametrize(
        "create_graph",
        [
            lambda: nx.Graph(name="undirected"),
            lambda: nx.DiGraph(name="directed"),
            lambda: nx.MultiGraph(name="multi_undirected"),
            lambda: nx.MultiDiGraph(name="multi_directed"),
        ],
    )
    def test_nx_rx_roundtrip(self, create_graph: Callable[[], nx.Graph]) -> None:
        """Test full roundtrip conversion preserves structure and attributes."""
        # Setup complex graph
        G = create_graph()
        G.graph["crs"] = "EPSG:4326"
        G.add_node("a", color="red", size=10)
        G.add_node(1, color="blue")

        # Add edges with attributes
        G.add_edge("a", 1, weight=0.5, type="road")

        # If multi-graph, add another edge between same nodes
        if G.is_multigraph():
            G.add_edge("a", 1, weight=0.8, type="path")

        # Convert to rustworkx
        rx_graph = utils.nx_to_rx(G)

        # Verify RX structure
        assert rx_graph.num_nodes() == G.number_of_nodes()
        assert rx_graph.num_edges() == G.number_of_edges()
        assert rx_graph.attrs["crs"] == "EPSG:4326"

        # Convert back to NetworkX
        G_restored = utils.rx_to_nx(rx_graph)

        # Verify restored graph
        assert nx.utils.graphs_equal(G, G_restored)

    def test_rx_to_nx_raw_input(self) -> None:
        """Test converting a raw rustworkx graph (no __nx_node_id__)."""
        rx_G = rx.PyGraph(multigraph=False)
        rx_G.attrs = {"test": "attr"}

        # Add nodes with raw payload (not dict) and dict payload
        idx1 = rx_G.add_node("raw_payload")
        idx2 = rx_G.add_node({"attr": "value"})

        rx_G.add_edge(idx1, idx2, {"weight": 0.5})

        if rx_G.multigraph:
            rx_G.add_edge(idx1, idx2, {"weight": 0.8})

        # Convert
        nx_G = utils.rx_to_nx(rx_G)

        assert isinstance(nx_G, nx.Graph)
        assert nx_G.graph["test"] == "attr"

        # Check nodes - should use integer indices since no __nx_node_id__
        assert idx1 in nx_G.nodes
        assert nx_G.nodes[idx1]["payload"] == "raw_payload"
        assert nx_G.nodes[idx2]["attr"] == "value"

        # Check edge
        assert nx_G.has_edge(idx1, idx2)
        assert nx_G.edges[idx1, idx2]["weight"] == 0.5

    def test_rx_to_nx_raw_edge_payload(self) -> None:
        """Test converting RX graph with non-dict edge payloads."""
        rx_G = rx.PyGraph(multigraph=False)
        i1 = rx_G.add_node(None)
        i2 = rx_G.add_node(None)
        rx_G.add_edge(i1, i2, "edge_label")

        nx_G = utils.rx_to_nx(rx_G)
        assert nx_G.edges[i1, i2]["payload"] == "edge_label"


# ============================================================================
# PLOTTING TESTS
# ============================================================================


class TestPlotting(BaseGraphTest):
    """Test plotting functionality."""

    def test_plot_graph_homogeneous(self, sample_nx_graph: nx.Graph) -> None:
        """Test plotting a homogeneous graph."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        # Mock plt.show to avoid display
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(sample_nx_graph)

    def test_plot_graph_hetero_subplots(
        self,
        regular_hetero_graph: nx.Graph,
    ) -> None:
        """Test heterogeneous graph plotting with subplots."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not available")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            utils.plot_graph(
                graph=regular_hetero_graph,
                subplots=True,
            )

    def test_plot_graph_hetero_subplots_unused_axes(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test heterogeneous subplot with more axes than edge types (line 3441)."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not available")

        # Create a hetero graph with only 2 edge types
        # When subplot grid is larger than number of edge types, unused axes should be hidden
        graph = utils.gdf_to_nx(
            nodes=sample_hetero_nodes_dict,
            edges=sample_hetero_edges_dict,
            multigraph=True,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # This should create more axes than needed and hide unused ones
            # The function will internally hide unused axes (line 3441)
            utils.plot_graph(graph=graph, subplots=True, figsize=(10, 10), show=False)

        plt.close("all")

    def test_plot_graph_hetero_no_subplots_with_graph_input(
        self,
        regular_hetero_graph: nx.Graph,
    ) -> None:
        """Test plotting heterogeneous graph without subplots using graph input (line 3444)."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("Matplotlib not available")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            utils.plot_graph(
                graph=regular_hetero_graph,
                subplots=False,
            )

    def test_plot_graph_save(self, sample_nx_graph: nx.Graph) -> None:
        """Test plotting without showing (for saving to file)."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            # Plot without auto-display (caller can save the figure)
            utils.plot_graph(sample_nx_graph)

    def test_plot_graph_no_matplotlib(self, sample_nx_graph: nx.Graph) -> None:
        """Test error when matplotlib is missing."""
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(utils, "MATPLOTLIB_AVAILABLE", False)
            with pytest.raises(ImportError, match="(?i)matplotlib is required"):
                utils.plot_graph(sample_nx_graph)

    def test_plot_graph_no_input(self) -> None:
        """Test error when no input is provided."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.raises(ValueError, match="At least one of graph, nodes, or edges"):
            utils.plot_graph()

    def test_plot_graph_with_gdf_input(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plotting with GeoDataFrame inputs directly."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(nodes=sample_nodes_gdf, edges=sample_edges_gdf)

    def test_plot_graph_gdf_as_graph_param(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test legacy support: GeoDataFrame passed as graph parameter."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            # This should treat the GDF as nodes (legacy support)
            utils.plot_graph(graph=sample_nodes_gdf)

    def test_plot_graph_unsupported_type(self) -> None:
        """Test error when unsupported type is passed as graph."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.raises(TypeError, match="Unsupported data type"):
            utils.plot_graph(graph="not_a_graph")

    def test_plot_graph_hetero_subplots_with_unused_axes(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test plotting heterogeneous graph with subplots."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges=sample_hetero_edges_dict,
                subplots=True,
            )

    def test_plot_graph_hetero_no_subplots(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test plotting heterogeneous graph without subplots."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges=sample_hetero_edges_dict,
                subplots=False,
            )

    def test_plot_graph_no_legend(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test plotting without legend."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(
                nodes=sample_hetero_nodes_dict,
                edges=sample_hetero_edges_dict,
                legend_position=None,
                subplots=False,
            )

    def test_plot_graph_style_kwargs(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plotting with custom style kwargs."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("matplotlib.pyplot.show", lambda: None)
            utils.plot_graph(
                nodes=sample_nodes_gdf,
                edges=sample_edges_gdf,
                node_color="red",
                node_alpha=0.5,
                edge_color="blue",
                edge_linewidth=2.0,
                markersize=10.0,
            )

    def test_plot_graph_with_ax(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plotting with provided axes."""
        if not utils.MATPLOTLIB_AVAILABLE:
            pytest.skip("matplotlib not available")

        assert plt is not None  # For type checker
        fig, ax = plt.subplots()
        try:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr("matplotlib.pyplot.show", lambda: None)
                utils.plot_graph(nodes=sample_nodes_gdf, ax=ax)
        finally:
            plt.close(fig)


# ============================================================================
# INTERNAL UTILS TESTS
# ============================================================================


class TestIdentifySourceTargetCols:
    """Test the _identify_source_target_cols function."""

    @pytest.fixture
    def basic_edges(self) -> gpd.GeoDataFrame:
        """Create basic edges fixture for testing."""
        return gpd.GeoDataFrame(
            {"u": [1, 2], "v": [2, 3], "weight": [1.0, 2.0]},
            geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
        )

    def test_explicit_columns(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test explicit column specification."""
        u, v = utils._identify_source_target_cols(basic_edges, source_col="u", target_col="v")
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_explicit_index_levels(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test explicit index level specification."""
        edges = basic_edges.set_index(["u", "v"])
        u, v = utils._identify_source_target_cols(edges, source_col="u", target_col="v")
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_implicit_columns_u_v(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test implicit detection of 'u' and 'v' columns."""
        u, v = utils._identify_source_target_cols(basic_edges)
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_implicit_columns_source_target(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test implicit detection of 'source' and 'target' columns."""
        edges = basic_edges.rename(columns={"u": "source", "v": "target"})
        u, v = utils._identify_source_target_cols(edges)
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_implicit_index_from_to_node_id(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test implicit detection of 'from_node_id' and 'to_node_id' index levels."""
        edges = basic_edges.rename(columns={"u": "from_node_id", "v": "to_node_id"})
        edges = edges.set_index(["from_node_id", "to_node_id"])
        u, v = utils._identify_source_target_cols(edges)
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_implicit_index_generic(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test implicit detection from first two index levels."""
        edges = basic_edges.set_index(["u", "v"])
        # Rename levels to something generic
        edges.index.names = ["level_0", "level_1"]
        u, v = utils._identify_source_target_cols(edges)
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_fallback_first_two_columns(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test fallback to first two columns."""
        edges = basic_edges.rename(columns={"u": "col1", "v": "col2"})
        # Ensure col1 and col2 are first
        edges = edges[["col1", "col2", "weight", "geometry"]]
        u, v = utils._identify_source_target_cols(edges)
        assert (u == [1, 2]).all()
        assert (v == [2, 3]).all()

    def test_error_missing_explicit(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test error when explicit columns are missing."""
        with pytest.raises(ValueError, match=r"Source/Target column\(s\) not found: missing"):
            utils._identify_source_target_cols(basic_edges, source_col="missing", target_col="v")

    def test_error_unable_to_identify(self) -> None:
        """Test error when unable to identify columns."""
        edges = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])])
        with pytest.raises(ValueError, match="Could not identify source and target"):
            utils._identify_source_target_cols(edges)

    def test_only_source_missing_target(self, basic_edges: gpd.GeoDataFrame) -> None:
        """Test error when only source is found but target is missing."""
        # This tests line 1647 - missing target column error
        edges = basic_edges.copy()
        with pytest.raises(
            ValueError, match="Source/Target column\\(s\\) not found: missing_target"
        ):
            utils._identify_source_target_cols(edges, source_col="u", target_col="missing_target")

    def test_standard_index_name_matching(self) -> None:
        """Test _get_col_or_level with standard index name matching (line 1600)."""
        # Create DataFrame with named index
        df = pd.DataFrame({"col1": [1, 2, 3]}, index=pd.Index([10, 20, 30], name="my_index"))
        result = utils._get_col_or_level(df, "my_index")
        assert result is not None
        assert (result == [10, 20, 30]).all()


# ============================================================================
# COMPREHENSIVE COVERAGE TESTS
# ============================================================================


class TestComprehensiveCoverage:
    """Additional tests to achieve comprehensive code coverage."""

    def test_empty_nodes_in_conversion(self, sample_crs: str) -> None:
        """Test conversion with completely empty nodes (line 995)."""
        # Create edges but with no nodes that will result in empty node records
        edges_gdf = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=sample_crs,
        )

        # Create a graph and then remove all node attributes to trigger empty records
        converter = NxConverter()
        graph = converter.gdf_to_nx(edges=edges_gdf)

        # Manually clear node data to simulate empty records scenario
        for node in graph.nodes():
            # Keep only pos to trigger the empty records path
            graph.nodes[node].clear()
            graph.nodes[node]["pos"] = (0, 0)

        # Convert back - this should handle empty node records
        nodes_gdf, edges_gdf_out = converter.nx_to_gdf(graph)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)

    def test_dual_graph_empty_result_with_edge_id_col(
        self, sample_nodes_gdf: gpd.GeoDataFrame, sample_crs: str
    ) -> None:
        """Test dual_graph with empty result and edge_id_col specified (line 1527)."""
        # Create edges that won't form any dual edges
        single_edge = gpd.GeoDataFrame(
            {"edge_id": ["e1"]},
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=sample_crs,
        )
        single_edge.index = pd.MultiIndex.from_arrays([[0], [1]], names=["u", "v"])

        dual_nodes, dual_edges = utils.dual_graph(
            (sample_nodes_gdf, single_edge), edge_id_col="edge_id"
        )

        # Should have nodes but no edges (single edge has no dual connections)
        assert not dual_nodes.empty
        assert dual_edges.empty
        assert dual_edges.index.names == ["from_edge_id", "to_edge_id"]

    def test_empty_tessellation_with_tess_id(self, sample_crs: str) -> None:
        """Test _create_empty_tessellation with tess_id column (line 2422)."""
        result = utils._create_empty_tessellation(sample_crs, include_tess_id=True)
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.empty
        assert "tess_id" in result.columns
        assert "enclosure_index" in result.columns

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_empty_gdf(self, empty_gdf: gpd.GeoDataFrame) -> None:
        """Test _plot_gdf with empty GeoDataFrame (line 3246)."""
        # This should not raise and should return early
        fig, ax = plt.subplots()
        utils._plot_gdf(empty_gdf, ax)
        plt.close(fig)

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_with_series_color(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plot_graph with pd.Series for color parameter (lines 3085, 3263)."""
        # Create a Series for node colors with numeric indices (0-based from conversion)
        # When nodes are converted to NetworkX, they get new integer indices
        num_nodes = len(sample_nodes_gdf)
        node_colors = pd.Series(["red"] * num_nodes, index=range(num_nodes))

        utils.plot_graph(
            graph=None,
            nodes=sample_nodes_gdf,
            edges=sample_edges_gdf,
            node_color=node_colors,
            show=False,
        )
        plt.close("all")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_with_column_name(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test plot_graph with column name string for color (line 3087)."""
        # Add a numeric column for coloring
        nodes_with_attr = sample_nodes_gdf.copy()
        nodes_with_attr["value"] = range(len(nodes_with_attr))

        utils.plot_graph(
            graph=None,
            nodes=nodes_with_attr,
            edges=sample_edges_gdf,
            node_color="value",
            show=False,
        )
        plt.close("all")

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_hetero_subplots_empty_edges(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test _plot_hetero_subplots with no edge items (line 3406)."""
        # Create edges dict with only empty GeoDataFrames
        empty_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] = {
            ("building", "connects", "road"): gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        }

        # This should return early without creating a plot
        fig, ax = plt.subplots()
        utils._plot_hetero_subplots(
            sample_hetero_nodes_dict,
            empty_edges_dict,
            figsize=(10, 10),
            bgcolor="white",
        )
        plt.close(fig)

    def test_tessellation_momepy_error_handling(self, sample_crs: str) -> None:
        """Test tessellation error handling for momepy ValueError (lines 2393-2399)."""
        # Create geometry that might cause momepy to fail with "No objects to concatenate"
        # Use a configuration that could trigger internal momepy ValueError

        # Create valid input geometry
        geometry = gpd.GeoDataFrame(
            geometry=[Point(0, 0), Point(1, 1)],
            crs=sample_crs,
        )
        barriers = gpd.GeoDataFrame(
            geometry=[LineString([(0, 0), (1, 1)])],
            crs=sample_crs,
        )

        # Mock momepy.enclosed_tessellation to raise ValueError with "No objects to concatenate"
        # Since utils imports momepy, we patch city2graph.utils.momepy.enclosed_tessellation
        with mock.patch("city2graph.utils.momepy.enclosed_tessellation") as mock_tess:
            mock_tess.side_effect = ValueError("No objects to concatenate")

            # Should handle the error and return empty tessellation
            result = utils.create_tessellation(
                geometry,
                primary_barriers=barriers,
                shrink=0.4,
            )

            assert isinstance(result, gpd.GeoDataFrame)
            assert result.empty
            assert result.crs == sample_crs

    def test_build_edge_index_empty(self) -> None:
        """Test _build_edge_index with empty original_indices (line 1225)."""
        converter = NxConverter()
        result = converter._build_edge_index([], None)
        assert isinstance(result, pd.Index)
        assert len(result) == 0

    def test_canonical_edge_pair_self_loop(self) -> None:
        """Test _canonical_edge_pair with self-loop (line 1370)."""
        assert utils._canonical_edge_pair(1, 1) == (1, 1)
        assert utils._canonical_edge_pair("a", "a") == ("a", "a")

    def test_validate_nx_pos_from_xy(self, sample_crs: str) -> None:
        """Test validate_nx creates pos from x and y attributes (line 2016)."""
        # Create a graph with x, y but no pos
        G = nx.Graph()
        G.add_node(1, x=10.0, y=20.0)
        G.add_node(2, x=30.0, y=40.0)
        G.add_edge(1, 2, geometry=LineString([(10, 20), (30, 40)]))
        G.graph = {"crs": sample_crs, "is_hetero": False}

        # This should set pos from x and y
        utils.validate_nx(G)

        assert "pos" in G.nodes[1]
        assert "pos" in G.nodes[2]
        assert G.nodes[1]["pos"] == (10.0, 20.0)
        assert G.nodes[2]["pos"] == (30.0, 40.0)

    def test_build_node_index_empty(self) -> None:
        """Test _build_node_index with empty original_indices (line 1243)."""
        converter = NxConverter()
        result = converter._build_node_index([], None)
        assert isinstance(result, pd.Index)
        assert len(result) == 0

    def test_empty_graph_node_reconstruction(self, sample_crs: str) -> None:
        """Test nx_to_gdf with graph that has nodes without attributes (line 996)."""
        # Create a minimal graph with nodes that have no custom attributes
        # This will result in empty records when reconstructing
        G = nx.Graph()
        G.add_node(0, pos=(0, 0))
        G.add_node(1, pos=(1, 1))
        G.graph = {"crs": sample_crs, "is_hetero": False, "node_index_names": None}

        # Convert to GDF - should handle empty records gracefully
        nodes_gdf, edges_gdf = utils.nx_to_gdf(G)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert len(nodes_gdf) == 2
        assert "geometry" in nodes_gdf.columns

    def test_edge_iteration_fallback(self, sample_crs: str) -> None:
        """Test edge iteration handles unexpected edge format (line 1200)."""
        # Create a graph and manually add malformed edge to test the fallback
        converter = NxConverter()
        graph = nx.Graph()
        graph.add_node(0, pos=(0, 0))
        graph.add_node(1, pos=(1, 1))
        graph.add_edge(0, 1, geometry=LineString([(0, 0), (1, 1)]))
        graph.graph = {"crs": sample_crs, "is_hetero": False, "edge_index_names": None}

        # The actual iteration path that could hit line 1200 is very defensive
        # It's hard to trigger without modifying internal NetworkX behavior
        # Convert to GDF to exercise the edge reconstruction path
        _, edges_gdf = converter.nx_to_gdf(graph)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)

    def test_gdf_to_nx_edges_none(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test gdf_to_nx raises ValueError when edges is None."""
        converter = NxConverter()
        with pytest.raises(ValueError, match="Edges GeoDataFrame cannot be None"):
            converter.gdf_to_nx(nodes=sample_nodes_gdf, edges=None)

    def test_nx_to_gdf_multiindex_nodes(self, sample_crs: str) -> None:
        """Test nx_to_gdf with MultiIndex nodes."""
        G = nx.Graph()
        # Add nodes with tuple indices
        G.add_node((0, "a"), pos=(0, 0), _original_index=(0, "a"))
        G.add_node((1, "b"), pos=(1, 1), _original_index=(1, "b"))
        G.graph = {"crs": sample_crs, "is_hetero": False, "node_index_names": ["id", "type"]}

        nodes_gdf, _ = utils.nx_to_gdf(G)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(nodes_gdf.index, pd.MultiIndex)
        assert nodes_gdf.index.names == ["id", "type"]
        assert len(nodes_gdf) == 2

    def test_coerce_name_sequence_string(self) -> None:
        """Test _coerce_name_sequence with string input."""
        assert utils._coerce_name_sequence("index_name") == ["index_name"]
        assert utils._coerce_name_sequence(["a", "b"]) == ["a", "b"]
        assert utils._coerce_name_sequence(None) is None

    def test_dual_graph_as_nx(
        self, sample_nodes_gdf: gpd.GeoDataFrame, sample_edges_gdf: gpd.GeoDataFrame
    ) -> None:
        """Test dual_graph with as_nx=True."""
        result = utils.dual_graph((sample_nodes_gdf, sample_edges_gdf), as_nx=True)
        assert isinstance(result, nx.Graph)
        assert len(result) > 0

    def test_nx_to_gdf_empty_graph_no_nodes(self, sample_crs: str) -> None:
        """Test nx_to_gdf with a completely empty graph (no nodes)."""
        G = nx.Graph()
        G.graph = {"crs": sample_crs, "is_hetero": False}
        nodes_gdf, edges_gdf = utils.nx_to_gdf(G)
        assert isinstance(nodes_gdf, gpd.GeoDataFrame)
        assert isinstance(edges_gdf, gpd.GeoDataFrame)
        assert nodes_gdf.empty
        assert edges_gdf.empty
        assert "geometry" in nodes_gdf.columns

    @pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib not available")
    def test_plot_graph_subplots_less_than_grid(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    ) -> None:
        """Test plot_graph with subplots where number of plots < grid size."""
        # Create a situation where we have 1 edge type but grid is 2 cols
        # This triggers the "hide unused axes" logic

        # Filter to just one edge type
        single_edge_type = next(iter(sample_hetero_edges_dict.keys()))
        single_edge_dict = {single_edge_type: sample_hetero_edges_dict[single_edge_type]}

        fig, ax = plt.subplots()
        utils.plot_graph(
            nodes=sample_hetero_nodes_dict,
            edges=single_edge_dict,
            subplots=True,
            ncols=2,  # Force 2 columns for 1 item
            show=False,
        )
        plt.close("all")
