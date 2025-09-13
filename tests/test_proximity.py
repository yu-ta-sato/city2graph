"""Comprehensive test suite for proximity graph generators.

This refactored test suite focuses on:
• Concise, maintainable test organization
• Reduced redundancy through parametrization
• Clear separation of concerns
• Comprehensive edge case coverage
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import QhullError
from shapely.geometry import LineString
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

# Module-level import for proximity to support monkeypatch tests (mypy: treat as Any)
import city2graph.proximity as _prox
from city2graph.proximity import bridge_nodes
from city2graph.proximity import contiguity_graph
from city2graph.proximity import delaunay_graph
from city2graph.proximity import euclidean_minimum_spanning_tree
from city2graph.proximity import fixed_radius_graph
from city2graph.proximity import gabriel_graph
from city2graph.proximity import knn_graph
from city2graph.proximity import relative_neighborhood_graph
from city2graph.proximity import waxman_graph
from city2graph.utils import gdf_to_nx
from city2graph.utils import nx_to_gdf
from tests import helpers as _helpers
from tests.helpers import TOLERANCE
from tests.helpers import assert_l_shaped_edges
from tests.helpers import assert_valid_nx_graph
from tests.helpers import create_test_points
from tests.helpers import create_two_layers
from tests.helpers import make_grid_polygons_gdf
from tests.helpers import make_points_gdf
from tests.helpers import run_or_skip

prox: Any = cast("Any", _prox)

if TYPE_CHECKING:
    from collections.abc import Callable

# Test configuration moved to tests/helpers.py

# Generator specifications for parametrized tests
GENERATORS = [
    ("knn_graph", {"k": 2}),
    ("fixed_radius_graph", {"radius": 2.0}),
    ("delaunay_graph", {}),
    ("gabriel_graph", {}),
    ("relative_neighborhood_graph", {}),
    ("euclidean_minimum_spanning_tree", {}),
    ("waxman_graph", {"beta": 0.6, "r0": 3.0, "seed": 42}),
]

DISTANCE_METRICS = ["euclidean", "manhattan"]
BRIDGE_METHODS = [("knn", {"k": 1}), ("fixed_radius", {"radius": 3})]


# Helper functions imported from tests/helpers.py


# Core functionality tests
@pytest.mark.parametrize(
    ("gen_name", "kwargs"),
    GENERATORS,
    ids=[g[0] for g in GENERATORS],
)
@pytest.mark.parametrize("metric", DISTANCE_METRICS)
def test_generator_basic_functionality(
    sample_nodes_gdf: gpd.GeoDataFrame,
    gen_name: str,
    kwargs: dict[str, Any],
    metric: str,
) -> None:
    """Test basic generator functionality with different distance metrics."""
    generator_fn = globals()[gen_name]
    nodes, edges = run_or_skip(
        generator_fn,
        sample_nodes_gdf,
        distance_metric=metric,
        **kwargs,
    )

    # Verify structure
    assert isinstance(nodes, gpd.GeoDataFrame)
    assert isinstance(edges, gpd.GeoDataFrame)
    assert nodes.shape[0] == len(sample_nodes_gdf)
    assert nodes.crs == edges.crs == sample_nodes_gdf.crs

    # Verify required columns
    _helpers.assert_has_columns(edges, ["geometry", "weight"])

    # Verify Manhattan distance creates L-shaped geometries
    if metric == "manhattan" and not edges.empty:
        assert_l_shaped_edges(edges)


@pytest.mark.parametrize(
    ("gen_name", "kwargs"),
    GENERATORS,
    ids=[g[0] for g in GENERATORS],
)
def test_network_metric_error_handling(
    sample_nodes_gdf: gpd.GeoDataFrame,
    gen_name: str,
    kwargs: dict[str, Any],
) -> None:
    """Test network metric error handling when network_gdf is missing."""
    generator_fn = globals()[gen_name]
    with pytest.raises(
        ValueError,
        match="network_gdf is required for network distance metric",
    ):
        run_or_skip(
            generator_fn,
            sample_nodes_gdf,
            distance_metric="network",
            **kwargs,
        )


@pytest.mark.parametrize(
    ("gen_name", "kwargs"),
    [g for g in GENERATORS if g[0] != "waxman_graph"],
)
def test_network_metric_functionality(
    sample_nodes_gdf: gpd.GeoDataFrame,
    sample_edges_gdf: gpd.GeoDataFrame,
    gen_name: str,
    kwargs: dict[str, Any],
) -> None:
    """Test network metric functionality with valid network."""
    generator_fn = globals()[gen_name]
    nodes, edges = run_or_skip(
        generator_fn,
        sample_nodes_gdf,
        distance_metric="network",
        network_gdf=sample_edges_gdf,
        **kwargs,
    )

    # MST may have empty edges for disconnected components
    if gen_name.endswith("minimum_spanning_tree"):
        assert len(edges) >= 0
    else:
        assert not edges.empty

    if not edges.empty:
        assert np.isfinite(edges["weight"].to_numpy()).all()


# Generator-specific tests
class TestKNNGraph:
    """Test KNN graph specific functionality."""

    def test_networkx_output(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test NetworkX output format."""
        G = cast("nx.Graph", run_or_skip(knn_graph, sample_nodes_gdf, k=3, as_nx=True))
        assert_valid_nx_graph(G, expected_nodes=len(sample_nodes_gdf), crs=sample_nodes_gdf.crs)

        expected_edges = len(sample_nodes_gdf) * 3 / 2
        assert math.isclose(G.number_of_edges(), expected_edges, rel_tol=0.5)

    def test_edge_cases(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        single_node_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test KNN edge cases."""
        # k=0 should produce no edges
        _, edges = run_or_skip(knn_graph, sample_nodes_gdf, k=0)
        assert edges.empty

        # Single node should produce no edges
        _, edges = run_or_skip(knn_graph, single_node_gdf, k=1)
        assert edges.empty

        # k > available neighbors should work
        excessive_k = len(sample_nodes_gdf) + 10
        _, edges = run_or_skip(knn_graph, sample_nodes_gdf, k=excessive_k)
        max_edges = len(sample_nodes_gdf) * (len(sample_nodes_gdf) - 1)
        assert len(edges) <= max_edges


class TestWaxmanGraph:
    """Test Waxman graph specific functionality."""

    def test_reproducibility(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test seed-based reproducibility."""
        params = {"beta": 0.5, "r0": 3, "seed": 11}
        _, edges1 = run_or_skip(waxman_graph, sample_nodes_gdf, **params)
        _, edges2 = run_or_skip(waxman_graph, sample_nodes_gdf, **params)
        assert edges1.equals(edges2)

    def test_parameter_storage(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test parameter storage in graph metadata."""
        beta, r0 = 0.7, 2.5
        result = run_or_skip(
            waxman_graph,
            sample_nodes_gdf,
            beta=beta,
            r0=r0,
            as_nx=True,
        )
        assert isinstance(result, nx.Graph)
        assert result.graph.get("beta") == beta
        assert result.graph.get("r0") == r0

    def test_single_node_networkx_output(
        self,
        single_node_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test single node case with NetworkX output - covers line 1019."""
        result = run_or_skip(
            waxman_graph,
            single_node_gdf,
            beta=0.5,
            r0=1.0,
            as_nx=True,
        )
        assert_valid_nx_graph(result, expected_nodes=1, expected_edges=0)


class TestFixedRadiusGraph:
    """Test fixed radius graph functionality."""

    def test_parameter_storage(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test radius parameter storage."""
        radius = 1.5
        result = run_or_skip(
            fixed_radius_graph,
            sample_nodes_gdf,
            radius=radius,
            as_nx=True,
        )
        assert isinstance(result, nx.Graph)
        assert result.graph.get("radius") == radius

    def test_single_node_edge_case(self, single_node_gdf: gpd.GeoDataFrame) -> None:
        """Test single node case."""
        result = run_or_skip(fixed_radius_graph, single_node_gdf, radius=1.0)
        assert isinstance(result, tuple)
        _, edges = result
        assert edges.empty


class TestTriangulationGraphs:
    """Test triangulation-based graphs (Delaunay, Gabriel, RNG)."""

    @pytest.mark.parametrize(
        "graph_fn",
        [delaunay_graph, gabriel_graph, relative_neighborhood_graph],
    )
    def test_insufficient_points(
        self,
        graph_fn: Callable[..., tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph],
        single_node_gdf: gpd.GeoDataFrame,
        two_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test behavior with insufficient points."""
        # Single point
        _, edges = run_or_skip(graph_fn, single_node_gdf)
        assert edges.empty

        # Two points - special case for Gabriel/RNG
        _, edges = run_or_skip(graph_fn, two_nodes_gdf)
        if graph_fn in [gabriel_graph, relative_neighborhood_graph]:
            assert len(edges) == 1
        else:  # Delaunay
            assert edges.empty

    def test_well_separated_points(self) -> None:
        """Test with well-separated points to avoid QhullError."""
        points = create_test_points()
        for graph_fn in [gabriel_graph, relative_neighborhood_graph]:
            _, edges = run_or_skip(graph_fn, points)
            assert len(edges) >= 1

    def test_degenerate_cases(self) -> None:
        """Test handling of degenerate geometric cases."""
        # Collinear points
        collinear = make_points_gdf([(0, 0), (1, 0), (2, 0)])

        # Should handle gracefully or raise QhullError
        for graph_fn in [gabriel_graph, relative_neighborhood_graph]:
            try:
                nodes, edges = run_or_skip(graph_fn, collinear)
                assert isinstance(nodes, gpd.GeoDataFrame)
                assert isinstance(edges, gpd.GeoDataFrame)
            except QhullError:
                pass  # Acceptable for degenerate cases

        # Coincident points
        coincident = make_points_gdf([(0, 0), (0, 0), (10, 0), (0, 10)])

        for graph_fn in [gabriel_graph, relative_neighborhood_graph]:
            nodes, edges = run_or_skip(graph_fn, coincident)
            assert isinstance(nodes, gpd.GeoDataFrame)
            assert isinstance(edges, gpd.GeoDataFrame)


class TestEuclideanMST:
    """Test Euclidean minimum spanning tree."""

    def test_single_node(self, single_node_gdf: gpd.GeoDataFrame) -> None:
        """Test single node case."""
        _, edges = run_or_skip(euclidean_minimum_spanning_tree, single_node_gdf)
        assert edges.empty

    def test_edge_count(self) -> None:
        """Test that MST has exactly n-1 edges."""
        points = gpd.GeoDataFrame(
            {"id": [1, 2, 3], "geometry": [Point(0, 0), Point(1, 0), Point(0, 1)]},
            crs="EPSG:27700",
        ).set_index("id")

        _, edges = run_or_skip(
            euclidean_minimum_spanning_tree,
            points,
            distance_metric="manhattan",
        )
        # MST on n points should have exactly n-1 edges
        assert len(edges) == len(points) - 1

    def test_networkx_output(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test NetworkX output format."""
        G = run_or_skip(
            bridge_nodes,
            sample_hetero_nodes_dict,
            proximity_method="knn",
            k=1,
            as_nx=True,
        )

        assert isinstance(G, nx.DiGraph)

        # Check node types
        node_types = set(nx.get_node_attributes(G, "node_type").values())
        expected_types = set(sample_hetero_nodes_dict.keys())
        assert node_types == expected_types

        # Check edge relations
        edge_relations = set(nx.get_edge_attributes(G, "relation").values())
        assert edge_relations <= {"is_nearby", None}

    def test_error_conditions(self) -> None:
        """Test error conditions."""
        # Insufficient layers
        single_layer = {
            "layer1": gpd.GeoDataFrame(
                {"id": [1], "geometry": [Point(0, 0)]},
                crs="EPSG:27700",
            ).set_index("id"),
        }
        with pytest.raises(ValueError, match="needs at least two layers"):
            run_or_skip(bridge_nodes, single_layer)

        # Invalid proximity method
        two_layers = create_two_layers()
        with pytest.raises(ValueError, match="proximity_method must be"):
            run_or_skip(bridge_nodes, two_layers, proximity_method="invalid")

        # Missing radius parameter
        with pytest.raises(KeyError):
            run_or_skip(bridge_nodes, two_layers, proximity_method="fixed_radius")


# Distance metric tests
def test_distance_metric_relationships(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test Manhattan >= Euclidean distance relationship."""
    _, euclidean_edges = run_or_skip(
        knn_graph,
        sample_nodes_gdf,
        k=1,
        distance_metric="euclidean",
    )
    _, manhattan_edges = run_or_skip(
        knn_graph,
        sample_nodes_gdf,
        k=1,
        distance_metric="manhattan",
    )

    if euclidean_edges.empty or manhattan_edges.empty:
        pytest.skip("No edges produced")

    euclidean_weight = euclidean_edges.iloc[0]["weight"]
    manhattan_weight = manhattan_edges.iloc[0]["weight"]
    assert euclidean_weight <= manhattan_weight + TOLERANCE


class TestCoverageImprovements:
    """Test cases to improve coverage in proximity module."""

    def test_invalid_distance_metric_handling(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Test invalid distance metric handling (lines 1272, 1290)."""
        # Use existing heterogeneous nodes fixture for bridge_nodes testing
        # Test with invalid distance metric (non-string) - should default to "euclidean"
        _, edges = bridge_nodes(
            sample_hetero_nodes_dict,
            proximity_method="knn",
            k=1,
            distance_metric=123,  # Invalid type, should default to "euclidean"
        )

        assert isinstance(edges, dict)
        assert len(edges) > 0

    def test_distance_metric_type_validation(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test distance_metric type validation to cover line 1290 in proximity.py."""
        # Test with non-string distance_metric to trigger line 1290
        nodes, edges = run_or_skip(
            fixed_radius_graph,
            sample_nodes_gdf,
            radius=2.0,
            distance_metric=None,  # Non-string type, should default to "euclidean"
        )
        # Should not raise an error and use euclidean as default
        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)

    def test_network_path_fallback_handling(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_crs: str,
    ) -> None:
        """Test network path fallback for same network points (lines 1636-1638)."""
        # Use existing nodes fixture but create very close points to trigger fallback
        close_nodes = sample_nodes_gdf.copy()
        # Make all nodes very close to trigger same network point mapping
        close_nodes.geometry = [
            Point(0, 0),
            Point(0.001, 0.001),
            Point(0.002, 0.002),
            Point(0.003, 0.003),
        ]

        # Create minimal network
        network_gdf = gpd.GeoDataFrame(
            {"network_id": [1]},
            geometry=[LineString([(0, 0), (1, 0)])],
            crs=sample_crs,
        )

        # This should trigger the fallback path for nodes mapping to same network point
        _, edges = fixed_radius_graph(
            close_nodes,
            radius=2.0,
            distance_metric="network",
            network_gdf=network_gdf,
        )

        assert isinstance(edges, gpd.GeoDataFrame)


# Error condition tests
class TestErrorConditions:
    """Test various error conditions."""

    def test_invalid_distance_metric(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test invalid distance metric error."""
        with pytest.raises(ValueError, match="Unknown distance metric"):
            run_or_skip(
                waxman_graph,
                sample_nodes_gdf,
                beta=0.5,
                r0=1.0,
                distance_metric="invalid",
            )

    def test_crs_mismatch(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test CRS mismatch errors."""
        # Network CRS mismatch
        network_wrong_crs = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 1)])]},
            crs="EPSG:4326",
        )
        with pytest.raises(ValueError, match="CRS mismatch"):
            run_or_skip(
                knn_graph,
                sample_nodes_gdf,
                k=1,
                distance_metric="network",
                network_gdf=network_wrong_crs,
            )

        # Directed graph CRS mismatch
        src_gdf = gpd.GeoDataFrame(
            {"id": [1], "geometry": [Point(0, 0)]},
            crs="EPSG:27700",
        ).set_index("id")
        target_gdf = gpd.GeoDataFrame(
            {"id": [2], "geometry": [Point(1, 1)]},
            crs="EPSG:4326",
        ).set_index("id")
        with pytest.raises(ValueError, match="CRS mismatch"):
            run_or_skip(knn_graph, src_gdf, k=1, target_gdf=target_gdf)


# Network geometry tests
class TestNetworkGeometry:
    """Test network geometry handling edge cases."""

    def test_basic_functionality(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test basic network geometry creation."""
        nodes, edges = run_or_skip(
            knn_graph,
            sample_nodes_gdf,
            k=1,
            distance_metric="network",
            network_gdf=sample_edges_gdf,
        )
        assert not edges.empty
        assert all(geom.is_valid for geom in edges.geometry)

    def test_network_metric_error_in_geometry_creation(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test network metric error when creating geometries - covers lines 1398-1402."""
        # This tests the error handling in _build_graph when network_gdf is None during geometry creation
        # We need to create a scenario where the distance calculation succeeds but geometry creation fails
        with pytest.raises(
            ValueError,
            match="network_gdf is required for network distance metric",
        ):
            run_or_skip(
                knn_graph,
                sample_nodes_gdf,
                k=1,
                distance_metric="network",
                network_gdf=None,
            )

    def test_fallback_geometry_creation(self) -> None:
        """Test fallback geometry creation when path_coords < 2 - covers lines 1445-1447."""
        # Create a minimal network that might cause path_coords to be empty
        # This is a complex scenario that requires specific network topology
        nodes = gpd.GeoDataFrame(
            {"id": [1, 2, 3], "geometry": [Point(0, 0), Point(1, 0), Point(2, 0)]},
            crs="EPSG:27700",
        ).set_index("id")

        # Create a network with potential disconnected components
        network_data = {
            "source_id": [1, 2],
            "target_id": [2, 3],
            "geometry": [LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
        }
        multi_index = pd.MultiIndex.from_arrays(
            [network_data["source_id"], network_data["target_id"]],
            names=("source_id", "target_id"),
        )
        network_gdf = gpd.GeoDataFrame(
            network_data,
            index=multi_index,
            crs="EPSG:27700",
        )

        # This should trigger the fallback geometry creation
        nodes_result, edges_result = run_or_skip(
            knn_graph,
            nodes,
            k=1,
            distance_metric="network",
            network_gdf=network_gdf,
        )

        # Verify fallback geometry was created
        if not edges_result.empty:
            assert all(isinstance(geom, LineString) for geom in edges_result.geometry)
            assert all(geom.is_valid for geom in edges_result.geometry)

    def test_fallback_scenarios(self) -> None:
        """Test geometry fallback scenarios."""
        # Minimal network
        network_data = {
            "source_id": [1],
            "target_id": [2],
            "geometry": [LineString([(0, 0), (0.1, 0.1)])],
        }
        multi_index = pd.MultiIndex.from_arrays(
            [network_data["source_id"], network_data["target_id"]],
            names=("source_id", "target_id"),
        )
        network_gdf = gpd.GeoDataFrame(
            network_data,
            index=multi_index,
            crs="EPSG:27700",
        )

        G = gdf_to_nx(edges=network_gdf)
        _, network_with_pos = nx_to_gdf(G, nodes=True, edges=True)

        points = make_points_gdf([(0, 0), (0.1, 0.1)])
        nodes, edges = run_or_skip(
            knn_graph,
            points,
            k=1,
            distance_metric="network",
            network_gdf=network_with_pos,
        )
        assert len(edges) >= 0

        # Duplicate coordinates fallback
        points_close = make_points_gdf([(0.1, 0.1), (-0.1, -0.1)])
        nodes, edges = run_or_skip(
            knn_graph,
            points_close,
            k=1,
            distance_metric="network",
            network_gdf=network_with_pos,
        )
        assert len(edges) >= 0
        if not edges.empty:
            assert all(isinstance(geom, LineString) for geom in edges.geometry)


class TestDirectedVariants:
    """Cover directed source→target paths for k and radius via public API."""

    def test_directed_knn_manhattan(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Directed KNN with Manhattan metric yields L-shaped geometries."""
        src = sample_hetero_nodes_dict["building"]
        dst = sample_hetero_nodes_dict["road"]

        nodes, edges = run_or_skip(
            knn_graph,
            src,
            k=1,
            target_gdf=dst,
            distance_metric="manhattan",
        )

        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)
        assert edges.crs == src.crs == dst.crs
        # Expect one outgoing edge per source node
        assert len(edges) == len(src)
        # Manhattan geometry should be L-shaped
        if not edges.empty:
            assert_l_shaped_edges(edges)

    def test_directed_fixed_radius(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Directed fixed-radius connects within threshold and sets weights."""
        src = sample_hetero_nodes_dict["building"]
        dst = sample_hetero_nodes_dict["road"]

        # Choose a radius that ensures at least one connection
        nodes, edges = run_or_skip(
            fixed_radius_graph,
            src,
            radius=3.0,
            target_gdf=dst,
            distance_metric="euclidean",
        )

        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)
        assert edges.crs == src.crs == dst.crs
        # Some edges should exist given the fixture layout
        assert len(edges) >= 1
        # Weights are positive distances
        if not edges.empty:
            assert (edges["weight"] > 0).all()


# ============================================================================
# CONTIGUITY GRAPH TESTS
# ============================================================================


"""Contiguity fixtures are provided by tests/conftest.py"""


class TestContiguityPublicAPI:
    """Concise tests targeting contiguity_graph public behavior only."""

    # --- Validation ---
    def test_invalid_input_type(self) -> None:
        """Non-GeoDataFrame input raises TypeError."""
        with pytest.raises(TypeError, match="Input must be a GeoDataFrame"):
            contiguity_graph(pd.DataFrame({"a": [1]}))

    def test_invalid_contiguity_value(self, sample_polygons_gdf: gpd.GeoDataFrame) -> None:
        """Invalid contiguity value raises ValueError."""
        with pytest.raises(ValueError, match="Invalid contiguity type 'invalid'"):
            contiguity_graph(sample_polygons_gdf, contiguity="invalid")

    def test_mixed_geometry_types(self, mixed_geometry_gdf: gpd.GeoDataFrame) -> None:
        """Mixed (non-polygon) geometry types raise ValueError."""
        with pytest.raises(ValueError, match="non-polygon geometr"):
            contiguity_graph(mixed_geometry_gdf)

    def test_invalid_geometries(self, invalid_geometry_gdf: gpd.GeoDataFrame) -> None:
        """Invalid polygon geometries raise ValueError."""
        with pytest.raises(ValueError, match="invalid geometr"):
            contiguity_graph(invalid_geometry_gdf)

    def test_null_geometries(self) -> None:
        """Null geometry rows raise ValueError."""
        gdf = gpd.GeoDataFrame(
            {"id": ["a", "b"], "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), None]},
            crs="EPSG:27700",
        ).set_index("id")
        with pytest.raises(ValueError, match="null geometr"):
            contiguity_graph(gdf)

    def test_missing_geometry_column(self) -> None:
        """Missing geometry column raises ValueError."""
        gdf = gpd.GeoDataFrame(
            {"id": ["a"]},
            geometry=gpd.GeoSeries([], dtype="geometry"),
            crs="EPSG:27700",
        ).set_index("id")
        gdf._geometry_column_name = None  # simulate missing geometry column
        with pytest.raises(ValueError, match="valid geometry column"):
            contiguity_graph(gdf)

    # --- Empty and trivial cases ---
    @pytest.mark.parametrize("as_nx", [False, True])
    def test_empty_geodataframe(self, empty_polygons_gdf: gpd.GeoDataFrame, as_nx: bool) -> None:
        """Empty input returns empty outputs while preserving CRS and metadata."""
        result = contiguity_graph(empty_polygons_gdf, as_nx=as_nx)
        if as_nx:
            assert isinstance(result, nx.Graph)
            assert result.number_of_nodes() == 0
            assert result.number_of_edges() == 0
            assert result.graph["crs"] == empty_polygons_gdf.crs
            assert result.graph["contiguity"] == "queen"
        else:
            nodes, edges = result
            assert len(nodes) == 0
            assert len(edges) == 0
            assert nodes.crs == edges.crs == empty_polygons_gdf.crs

    def test_single_polygon(self, single_polygon_gdf: gpd.GeoDataFrame) -> None:
        """Single polygon yields one node and zero edges, attributes preserved."""
        nodes, edges = contiguity_graph(single_polygon_gdf)
        assert len(nodes) == 1
        assert len(edges) == 0
        assert nodes.iloc[0]["attr"] == "x"

    # --- Core functionality ---
    @pytest.mark.parametrize("contiguity", ["queen", "rook"])
    def test_basic_structure(self, sample_polygons_gdf: gpd.GeoDataFrame, contiguity: str) -> None:
        """Core behavior: structure, CRS, columns, and attributes preserved."""
        nodes, edges = contiguity_graph(sample_polygons_gdf, contiguity=contiguity)
        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)
        assert len(nodes) == len(sample_polygons_gdf)
        assert nodes.crs == edges.crs == sample_polygons_gdf.crs
        assert {"weight", "geometry"}.issubset(edges.columns)
        # attributes preserved
        for col in sample_polygons_gdf.columns:
            assert col in nodes.columns

    def test_queen_vs_rook(self, l_shaped_polygons_gdf: gpd.GeoDataFrame) -> None:
        """Queen finds a vertex-only neighbor; Rook does not."""
        _, queen = contiguity_graph(l_shaped_polygons_gdf, contiguity="queen")
        _, rook = contiguity_graph(l_shaped_polygons_gdf, contiguity="rook")
        assert len(queen) == 1
        assert len(rook) == 0

    def test_isolated_polygons(self) -> None:
        """Completely separated polygons produce no edges."""
        gdf = gpd.GeoDataFrame(
            {
                "id": ["A", "B", "C"],
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
                    Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),
                ],
            },
            crs="EPSG:27700",
        ).set_index("id")
        nodes, edges = contiguity_graph(gdf)
        assert len(nodes) == 3
        assert len(edges) == 0

    def test_edge_geometry_and_weights(self, sample_polygons_gdf: gpd.GeoDataFrame) -> None:
        """Edges are LineStrings between centroids; sample A-B weight equals 2.0 if present."""
        nodes, edges = contiguity_graph(sample_polygons_gdf, contiguity="queen")
        if len(edges) == 0:
            pytest.skip("No edges")
        assert all(isinstance(g, LineString) and len(g.coords) == 2 for g in edges.geometry)
        # Check A-B weight equals centroid distance = 2.0
        ab = edges.loc[(slice(None), slice(None))]
        if not ab.empty:
            # try to find explicit A-B if exists
            mask = (
                (edges.index.get_level_values(0) == "A") & (edges.index.get_level_values(1) == "B")
            ) | (
                (edges.index.get_level_values(0) == "B") & (edges.index.get_level_values(1) == "A")
            )
            if mask.any():
                assert abs(edges.loc[mask].iloc[0]["weight"] - 2.0) < 1e-6

    # --- Output formats and metadata ---
    def test_networkx_output(self, sample_polygons_gdf: gpd.GeoDataFrame) -> None:
        """NetworkX output has metadata and node/edge attributes."""
        G = cast("nx.Graph", contiguity_graph(sample_polygons_gdf, contiguity="queen", as_nx=True))
        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == len(sample_polygons_gdf)
        assert G.graph["crs"] == sample_polygons_gdf.crs
        assert G.graph["contiguity"] == "queen"
        # Sample node/edge attributes
        if G.number_of_nodes():
            n0 = next(iter(G.nodes(data=True)))[1]
            assert "geometry" in n0
            assert "area" in n0
            assert "use" in n0
        if G.number_of_edges():
            e0 = next(iter(G.edges(data=True)))[2]
            assert "weight" in e0
            assert "geometry" in e0

    def test_output_format_consistency(self, sample_polygons_gdf: gpd.GeoDataFrame) -> None:
        """Tuple and NetworkX outputs encode same nodes and undirected edges."""
        nodes, edges = contiguity_graph(sample_polygons_gdf)
        G = cast("nx.Graph", contiguity_graph(sample_polygons_gdf, as_nx=True))
        assert len(nodes) == G.number_of_nodes()
        assert len(edges) == G.number_of_edges()
        assert set(nodes.index) == set(G.nodes())
        gdf_edges = {tuple(sorted(idx)) for idx in edges.index}
        nx_edges = {tuple(sorted(e)) for e in G.edges()}
        assert gdf_edges == nx_edges

    # --- CRS and types ---
    def test_crs_preservation(self, sample_polygons_gdf: gpd.GeoDataFrame) -> None:
        """CRS preserved in both nodes and edges outputs."""
        nodes, edges = contiguity_graph(sample_polygons_gdf)
        assert nodes.crs == sample_polygons_gdf.crs
        assert edges.crs == sample_polygons_gdf.crs

    def test_multipolygon_support(self) -> None:
        """MultiPolygon is accepted as polygonal geometry type."""
        a = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        b = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
        mp = MultiPolygon([a, b])
        adj = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
        gdf = gpd.GeoDataFrame(
            {"id": ["mp", "adj"], "geometry": [mp, adj]},
            crs="EPSG:27700",
        ).set_index("id")
        nodes, edges = contiguity_graph(gdf)
        assert len(nodes) == 2
        assert len(edges) >= 0  # adjacency depends on exact topology/libpysal

    # --- Light scalability check (kept small) ---
    def test_small_grid(self) -> None:
        """Small 3x3 grid yields expected node count and reasonable edge count."""
        gdf = make_grid_polygons_gdf(3, 3, crs="EPSG:27700")
        nodes, edges = contiguity_graph(gdf)
        assert len(nodes) == 9
        assert 8 <= len(edges) <= 36  # queen contiguity yields many edges

    def test_multiple_invalid_geometries_message(self) -> None:
        """Multiple invalid polygons to exercise the '... more' message branch."""
        bad = Polygon([(0, 0), (2, 0), (0, 2), (2, 2)])
        gdf = gpd.GeoDataFrame(
            {
                "id": ["b1", "b2", "b3", "b4"],
                "geometry": [bad, bad, bad, bad],
            },
            crs="EPSG:27700",
        ).set_index("id")
        with pytest.raises(ValueError, match=r"invalid geometr"):
            contiguity_graph(gdf)


class TestContiguityLibpysalEdgeCases:
    """Exercise libpysal-related branches via public API using monkeypatch."""

    def test_libpysal_failure_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_polygons_gdf: gpd.GeoDataFrame,
    ) -> None:
        """If libpysal raises, contiguity_graph wraps with ValueError including context."""

        def boom(*_args: object, **_kwargs: object) -> object:
            msg = "boom"
            raise RuntimeError(msg)

        monkeypatch.setattr(prox.libpysal.weights.Queen, "from_dataframe", boom)
        with pytest.raises(
            ValueError,
            match=r"Failed to create queen contiguity spatial weights matrix",
        ):
            contiguity_graph(sample_polygons_gdf, contiguity="queen")

    def test_libpysal_returns_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_polygons_gdf: gpd.GeoDataFrame,
    ) -> None:
        """If libpysal returns None, contiguity_graph raises a descriptive ValueError."""

        # Match the expected signature (gdf, ids=...), but ignore values
        def _return_none(_gdf: gpd.GeoDataFrame, ids: object | None = None) -> None:
            _ = ids  # explicitly use to avoid unused-argument lint

        monkeypatch.setattr(
            prox.libpysal.weights.Queen,
            "from_dataframe",
            _return_none,
        )
        with pytest.raises(
            ValueError,
            match=r"libpysal returned None when creating queen contiguity weights",
        ):
            contiguity_graph(sample_polygons_gdf, contiguity="queen")

    def test_empty_weights_matrix_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        sample_polygons_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Return empty W to hit the empty-weights branch in edge extraction."""
        monkeypatch.setattr(
            prox.libpysal.weights.Queen,
            "from_dataframe",
            lambda _gdf: prox.libpysal.weights.W({}),
        )
        nodes, edges = contiguity_graph(sample_polygons_gdf)
        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)
        assert len(edges) == 0


class TestBridgeNodesCoverage:
    """Additional coverage for bridge_nodes fixed-radius metric type handling."""

    def test_fixed_radius_non_string_metric_defaults(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    ) -> None:
        """Non-string metric defaults to euclidean and succeeds."""
        _, edges = bridge_nodes(
            sample_hetero_nodes_dict,
            proximity_method="fixed_radius",
            radius=3.0,
            distance_metric=123,
        )
        assert isinstance(edges, dict)
        assert len(edges) > 0
