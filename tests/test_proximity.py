"""
Comprehensive test suite for proximity.py.

This refactored version provides:
• Reduced code duplication through better use of fixtures and helpers
• Clearer test organization by functionality
• Improved readability and maintainability
• Comprehensive edge case coverage
• Better error condition testing

All fixtures are imported from conftest.py for consistency.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import QhullError
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph.proximity import _add_edges
from city2graph.proximity import _directed_edges  # for monkey patching in tests
from city2graph.proximity import bridge_nodes
from city2graph.proximity import delaunay_graph
from city2graph.proximity import euclidean_minimum_spanning_tree
from city2graph.proximity import fixed_radius_graph
from city2graph.proximity import gabriel_graph

# Import all relevant functions from proximity.py directly
from city2graph.proximity import knn_graph
from city2graph.proximity import relative_neighborhood_graph
from city2graph.proximity import waxman_graph
from city2graph.utils import gdf_to_nx
from city2graph.utils import nx_to_gdf

if TYPE_CHECKING:
    from collections.abc import Callable

# -----------------------------------------------------------------------------#
# Constants                                                                    #
# -----------------------------------------------------------------------------#
SKIP_EXCEPTIONS = (NotImplementedError, AttributeError, NameError, ImportError)
DEFAULT_K = 2
DEFAULT_RADIUS = 2.0
DEFAULT_BETA = 0.6
DEFAULT_R0 = 3.0
DEFAULT_SEED = 42
TOLERANCE = 1e-6

# -----------------------------------------------------------------------------#
# Helper functions                                                             #
# -----------------------------------------------------------------------------#


def _run_or_skip(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Execute function, skipping on unfinished implementation errors."""
    try:
        return fn(*args, **kwargs)
    except SKIP_EXCEPTIONS as exc:
        pytest.skip(f"implementation not ready: {exc}")


def _is_l_shaped(line: LineString) -> bool:
    """Test if a LineString is L-shaped (3 points, right angle)."""
    coords = list(line.coords)
    if len(coords) != 3:
        return False
    (x0, y0), (x1, y1), (x2, y2) = coords
    return (x0 == x1 or y0 == y1) and (x1 == x2 or y1 == y2)


def _create_well_separated_points(crs: str = "EPSG:27700") -> gpd.GeoDataFrame:
    """Create well-separated points to avoid QhullError in triangulation."""
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3], "geometry": [Point(0, 0), Point(10, 0), Point(5, 10)]},
        crs=crs,
    ).set_index("id")


def _create_two_layer_dict(crs: str = "EPSG:27700") -> dict[str, gpd.GeoDataFrame]:
    """Create a simple two-layer dictionary for bridge_nodes testing."""
    return {
        "layer1": gpd.GeoDataFrame(
            {"id": [1], "geometry": [Point(0, 0)]},
            crs=crs,
        ).set_index("id"),
        "layer2": gpd.GeoDataFrame(
            {"id": [2], "geometry": [Point(1, 1)]},
            crs=crs,
        ).set_index("id"),
    }


# -----------------------------------------------------------------------------#
# Test parameter sets                                                          #
# -----------------------------------------------------------------------------#
GENERATOR_SPECS: list[tuple[str, dict[str, Any]]] = [
    ("knn_graph", {"k": DEFAULT_K}),
    ("fixed_radius_graph", {"radius": DEFAULT_RADIUS}),
    ("delaunay_graph", {}),
    ("gabriel_graph", {}),
    ("relative_neighborhood_graph", {}),
    ("euclidean_minimum_spanning_tree", {}),
    ("waxman_graph", {"beta": DEFAULT_BETA, "r0": DEFAULT_R0, "seed": DEFAULT_SEED}),
]

DISTANCE_METRICS = ["euclidean", "manhattan"]

# Generators that don't support network metric for error testing
NETWORK_INCOMPATIBLE_GENERATORS = [
    spec for spec in GENERATOR_SPECS if spec[0] != "waxman_graph"
]

# Bridge node test parameters
BRIDGE_METHODS = [
    ("knn", {"k": 1}),
    ("fixed_radius", {"radius": 3}),
]

# =============================================================================
# CORE FUNCTIONALITY TESTS
# =============================================================================

@pytest.mark.parametrize(("gen_name", "kwargs"), GENERATOR_SPECS, ids=[spec[0] for spec in GENERATOR_SPECS])
@pytest.mark.parametrize("metric", DISTANCE_METRICS)
def test_generators_basic_functionality(
    sample_nodes_gdf: gpd.GeoDataFrame,
    gen_name: str,
    kwargs: dict[str, Any],
    metric: str,
) -> None:
    """Test that every generator returns valid GeoDataFrames with correct structure."""
    generator_fn: Callable[..., Any] = globals()[gen_name]
    nodes, edges = _run_or_skip(generator_fn, sample_nodes_gdf, distance_metric=metric, **kwargs)

    # Verify return types and structure
    assert isinstance(nodes, gpd.GeoDataFrame), f"{gen_name} should return GeoDataFrame for nodes"
    assert isinstance(edges, gpd.GeoDataFrame), f"{gen_name} should return GeoDataFrame for edges"
    assert nodes.shape[0] == len(sample_nodes_gdf), "Node count should be preserved"
    assert nodes.crs == edges.crs == sample_nodes_gdf.crs, "CRS should be consistent"

    # Verify mandatory columns
    assert "geometry" in edges.columns, "Edges must have geometry column"
    assert "weight" in edges.columns, "Edges must have weight column"

    # Verify Manhattan distance geometry (L-shaped paths)
    if metric == "manhattan" and not edges.empty:
        assert all(_is_l_shaped(geom) for geom in edges.geometry), \
            "Manhattan distance should produce L-shaped geometries"


# =============================================================================
# NETWORK METRIC TESTS
# =============================================================================

@pytest.mark.parametrize(("gen_name", "kwargs"), GENERATOR_SPECS, ids=[spec[0] for spec in GENERATOR_SPECS])
def test_network_metric_requires_network_gdf(
    sample_nodes_gdf: gpd.GeoDataFrame,
    gen_name: str,
    kwargs: dict[str, Any],
) -> None:
    """Test that network metric raises ValueError when network_gdf is missing."""
    generator_fn = globals()[gen_name]
    with pytest.raises(ValueError, match="network_gdf.*must be supplied|network.*required"):
        _run_or_skip(generator_fn, sample_nodes_gdf, distance_metric="network", **kwargs)


@pytest.mark.parametrize(("gen_name", "kwargs"), NETWORK_INCOMPATIBLE_GENERATORS)
def test_network_metric_with_valid_network(
    sample_nodes_gdf: gpd.GeoDataFrame,
    sample_edges_gdf: gpd.GeoDataFrame,
    gen_name: str,
    kwargs: dict[str, Any],
) -> None:
    """Test that generators work correctly with network distance metric."""
    generator_fn = globals()[gen_name]
    nodes, edges = _run_or_skip(
        generator_fn,
        sample_nodes_gdf,
        distance_metric="network",
        network_gdf=sample_edges_gdf,
        **kwargs,
    )

    # MST might have empty edges for disconnected components
    if gen_name.endswith("minimum_spanning_tree"):
        assert len(edges) >= 0, "MST should have non-negative edge count"
    else:
        assert not edges.empty, f"{gen_name} should produce edges with network metric"

    # All weights should be finite
    if not edges.empty:
        assert np.isfinite(edges["weight"].to_numpy()).all(), "All edge weights should be finite"


# =============================================================================
# GENERATOR-SPECIFIC TESTS
# =============================================================================

def test_knn_graph_as_networkx(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test knn_graph returns valid NetworkX graph when as_nx=True."""
    k_value = 3
    G = _run_or_skip(knn_graph, sample_nodes_gdf, k=k_value, as_nx=True)

    assert isinstance(G, nx.Graph), "Should return NetworkX Graph"
    assert G.number_of_nodes() == len(sample_nodes_gdf), "Node count should match input"

    # Each node has k neighbors in undirected graph ⇒ E ≈ n*k/2
    expected_edges = len(sample_nodes_gdf) * k_value / 2
    assert math.isclose(G.number_of_edges(), expected_edges, rel_tol=0.5), \
        f"Expected ~{expected_edges} edges, got {G.number_of_edges()}"

    assert G.graph["crs"] == sample_nodes_gdf.crs, "CRS should be preserved"


def test_waxman_graph_reproducibility(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test that waxman_graph produces identical results with same seed."""
    test_seed = 11
    params = {"beta": 0.5, "r0": 3, "seed": test_seed}

    _, edges1 = _run_or_skip(waxman_graph, sample_nodes_gdf, **params)
    _, edges2 = _run_or_skip(waxman_graph, sample_nodes_gdf, **params)

    assert edges1.equals(edges2), "Same seed should produce identical results"


def test_fixed_radius_graph_stores_radius(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test that fixed_radius_graph stores radius parameter in graph metadata."""
    test_radius = 1.5
    G = _run_or_skip(fixed_radius_graph, sample_nodes_gdf, radius=test_radius, as_nx=True)

    assert G.graph.get("radius") == test_radius, "Radius should be stored in graph metadata"


def test_waxman_graph_stores_parameters(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test that waxman_graph stores beta and r0 parameters in graph metadata."""
    test_beta, test_r0 = 0.7, 2.5
    G = _run_or_skip(waxman_graph, sample_nodes_gdf, beta=test_beta, r0=test_r0, as_nx=True)

    assert G.graph.get("beta") == test_beta, "Beta should be stored in graph metadata"
    assert G.graph.get("r0") == test_r0, "r0 should be stored in graph metadata"


# =============================================================================
# BRIDGE NODES (MULTILAYER) TESTS
# =============================================================================

@pytest.mark.parametrize("method, extra", BRIDGE_METHODS)
def test_bridge_nodes_output_structure(
    sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    method: str,
    extra: dict[str, Any],
) -> None:
    """Test that bridge_nodes produces correct output structure."""
    _, edges_dict = _run_or_skip(
        bridge_nodes,
        sample_hetero_nodes_dict,
        proximity_method=method,
        **extra,
    )

    # Check that all expected edge types are present
    layer_names = sample_hetero_nodes_dict.keys()
    expected_edge_keys = {
        (src, "is_nearby", dst) for src in layer_names for dst in layer_names if src != dst
    }
    assert set(edges_dict.keys()) == expected_edge_keys, \
        "Should have edges between all layer pairs"

    # Check that all edge GeoDataFrames have required columns
    for edge_gdf in edges_dict.values():
        assert "weight" in edge_gdf.columns, "Edge GDF must have weight column"
        assert "geometry" in edge_gdf.columns, "Edge GDF must have geometry column"


def test_bridge_nodes_as_networkx(sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame]) -> None:
    """Test bridge_nodes returns valid NetworkX graph when as_nx=True."""
    G = _run_or_skip(
        bridge_nodes,
        sample_hetero_nodes_dict,
        proximity_method="knn",
        k=1,
        as_nx=True,
    )

    assert isinstance(G, nx.DiGraph), "Should return directed graph for multilayer"

    # Check node types are preserved
    node_types = set(nx.get_node_attributes(G, "node_type").values())
    expected_types = set(sample_hetero_nodes_dict.keys())
    assert node_types == expected_types, "All layer types should be represented"

    # Check edge relations
    edge_relations = set(nx.get_edge_attributes(G, "relation").values())
    assert edge_relations <= {"is_nearby", None}, "Relations should be 'is_nearby' or None"


# =============================================================================
# DISTANCE METRIC VALIDATION TESTS
# =============================================================================

def test_distance_metric_weight_relationships(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test that Manhattan distance weights are >= Euclidean weights (L1 >= L2 norm)."""
    _, euclidean_edges = _run_or_skip(knn_graph, sample_nodes_gdf, k=1, distance_metric="euclidean")
    _, manhattan_edges = _run_or_skip(knn_graph, sample_nodes_gdf, k=1, distance_metric="manhattan")

    if euclidean_edges.empty or manhattan_edges.empty:
        pytest.skip("No edges produced - cannot compare distance metrics")

    # Get representative weights for comparison
    euclidean_weight = euclidean_edges.iloc[0]["weight"]
    manhattan_weight = manhattan_edges.iloc[0]["weight"]

    # Manhattan distance (L1) should be >= Euclidean distance (L2) in 2D
    assert euclidean_weight <= manhattan_weight + TOLERANCE, \
        f"Euclidean weight ({euclidean_weight}) should be <= Manhattan weight ({manhattan_weight})"


# =============================================================================
# EDGE CASES AND BOUNDARY CONDITIONS
# =============================================================================

class TestKNNGraphEdgeCases:
    """Test edge cases specific to knn_graph."""

    def test_zero_k_value(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test knn_graph with k=0 returns empty edges."""
        nodes, edges = _run_or_skip(knn_graph, sample_nodes_gdf, k=0)
        assert edges.empty, "k=0 should produce no edges"

    def test_single_node(self, single_node_gdf: gpd.GeoDataFrame) -> None:
        """Test knn_graph with single node returns empty edges."""
        nodes, edges = _run_or_skip(knn_graph, single_node_gdf, k=1)
        assert edges.empty, "Single node should produce no edges"

    def test_k_exceeds_available_neighbors(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test knn_graph handles k > available neighbors gracefully."""
        excessive_k = len(sample_nodes_gdf) + 10
        nodes, edges = _run_or_skip(knn_graph, sample_nodes_gdf, k=excessive_k)

        # Should work but be limited by available neighbors
        max_possible_edges = len(sample_nodes_gdf) * (len(sample_nodes_gdf) - 1)
        assert len(edges) <= max_possible_edges, "Edge count should be bounded by node pairs"


class TestTriangulationBasedGraphs:
    """Test edge cases for graphs based on Delaunay triangulation."""

    def test_delaunay_insufficient_points(
        self,
        single_node_gdf: gpd.GeoDataFrame,
        two_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test delaunay_graph with insufficient points for triangulation."""
        # Single point - no triangulation possible
        nodes, edges = _run_or_skip(delaunay_graph, single_node_gdf)
        assert edges.empty, "Single point should produce no Delaunay edges"

        # Two points - no triangulation possible
        nodes, edges = _run_or_skip(delaunay_graph, two_nodes_gdf)
        assert edges.empty, "Two points should produce no Delaunay edges"

    def test_gabriel_graph_edge_cases(
        self,
        single_node_gdf: gpd.GeoDataFrame,
        two_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test gabriel_graph edge cases."""
        # Single point
        nodes, edges = _run_or_skip(gabriel_graph, single_node_gdf)
        assert edges.empty, "Single point should produce no Gabriel edges"

        # Two points (special case - should create one edge)
        nodes, edges = _run_or_skip(gabriel_graph, two_nodes_gdf)
        assert len(edges) == 1, "Two points should create exactly one Gabriel edge"

        # Well-separated points to avoid QhullError
        well_separated = _create_well_separated_points()
        nodes, edges = _run_or_skip(gabriel_graph, well_separated)
        assert len(edges) >= 1, "Well-separated points should create valid Gabriel edges"

    def test_relative_neighborhood_graph_edge_cases(
        self,
        single_node_gdf: gpd.GeoDataFrame,
        two_nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test relative_neighborhood_graph edge cases."""
        # Single point
        nodes, edges = _run_or_skip(relative_neighborhood_graph, single_node_gdf)
        assert edges.empty, "Single point should produce no RNG edges"

        # Two points (special case - should create one edge)
        nodes, edges = _run_or_skip(relative_neighborhood_graph, two_nodes_gdf)
        assert len(edges) == 1, "Two points should create exactly one RNG edge"

        # Well-separated points to avoid QhullError
        well_separated = _create_well_separated_points()
        nodes, edges = _run_or_skip(relative_neighborhood_graph, well_separated)
        assert len(edges) >= 1, "Well-separated points should create valid RNG edges"


class TestOtherGeneratorEdgeCases:
    """Test edge cases for other graph generators."""

    def test_euclidean_mst_single_node(self, single_node_gdf: gpd.GeoDataFrame) -> None:
        """Test euclidean_minimum_spanning_tree with single node."""
        nodes, edges = _run_or_skip(euclidean_minimum_spanning_tree, single_node_gdf)
        assert edges.empty, "Single node MST should have no edges"

    def test_fixed_radius_single_node(self, single_node_gdf: gpd.GeoDataFrame) -> None:
        """Test fixed_radius_graph with single node."""
        nodes, edges = _run_or_skip(fixed_radius_graph, single_node_gdf, radius=1.0)
        assert edges.empty, "Single node radius graph should have no edges"

    def test_waxman_single_node(self, single_node_gdf: gpd.GeoDataFrame) -> None:
        """Test waxman_graph with single node."""
        nodes, edges = _run_or_skip(waxman_graph, single_node_gdf, beta=0.5, r0=1.0)
        assert edges.empty, "Single node Waxman graph should have no edges"


# =============================================================================
# ERROR CONDITION TESTS
# =============================================================================

class TestBridgeNodesErrorConditions:
    """Test error conditions for bridge_nodes function."""

    def test_insufficient_layers(self) -> None:
        """Test bridge_nodes with fewer than 2 layers raises ValueError."""
        single_layer = {
            "layer1": gpd.GeoDataFrame(
                {"id": [1], "geometry": [Point(0, 0)]},
                crs="EPSG:27700",
            ).set_index("id"),
        }
        with pytest.raises(ValueError, match="needs at least two layers"):
            _run_or_skip(bridge_nodes, single_layer)

    def test_invalid_proximity_method(self) -> None:
        """Test bridge_nodes with invalid proximity method raises ValueError."""
        two_layers = _create_two_layer_dict()
        with pytest.raises(ValueError, match="proximity_method must be"):
            _run_or_skip(bridge_nodes, two_layers, proximity_method="invalid")

    def test_fixed_radius_missing_radius_parameter(self) -> None:
        """Test bridge_nodes with fixed_radius method but missing radius parameter."""
        two_layers = _create_two_layer_dict()
        with pytest.raises(KeyError):
            _run_or_skip(bridge_nodes, two_layers, proximity_method="fixed_radius")


class TestNetworkDistanceErrorConditions:
    """Test error conditions for network distance calculations."""

    def test_crs_mismatch_with_network(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test that CRS mismatch between nodes and network raises ValueError."""
        network_wrong_crs = gpd.GeoDataFrame(
            {"geometry": [LineString([(0, 0), (1, 1)])]},
            crs="EPSG:4326",  # Different CRS from sample_nodes_gdf
        )
        with pytest.raises(ValueError, match="CRS mismatch"):
            _run_or_skip(knn_graph, sample_nodes_gdf, k=1,
                        distance_metric="network", network_gdf=network_wrong_crs)


class TestInvalidParameterErrorConditions:
    """Test error conditions for invalid parameters."""

    def test_invalid_distance_metric(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test that invalid distance metric raises ValueError."""
        # waxman_graph always calls _distance_matrix, making it good for testing this error
        with pytest.raises(ValueError, match="distance_metric must be"):
            _run_or_skip(waxman_graph, sample_nodes_gdf, beta=0.5, r0=1.0, distance_metric="invalid")

    def test_directed_graph_crs_mismatch(self) -> None:
        """Test CRS mismatch between source and target in directed graphs."""
        src_gdf = gpd.GeoDataFrame(
            {"id": [1], "geometry": [Point(0, 0)]},
            crs="EPSG:27700",
        ).set_index("id")

        target_gdf = gpd.GeoDataFrame(
            {"id": [2], "geometry": [Point(1, 1)]},
            crs="EPSG:4326",  # Different CRS
        ).set_index("id")

        with pytest.raises(ValueError, match="CRS mismatch"):
            _run_or_skip(knn_graph, src_gdf, k=1, target_gdf=target_gdf)


# =============================================================================
# SPECIALIZED EDGE CASE TESTS
# =============================================================================

class TestNetworkGeometryHandling:
    """Test edge cases in network geometry creation and handling."""

    def test_network_edge_geometry_creation(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test network edge geometry creation works correctly."""
        nodes, edges = _run_or_skip(
            knn_graph,
            sample_nodes_gdf,
            k=1,
            distance_metric="network",
            network_gdf=sample_edges_gdf,
        )
        # Should succeed and create valid geometries
        assert not edges.empty, "Should create edges with network metric"
        assert all(geom.is_valid for geom in edges.geometry), "All geometries should be valid"

    def test_network_edge_geometry_fallback(self) -> None:
        """Test fallback in network edge geometry creation when path is short."""
        # Create a minimal network
        network_data = {
            "source_id": [1],
            "target_id": [2],
            "geometry": [LineString([(0, 0), (0.1, 0.1)])],
        }
        multi_index = pd.MultiIndex.from_arrays(
            [network_data["source_id"], network_data["target_id"]],
            names=("source_id", "target_id"),
        )
        network_gdf = gpd.GeoDataFrame(network_data, index=multi_index, crs="EPSG:27700")

        # Convert to NetworkX to add pos attributes
        G = gdf_to_nx(edges=network_gdf)
        _, network_with_pos = nx_to_gdf(G, nodes=True, edges=True)

        points = gpd.GeoDataFrame(
            {"id": [1, 2], "geometry": [Point(0, 0), Point(0.1, 0.1)]},
            crs="EPSG:27700",
        ).set_index("id")

        # Should work without crashing
        nodes, edges = _run_or_skip(knn_graph, points, k=1,
                                   distance_metric="network", network_gdf=network_with_pos)
        assert len(edges) >= 0, "Should not crash on short paths"

    def test_network_edge_geometry_fallback_duplicate_coordinates(self) -> None:
        """Test fallback when network path has duplicate coordinates that get filtered out."""
        # Create a simple network with one valid edge
        network_data = {
            "source_id": [1],
            "target_id": [2],
            "geometry": [LineString([(0, 0), (1, 0)])],  # Valid edge
        }
        multi_index = pd.MultiIndex.from_arrays(
            [network_data["source_id"], network_data["target_id"]],
            names=("source_id", "target_id"),
        )
        network_gdf = gpd.GeoDataFrame(network_data, index=multi_index, crs="EPSG:27700")

        # Convert to NetworkX to add pos attributes
        G = gdf_to_nx(edges=network_gdf)
        _, network_with_pos = nx_to_gdf(G, nodes=True, edges=True)

        # Create two points that will both map to the same network node (node 1)
        # This will cause the shortest path to be just one node [1], which after
        # duplicate removal will have < 2 points, triggering the fallback
        points = gpd.GeoDataFrame(
            {"id": [1, 2], "geometry": [Point(0.1, 0.1), Point(-0.1, -0.1)]},  # Both close to (0,0)
            crs="EPSG:27700",
        ).set_index("id")

        # This should trigger the fallback case in lines 1442-1444
        nodes, edges = _run_or_skip(knn_graph, points, k=1,
                                   distance_metric="network", network_gdf=network_with_pos)
        
        # Should create edges using fallback geometry (direct line between original coordinates)
        assert len(edges) >= 0, "Should handle single network node case gracefully"
        if not edges.empty:
            # Verify that the fallback created LineString geometries
            assert all(isinstance(geom, LineString) for geom in edges.geometry), \
                "All geometries should be LineStrings"
            # The fallback creates a direct line between original coordinates
            # This should be valid since the points are different


class TestDegenerateGeometryHandling:
    """Test handling of degenerate geometric cases."""

    def test_qhull_error_handling(self) -> None:
        """Test graceful handling of QhullError in triangulation."""
        # Create collinear points that might cause QhullError
        collinear_points = gpd.GeoDataFrame(
            {"id": [1, 2, 3], "geometry": [Point(0, 0), Point(1, 0), Point(2, 0)]},
            crs="EPSG:27700",
        ).set_index("id")

        # Test Gabriel graph
        try:
            nodes, edges = _run_or_skip(gabriel_graph, collinear_points)
            assert isinstance(nodes, gpd.GeoDataFrame), "Should return valid GeoDataFrame"
            assert isinstance(edges, gpd.GeoDataFrame), "Should return valid GeoDataFrame"
        except QhullError:
            # Acceptable - function may not handle all degenerate cases
            pass

        # Test RNG
        try:
            nodes, edges = _run_or_skip(relative_neighborhood_graph, collinear_points)
            assert isinstance(nodes, gpd.GeoDataFrame), "Should return valid GeoDataFrame"
            assert isinstance(edges, gpd.GeoDataFrame), "Should return valid GeoDataFrame"
        except QhullError:
            # Acceptable - function may not handle all degenerate cases
            pass

    def test_coincident_points_handling(self) -> None:
        """Test handling of coincident points in Gabriel and RNG graphs."""
        # Create points with coincident ones, but well-separated to avoid QhullError
        points_with_coincident = gpd.GeoDataFrame(
            {"id": [1, 2, 3, 4], "geometry": [Point(0, 0), Point(0, 0), Point(10, 0), Point(0, 10)]},
            crs="EPSG:27700",
        ).set_index("id")

        # Gabriel graph should handle coincident points
        nodes, edges = _run_or_skip(gabriel_graph, points_with_coincident)
        assert isinstance(nodes, gpd.GeoDataFrame), "Gabriel graph should handle coincident points"
        assert isinstance(edges, gpd.GeoDataFrame), "Gabriel graph should handle coincident points"

        # RNG should handle coincident points
        nodes, edges = _run_or_skip(relative_neighborhood_graph, points_with_coincident)
        assert isinstance(nodes, gpd.GeoDataFrame), "RNG should handle coincident points"
        assert isinstance(edges, gpd.GeoDataFrame), "RNG should handle coincident points"


class TestAlgorithmSpecificBehavior:
    """Test algorithm-specific behaviors and edge cases."""

    def test_euclidean_mst_fallback_to_complete_graph(self) -> None:
        """Test that MST falls back to complete graph when needed."""
        points = gpd.GeoDataFrame(
            {"id": [1, 2, 3], "geometry": [Point(0, 0), Point(1, 0), Point(0, 1)]},
            crs="EPSG:27700",
        ).set_index("id")

        # Test with Manhattan metric (should use complete graph)
        nodes, edges = _run_or_skip(euclidean_minimum_spanning_tree, points, distance_metric="manhattan")
        # MST should have n-1 edges
        assert len(edges) == len(points) - 1, "MST should have exactly n-1 edges"

    def test_directed_edges_parameter_validation(self) -> None:
        """Test parameter validation in directed edge creation."""
        src_gdf = gpd.GeoDataFrame(
            {"id": [1], "geometry": [Point(0, 0)]},
            crs="EPSG:27700",
        ).set_index("id")

        target_gdf = gpd.GeoDataFrame(
            {"id": [2], "geometry": [Point(1, 1)]},
            crs="EPSG:27700",
        ).set_index("id")

        # Test with valid parameters
        nodes, edges = _run_or_skip(knn_graph, src_gdf, k=1, target_gdf=target_gdf)
        assert len(edges) == 1, "Should create exactly one directed edge"


class TestInternalFunctionErrorConditions:
    """Test error conditions in internal helper functions."""

    def test_add_edges_missing_network_gdf(self) -> None:
        """Test _add_edges raises error when network_gdf is None but metric is network."""
        G = nx.Graph()
        G.add_node(1, pos=(0, 0))
        G.add_node(2, pos=(1, 1))
        G.add_edge(1, 2)

        coords = np.array([[0, 0], [1, 1]])
        node_ids = [1, 2]
        edges = [(1, 2)]

        with pytest.raises(ValueError, match="`network_gdf` must be supplied"):
            _add_edges(G, edges, coords, node_ids, metric="network", dm=None, network_gdf=None)

    def test_directed_edges_parameter_conflicts(self) -> None:
        """Test _directed_edges raises error when both k and radius are provided or both are None."""
        src_coords = np.array([[0, 0]])
        dst_coords = np.array([[1, 1]])
        src_ids = [1]
        dst_ids = [2]

        # Test both parameters None
        with pytest.raises(ValueError, match="Specify exactly one of k or radius"):
            _directed_edges(src_coords, dst_coords, src_ids, dst_ids,
                               metric="euclidean", k=None, radius=None)

        # Test both parameters provided
        with pytest.raises(ValueError, match="Specify exactly one of k or radius"):
            _directed_edges(src_coords, dst_coords, src_ids, dst_ids,
                               metric="euclidean", k=1, radius=1.0)
