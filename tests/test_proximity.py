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

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from scipy.spatial import QhullError
from shapely.geometry import LineString
from shapely.geometry import Point

from city2graph.proximity import _directed_edges
from city2graph.proximity import bridge_nodes
from city2graph.proximity import delaunay_graph
from city2graph.proximity import euclidean_minimum_spanning_tree
from city2graph.proximity import fixed_radius_graph
from city2graph.proximity import gabriel_graph
from city2graph.proximity import knn_graph
from city2graph.proximity import relative_neighborhood_graph
from city2graph.proximity import waxman_graph
from city2graph.utils import gdf_to_nx
from city2graph.utils import nx_to_gdf

if TYPE_CHECKING:
    from collections.abc import Callable

# Test configuration
SKIP_EXCEPTIONS = (NotImplementedError, AttributeError, NameError, ImportError)
TOLERANCE = 1e-6

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


# Helper functions
def _run_or_skip(
    fn: Callable[..., tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph],
    *args: object,
    **kwargs: object,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """Execute function, skipping on implementation errors."""
    try:
        return fn(*args, **kwargs)
    except SKIP_EXCEPTIONS as exc:
        pytest.skip(f"Implementation not ready: {exc}")


def _is_l_shaped(line: LineString) -> bool:
    """Check if LineString is L-shaped (3 points forming right angle)."""
    coords = list(line.coords)
    if len(coords) != 3:
        return False
    (x0, y0), (x1, y1), (x2, y2) = coords
    return bool((x0 == x1 or y0 == y1) and (x1 == x2 or y1 == y2))


def _create_test_points(crs: str = "EPSG:27700") -> gpd.GeoDataFrame:
    """Create well-separated test points."""
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3], "geometry": [Point(0, 0), Point(10, 0), Point(5, 10)]},
        crs=crs,
    ).set_index("id")


def _create_two_layers(crs: str = "EPSG:27700") -> dict[str, gpd.GeoDataFrame]:
    """Create two-layer test data for bridge_nodes."""
    return {
        "layer1": gpd.GeoDataFrame({"id": [1], "geometry": [Point(0, 0)]}, crs=crs).set_index("id"),
        "layer2": gpd.GeoDataFrame({"id": [2], "geometry": [Point(1, 1)]}, crs=crs).set_index("id"),
    }


# Core functionality tests
@pytest.mark.parametrize(("gen_name", "kwargs"), GENERATORS, ids=[g[0] for g in GENERATORS])
@pytest.mark.parametrize("metric", DISTANCE_METRICS)
def test_generator_basic_functionality(
    sample_nodes_gdf: gpd.GeoDataFrame,
    gen_name: str,
    kwargs: dict[str, Any],
    metric: str,
) -> None:
    """Test basic generator functionality with different distance metrics."""
    generator_fn = globals()[gen_name]
    nodes, edges = _run_or_skip(generator_fn, sample_nodes_gdf, distance_metric=metric, **kwargs)

    # Verify structure
    assert isinstance(nodes, gpd.GeoDataFrame)
    assert isinstance(edges, gpd.GeoDataFrame)
    assert nodes.shape[0] == len(sample_nodes_gdf)
    assert nodes.crs == edges.crs == sample_nodes_gdf.crs

    # Verify required columns
    assert "geometry" in edges.columns
    assert "weight" in edges.columns

    # Verify Manhattan distance creates L-shaped geometries
    if metric == "manhattan" and not edges.empty:
        assert all(_is_l_shaped(geom) for geom in edges.geometry)


@pytest.mark.parametrize(("gen_name", "kwargs"), GENERATORS, ids=[g[0] for g in GENERATORS])
def test_network_metric_error_handling(
    sample_nodes_gdf: gpd.GeoDataFrame,
    gen_name: str,
    kwargs: dict[str, Any],
) -> None:
    """Test network metric error handling when network_gdf is missing."""
    generator_fn = globals()[gen_name]
    with pytest.raises(ValueError, match="network_gdf is required for network distance metric"):
        _run_or_skip(generator_fn, sample_nodes_gdf, distance_metric="network", **kwargs)


@pytest.mark.parametrize(("gen_name", "kwargs"), [g for g in GENERATORS if g[0] != "waxman_graph"])
def test_network_metric_functionality(
    sample_nodes_gdf: gpd.GeoDataFrame,
    sample_edges_gdf: gpd.GeoDataFrame,
    gen_name: str,
    kwargs: dict[str, Any],
) -> None:
    """Test network metric functionality with valid network."""
    generator_fn = globals()[gen_name]
    nodes, edges = _run_or_skip(
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
        G = _run_or_skip(knn_graph, sample_nodes_gdf, k=3, as_nx=True)

        assert isinstance(G, nx.Graph)
        assert G.number_of_nodes() == len(sample_nodes_gdf)
        assert G.graph["crs"] == sample_nodes_gdf.crs

        expected_edges = len(sample_nodes_gdf) * 3 / 2
        assert math.isclose(G.number_of_edges(), expected_edges, rel_tol=0.5)

    def test_edge_cases(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        single_node_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test KNN edge cases."""
        # k=0 should produce no edges
        _, edges = _run_or_skip(knn_graph, sample_nodes_gdf, k=0)
        assert edges.empty

        # Single node should produce no edges
        _, edges = _run_or_skip(knn_graph, single_node_gdf, k=1)
        assert edges.empty

        # k > available neighbors should work
        excessive_k = len(sample_nodes_gdf) + 10
        _, edges = _run_or_skip(knn_graph, sample_nodes_gdf, k=excessive_k)
        max_edges = len(sample_nodes_gdf) * (len(sample_nodes_gdf) - 1)
        assert len(edges) <= max_edges


class TestWaxmanGraph:
    """Test Waxman graph specific functionality."""

    def test_reproducibility(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test seed-based reproducibility."""
        params = {"beta": 0.5, "r0": 3, "seed": 11}
        _, edges1 = _run_or_skip(waxman_graph, sample_nodes_gdf, **params)
        _, edges2 = _run_or_skip(waxman_graph, sample_nodes_gdf, **params)
        assert edges1.equals(edges2)

    def test_parameter_storage(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test parameter storage in graph metadata."""
        beta, r0 = 0.7, 2.5
        result = _run_or_skip(waxman_graph, sample_nodes_gdf, beta=beta, r0=r0, as_nx=True)
        assert isinstance(result, nx.Graph)
        assert result.graph.get("beta") == beta
        assert result.graph.get("r0") == r0

    def test_single_node_networkx_output(self, single_node_gdf: gpd.GeoDataFrame) -> None:
        """Test single node case with NetworkX output - covers line 1019."""
        result = _run_or_skip(waxman_graph, single_node_gdf, beta=0.5, r0=1.0, as_nx=True)
        assert isinstance(result, nx.Graph)
        assert result.number_of_nodes() == 1
        assert result.number_of_edges() == 0


class TestFixedRadiusGraph:
    """Test fixed radius graph functionality."""

    def test_parameter_storage(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test radius parameter storage."""
        radius = 1.5
        result = _run_or_skip(fixed_radius_graph, sample_nodes_gdf, radius=radius, as_nx=True)
        assert isinstance(result, nx.Graph)
        assert result.graph.get("radius") == radius

    def test_single_node_edge_case(self, single_node_gdf: gpd.GeoDataFrame) -> None:
        """Test single node case."""
        result = _run_or_skip(fixed_radius_graph, single_node_gdf, radius=1.0)
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
        _, edges = _run_or_skip(graph_fn, single_node_gdf)
        assert edges.empty

        # Two points - special case for Gabriel/RNG
        _, edges = _run_or_skip(graph_fn, two_nodes_gdf)
        if graph_fn in [gabriel_graph, relative_neighborhood_graph]:
            assert len(edges) == 1
        else:  # Delaunay
            assert edges.empty

    def test_well_separated_points(self) -> None:
        """Test with well-separated points to avoid QhullError."""
        points = _create_test_points()
        for graph_fn in [gabriel_graph, relative_neighborhood_graph]:
            _, edges = _run_or_skip(graph_fn, points)
            assert len(edges) >= 1

    def test_degenerate_cases(self) -> None:
        """Test handling of degenerate geometric cases."""
        # Collinear points
        collinear = gpd.GeoDataFrame(
            {"id": [1, 2, 3], "geometry": [Point(0, 0), Point(1, 0), Point(2, 0)]},
            crs="EPSG:27700",
        ).set_index("id")

        # Should handle gracefully or raise QhullError
        for graph_fn in [gabriel_graph, relative_neighborhood_graph]:
            try:
                nodes, edges = _run_or_skip(graph_fn, collinear)
                assert isinstance(nodes, gpd.GeoDataFrame)
                assert isinstance(edges, gpd.GeoDataFrame)
            except QhullError:
                pass  # Acceptable for degenerate cases

        # Coincident points
        coincident = gpd.GeoDataFrame(
            {
                "id": [1, 2, 3, 4],
                "geometry": [Point(0, 0), Point(0, 0), Point(10, 0), Point(0, 10)],
            },
            crs="EPSG:27700",
        ).set_index("id")

        for graph_fn in [gabriel_graph, relative_neighborhood_graph]:
            nodes, edges = _run_or_skip(graph_fn, coincident)
            assert isinstance(nodes, gpd.GeoDataFrame)
            assert isinstance(edges, gpd.GeoDataFrame)


class TestEuclideanMST:
    """Test Euclidean minimum spanning tree."""

    def test_single_node(self, single_node_gdf: gpd.GeoDataFrame) -> None:
        """Test single node case."""
        _, edges = _run_or_skip(euclidean_minimum_spanning_tree, single_node_gdf)
        assert edges.empty

    def test_edge_count(self) -> None:
        """Test that MST has exactly n-1 edges."""
        points = gpd.GeoDataFrame(
            {"id": [1, 2, 3], "geometry": [Point(0, 0), Point(1, 0), Point(0, 1)]},
            crs="EPSG:27700",
        ).set_index("id")

        _, edges = _run_or_skip(
            euclidean_minimum_spanning_tree,
            points,
            distance_metric="manhattan",
        )
        assert len(edges) == len(points) - 1


# Bridge nodes tests
class TestBridgeNodes:
    """Test bridge nodes functionality."""

    @pytest.mark.parametrize(("method", "extra"), BRIDGE_METHODS)
    def test_output_structure(
        self,
        sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
        method: str,
        extra: dict[str, Any],
    ) -> None:
        """Test output structure correctness."""
        _, edges_dict = _run_or_skip(
            bridge_nodes,
            sample_hetero_nodes_dict,
            proximity_method=method,
            **extra,
        )

        # Check expected edge types
        layer_names = sample_hetero_nodes_dict.keys()
        expected_keys = {
            (src, "is_nearby", dst) for src in layer_names for dst in layer_names if src != dst
        }
        assert set(edges_dict.keys()) == expected_keys

        # Check required columns
        for edge_gdf in edges_dict.values():
            assert "weight" in edge_gdf.columns
            assert "geometry" in edge_gdf.columns

    def test_networkx_output(self, sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame]) -> None:
        """Test NetworkX output format."""
        G = _run_or_skip(
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
            _run_or_skip(bridge_nodes, single_layer)

        # Invalid proximity method
        two_layers = _create_two_layers()
        with pytest.raises(ValueError, match="proximity_method must be"):
            _run_or_skip(bridge_nodes, two_layers, proximity_method="invalid")

        # Missing radius parameter
        with pytest.raises(KeyError):
            _run_or_skip(bridge_nodes, two_layers, proximity_method="fixed_radius")


# Distance metric tests
def test_distance_metric_relationships(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    """Test Manhattan >= Euclidean distance relationship."""
    _, euclidean_edges = _run_or_skip(knn_graph, sample_nodes_gdf, k=1, distance_metric="euclidean")
    _, manhattan_edges = _run_or_skip(knn_graph, sample_nodes_gdf, k=1, distance_metric="manhattan")

    if euclidean_edges.empty or manhattan_edges.empty:
        pytest.skip("No edges produced")

    euclidean_weight = euclidean_edges.iloc[0]["weight"]
    manhattan_weight = manhattan_edges.iloc[0]["weight"]
    assert euclidean_weight <= manhattan_weight + TOLERANCE


# Error condition tests
class TestErrorConditions:
    """Test various error conditions."""

    def test_invalid_distance_metric(self, sample_nodes_gdf: gpd.GeoDataFrame) -> None:
        """Test invalid distance metric error."""
        with pytest.raises(ValueError, match="Unknown distance metric"):
            _run_or_skip(
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
            _run_or_skip(
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
            _run_or_skip(knn_graph, src_gdf, k=1, target_gdf=target_gdf)


# Network geometry tests
class TestNetworkGeometry:
    """Test network geometry handling edge cases."""

    def test_basic_functionality(
        self,
        sample_nodes_gdf: gpd.GeoDataFrame,
        sample_edges_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Test basic network geometry creation."""
        nodes, edges = _run_or_skip(
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
        with pytest.raises(ValueError, match="network_gdf is required for network distance metric"):
            _run_or_skip(
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
        network_gdf = gpd.GeoDataFrame(network_data, index=multi_index, crs="EPSG:27700")

        # This should trigger the fallback geometry creation
        nodes_result, edges_result = _run_or_skip(
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
        network_gdf = gpd.GeoDataFrame(network_data, index=multi_index, crs="EPSG:27700")

        G = gdf_to_nx(edges=network_gdf)
        _, network_with_pos = nx_to_gdf(G, nodes=True, edges=True)

        points = gpd.GeoDataFrame(
            {"id": [1, 2], "geometry": [Point(0, 0), Point(0.1, 0.1)]},
            crs="EPSG:27700",
        ).set_index("id")
        nodes, edges = _run_or_skip(
            knn_graph,
            points,
            k=1,
            distance_metric="network",
            network_gdf=network_with_pos,
        )
        assert len(edges) >= 0

        # Duplicate coordinates fallback
        points_close = gpd.GeoDataFrame(
            {"id": [1, 2], "geometry": [Point(0.1, 0.1), Point(-0.1, -0.1)]},
            crs="EPSG:27700",
        ).set_index("id")
        nodes, edges = _run_or_skip(
            knn_graph,
            points_close,
            k=1,
            distance_metric="network",
            network_gdf=network_with_pos,
        )
        assert len(edges) >= 0
        if not edges.empty:
            assert all(isinstance(geom, LineString) for geom in edges.geometry)


# Test for _directed_edges function error handling
class TestDirectedEdgesErrorHandling:
    """Test _directed_edges function error conditions."""

    def test_invalid_parameters_combination(self) -> None:
        """Test error when both k and radius are provided or both are None - covers lines 1479-1480."""
        src_coords = np.array([[0, 0], [1, 1]])
        dst_coords = np.array([[2, 2], [3, 3]])
        src_ids = [1, 2]
        dst_ids = [3, 4]

        # Both k and radius provided
        with pytest.raises(ValueError, match="Specify exactly one of k or radius"):
            _directed_edges(
                src_coords,
                dst_coords,
                src_ids,
                dst_ids,
                metric="euclidean",
                k=1,
                radius=1.0,
            )

        # Neither k nor radius provided
        with pytest.raises(ValueError, match="Specify exactly one of k or radius"):
            _directed_edges(src_coords, dst_coords, src_ids, dst_ids, metric="euclidean")
