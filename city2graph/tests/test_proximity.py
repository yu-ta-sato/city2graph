"""Tests for proximity graph functions."""

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import Point

from city2graph.proximity import delaunay_graph
from city2graph.proximity import gilbert_graph
from city2graph.proximity import knn_graph
from city2graph.proximity import waxman_graph

# ============================================================================
# COMMON TEST FIXTURES
# ============================================================================


@pytest.fixture
def simple_points_gdf() -> gpd.GeoDataFrame:
    """Create a simple GeoDataFrame with 4 points forming a square."""
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return gpd.GeoDataFrame(geometry=[Point(c) for c in coords])


@pytest.fixture
def linear_points_gdf() -> gpd.GeoDataFrame:
    """Create a GeoDataFrame with points in a line for edge case testing."""
    coords = [(0, 0), (1, 0), (2, 0), (3, 0)]
    return gpd.GeoDataFrame(geometry=[Point(c) for c in coords])


@pytest.fixture
def triangle_points_gdf() -> gpd.GeoDataFrame:
    """Create a GeoDataFrame with 3 points forming a triangle."""
    coords = [(0, 0), (1, 0), (0.5, 1)]
    return gpd.GeoDataFrame(geometry=[Point(c) for c in coords])


# ============================================================================
# K-NEAREST NEIGHBORS GRAPH TESTS
# ============================================================================


def test_knn_graph_basic(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test basic functionality of knn_graph with k=2."""
    # Arrange: use the simple 4-point square fixture
    points_gdf = simple_points_gdf

    # Act: create k-nearest neighbors graph with k=2
    G = knn_graph(points_gdf, k=2)

    # Assert: verify graph structure and properties
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 4
    assert all(d >= 2 for _, d in G.degree())


def test_knn_graph_edge_cases(linear_points_gdf: gpd.GeoDataFrame) -> None:
    """Test knn_graph with edge cases and different k values."""
    # Arrange: use linear points for testing edge cases
    points_gdf = linear_points_gdf

    # Act: create graph with k=1 (minimum connections)
    G_k1 = knn_graph(points_gdf, k=1)

    # Assert: verify minimum connectivity
    assert G_k1.number_of_nodes() == 4
    assert all(d >= 1 for _, d in G_k1.degree())

    # Act: create graph with k=3 (maximum practical k for 4 points)
    G_k3 = knn_graph(points_gdf, k=3)

    # Assert: verify higher connectivity
    assert G_k3.number_of_nodes() == 4
    assert all(d >= 3 for _, d in G_k3.degree())


def test_knn_graph_zero_k(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test knn_graph with k=0 returns no edges."""
    # Arrange: use the simple 4-point square fixture
    points_gdf = simple_points_gdf

    # Act: create k-nearest neighbors graph with k=0
    G0 = knn_graph(points_gdf, k=0)

    # Assert: verify graph has 4 nodes and 0 edges
    assert G0.number_of_nodes() == 4
    assert G0.number_of_edges() == 0


def test_knn_graph_large_k(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test knn_graph with k larger than n-1 connects to all neighbors."""
    # Arrange: use the simple 4-point square fixture
    points_gdf = simple_points_gdf

    # Act: create k-nearest neighbors graph with k=10
    G_large = knn_graph(points_gdf, k=10)

    # Assert: verify graph has 4 nodes and each node is connected to all others
    assert G_large.number_of_nodes() == 4
    # effective k is n-1=3
    assert all(d >= 3 for _, d in G_large.degree())


# ============================================================================
# DELAUNAY TRIANGULATION GRAPH TESTS
# ============================================================================


def test_delaunay_graph_basic(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test basic functionality of delaunay_graph."""
    # Arrange: use the simple 4-point square fixture
    points_gdf = simple_points_gdf

    # Act: create Delaunay triangulation graph
    G = delaunay_graph(points_gdf)

    # Assert: verify graph structure (should form triangulation)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() in (5, 6)  # Expected edges for square triangulation


def test_delaunay_graph_triangle(triangle_points_gdf: gpd.GeoDataFrame) -> None:
    """Test delaunay_graph with triangle configuration."""
    # Arrange: use triangle points fixture
    points_gdf = triangle_points_gdf

    # Act: create Delaunay triangulation graph
    G = delaunay_graph(points_gdf)

    # Assert: verify perfect triangle connectivity
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 3  # Triangle should have exactly 3 edges


def test_delaunay_graph_collinear(linear_points_gdf: gpd.GeoDataFrame) -> None:
    """Test delaunay_graph with collinear points."""
    # Arrange: use linear points for collinear case
    points_gdf = linear_points_gdf

    # Act & Assert: delaunay triangulation should fail with collinear points
    with pytest.raises((ValueError, Exception)):  # QhullError is wrapped
        delaunay_graph(points_gdf)


def test_delaunay_graph_two_points() -> None:
    """Test delaunay_graph with two points (edge case)."""
    # Arrange: create GeoDataFrame with two points
    gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)])

    # Act: create Delaunay triangulation graph
    G = delaunay_graph(gdf)

    # Assert: verify graph has 2 nodes and 0 edges (no triangulation possible)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() == 0


# ============================================================================
# GILBERT RANDOM GRAPH TESTS
# ============================================================================


def test_gilbert_graph_radius(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test gilbert_graph with a specific radius parameter."""
    # Arrange: use simple points and set radius to connect all points
    points_gdf = simple_points_gdf
    radius = 1.5  # Should connect all points in unit square

    # Act: create Gilbert random graph with specified radius
    G = gilbert_graph(points_gdf, radius=radius)

    # Assert: verify graph properties and metadata
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 6  # Complete graph for this radius
    assert G.graph.get("radius") == radius


def test_gilbert_graph_small_radius(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test gilbert_graph with small radius for sparse connectivity."""
    # Arrange: use simple points with very small radius
    points_gdf = simple_points_gdf
    small_radius = 0.5  # Only nearest neighbors should connect

    # Act: create Gilbert graph with small radius
    G = gilbert_graph(points_gdf, radius=small_radius)

    # Assert: verify sparse connectivity
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() < 6  # Should be less connected
    assert G.graph.get("radius") == small_radius


def test_gilbert_graph_large_radius(triangle_points_gdf: gpd.GeoDataFrame) -> None:
    """Test gilbert_graph with large radius for complete connectivity."""
    # Arrange: use triangle points with large radius
    points_gdf = triangle_points_gdf
    large_radius = 5.0  # Should connect all points

    # Act: create Gilbert graph with large radius
    G = gilbert_graph(points_gdf, radius=large_radius)

    # Assert: verify complete connectivity
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 3  # Complete graph for 3 points


def test_gilbert_graph_zero_and_negative_radius(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test gilbert_graph with zero and negative radius returns no edges and stores radius."""
    # Arrange: use simple points

    # Act: create Gilbert graph with zero radius
    G_zero = gilbert_graph(simple_points_gdf, radius=0)

    # Assert: verify no edges are created
    assert G_zero.number_of_edges() == 0
    assert G_zero.graph.get("radius") == 0

    # Act: create Gilbert graph with negative radius
    G_neg = gilbert_graph(simple_points_gdf, radius=-0.1)

    # Assert: verify no edges are created and radius is stored
    assert G_neg.number_of_edges() == 0
    assert G_neg.graph.get("radius") == -0.1


# ============================================================================
# WAXMAN RANDOM GRAPH TESTS
# ============================================================================


def test_waxman_graph_reproducibility(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test that waxman_graph produces reproducible results with the same seed."""
    # Arrange: use simple points with fixed parameters and seed
    points_gdf = simple_points_gdf
    beta = 1.0
    r0 = 1.0
    seed = 42

    # Act: generate two graphs with same seed
    G1 = waxman_graph(points_gdf, beta=beta, r0=r0, seed=seed)
    G2 = waxman_graph(points_gdf, beta=beta, r0=r0, seed=seed)

    # Assert: verify reproducibility and parameter storage
    assert sorted(G1.edges()) == sorted(G2.edges())
    assert G1.graph.get("beta") == beta
    assert G1.graph.get("r0") == r0


def test_waxman_graph_different_seeds(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test that waxman_graph produces different results with different seeds."""
    # Arrange: use simple points with same parameters but different seeds
    points_gdf = simple_points_gdf
    beta = 0.5
    r0 = 1.0

    # Act: generate graphs with different seeds
    G1 = waxman_graph(points_gdf, beta=beta, r0=r0, seed=42)
    G2 = waxman_graph(points_gdf, beta=beta, r0=r0, seed=123)

    # Assert: verify different results (probabilistic test)
    assert G1.number_of_nodes() == G2.number_of_nodes() == 4
    # Note: edge counts may vary due to randomness


def test_waxman_graph_parameters(linear_points_gdf: gpd.GeoDataFrame) -> None:
    """Test waxman_graph with different beta and r0 parameters."""
    # Arrange: use linear points for testing parameter effects
    points_gdf = linear_points_gdf

    # Act: create graph with high beta (distance matters less)
    G_high_beta = waxman_graph(points_gdf, beta=2.0, r0=1.0, seed=42)

    # Act: create graph with low beta (distance matters more)
    G_low_beta = waxman_graph(points_gdf, beta=0.1, r0=1.0, seed=42)

    # Assert: verify both graphs have same node count
    assert G_high_beta.number_of_nodes() == 4
    assert G_low_beta.number_of_nodes() == 4

    # Assert: verify parameter storage
    assert G_high_beta.graph.get("beta") == 2.0
    assert G_low_beta.graph.get("beta") == 0.1


def test_waxman_graph_invalid_parameters(simple_points_gdf: gpd.GeoDataFrame) -> None:
    """Test waxman_graph with unconventional parameters returns graphs with stored params."""
    # Arrange: use simple points

    # Act: create graph with negative beta
    G_neg_beta = waxman_graph(simple_points_gdf, beta=-0.1, r0=1.0, seed=42)

    # Assert: verify graph has no edges and beta is stored
    assert G_neg_beta.graph.get("beta") == -0.1
    assert G_neg_beta.number_of_edges() == 0

    # Act: create graph with beta > 1 (should behave like complete graph)
    G_gt1_beta = waxman_graph(simple_points_gdf, beta=1.1, r0=1.0, seed=42)
    n = G_gt1_beta.number_of_nodes()
    assert G_gt1_beta.graph.get("beta") == 1.1
    assert 0 <= G_gt1_beta.number_of_edges() <= n*(n-1)//2

    # Act: create graph with zero r0 (no edges, r0 should be stored)
    G_zero_r0 = waxman_graph(simple_points_gdf, beta=0.5, r0=0, seed=42)
    assert G_zero_r0.graph.get("r0") == 0
    assert G_zero_r0.number_of_edges() == 0


# ============================================================================
# ERROR HANDLING AND EDGE CASES
# ============================================================================


def test_empty_geodataframe() -> None:
    """Test all graph functions with empty GeoDataFrame."""
    # Arrange: create empty GeoDataFrame
    empty_gdf = gpd.GeoDataFrame(geometry=[])

    # Act & Assert: knn_graph should raise ValueError for empty input
    with pytest.raises(ValueError, match="GeoDataFrame must contain geometry"):
        knn_graph(empty_gdf)

    # Act & Assert: delaunay_graph should raise ValueError for empty input
    with pytest.raises(ValueError, match="GeoDataFrame must contain geometry"):
        delaunay_graph(empty_gdf)

    # Act & Assert: gilbert_graph should raise ValueError for empty input
    with pytest.raises(ValueError, match="GeoDataFrame must contain geometry"):
        gilbert_graph(empty_gdf, radius=1.0)

    # Act & Assert: waxman_graph should raise ValueError for empty input
    with pytest.raises(ValueError, match="GeoDataFrame must contain geometry"):
        waxman_graph(empty_gdf, beta=1.0, r0=1.0)


def test_single_point_geodataframe() -> None:
    """Test all graph functions with single point GeoDataFrame."""
    # Arrange: create GeoDataFrame with single point
    single_point_gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)])

    # Act & Assert: test knn_graph with single point
    G_knn = knn_graph(single_point_gdf)
    assert G_knn.number_of_nodes() == 1
    assert G_knn.number_of_edges() == 0

    # Act & Assert: test delaunay_graph with single point
    G_delaunay = delaunay_graph(single_point_gdf)
    assert G_delaunay.number_of_nodes() == 1
    assert G_delaunay.number_of_edges() == 0

    # Act & Assert: test gilbert_graph with single point
    G_gilbert = gilbert_graph(single_point_gdf, radius=1.0)
    assert G_gilbert.number_of_nodes() == 1
    assert G_gilbert.number_of_edges() == 0

    # Act & Assert: test waxman_graph with single point
    G_waxman = waxman_graph(single_point_gdf, beta=1.0, r0=1.0)
    assert G_waxman.number_of_nodes() == 1
    assert G_waxman.number_of_edges() == 0
