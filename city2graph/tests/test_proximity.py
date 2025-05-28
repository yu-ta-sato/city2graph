import pytest
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx

from city2graph.proximity import (
    knn_graph,
    delaunay_graph,
    gilbert_graph,
    waxman_graph,
)


@pytest.fixture
def simple_points_gdf():
    coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return gpd.GeoDataFrame(geometry=[Point(c) for c in coords])


def test_knn_graph_basic(simple_points_gdf):
    G = knn_graph(simple_points_gdf, k=2)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == 4
    assert all(d >= 2 for _, d in G.degree())


def test_delaunay_graph_basic(simple_points_gdf):
    G = delaunay_graph(simple_points_gdf)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() in (5, 6)


def test_gilbert_graph_radius(simple_points_gdf):
    G = gilbert_graph(simple_points_gdf, radius=1.5)
    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 6
    assert G.graph.get("radius") == 1.5


def test_waxman_graph_reproducibility(simple_points_gdf):
    G1 = waxman_graph(simple_points_gdf, beta=1.0, r0=1.0, seed=42)
    G2 = waxman_graph(simple_points_gdf, beta=1.0, r0=1.0, seed=42)
    assert sorted(G1.edges()) == sorted(G2.edges())
    assert G1.graph.get("beta") == 1.0
    assert G1.graph.get("r0") == 1.0
