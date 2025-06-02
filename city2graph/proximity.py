"""
Module for generating proximity-based graphs from geospatial data.

This module provides functions to create k-nearest neighbor and Delaunay
triangulation graphs from GeoDataFrame geometries.
"""

from itertools import combinations

import geopandas as gpd
import networkx as nx
import numpy as np
import scipy.spatial.qhull
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors

from city2graph.utils import nx_to_gdf

__all__ = ["delaunay_graph", "gilbert_graph", "knn_graph", "waxman_graph"]


def _build_knn_edges(indices: np.ndarray,
                     node_indices: list | None = None) -> list[tuple]:
    """
    Build k-nearest neighbor edges from indices array.

    Parameters
    ----------
    indices : np.ndarray
        Array of neighbor indices for each node.
    node_indices : list | None, optional
        List mapping array indices to actual node IDs. If None, uses array indices.

    Returns
    -------
    list[tuple]
        List of edge tuples (source, target).
    """
    if node_indices is not None:
        return [
            (node_indices[i], node_indices[j])
            for i, neighbors in enumerate(indices)
            for j in neighbors[1:]
        ]  # Skip self (first neighbor)
    return [
        (i, j) for i, neighbors in enumerate(indices) for j in neighbors[1:]
    ]  # Skip self (first neighbor)


def _build_delaunay_edges(coords: np.ndarray,
                          node_indices: list) -> set[tuple]:
    """
    Build Delaunay triangulation edges from coordinates.

    Parameters
    ----------
    coords : np.ndarray
        Array of (x, y) coordinates.
    node_indices : list
        List of node IDs corresponding to coordinates.

    Returns
    -------
    set[tuple]
        Set of unique edge tuples (source, target).
    """
    try:
        tri = Delaunay(coords)
        return {
            (node_indices[i], node_indices[j])
            for simplex in tri.simplices
            for i, j in combinations(simplex, 2)
        }
    except scipy.spatial.qhull.QhullError:
        # Handle collinear points or other geometric issues
        return set()


def _extract_coords_and_attrs_from_gdf(gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, dict]:
    """
    Extract centroid coordinates and prepare node attributes from GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with geometry column.

    Returns
    -------
    tuple[np.ndarray, dict]
        Coordinate array and node attributes dictionary.
    """
    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])

    # Vectorized node attributes preparation
    node_attrs = {
        idx: {"geometry": geom, "pos": (centroid.x, centroid.y)}
        for idx, (geom, centroid) in zip(gdf.index, zip(gdf.geometry, centroids, strict=False), strict=False)
    }

    return coords, node_attrs


def _init_graph_and_nodes(data: gpd.GeoDataFrame) -> tuple[nx.Graph, np.ndarray | None, list | None]:
    """
    Initialize graph and extract nodes from GeoDataFrame.

    Validates input, creates NetworkX graph with CRS, extracts coordinates
    and node attributes, then adds nodes to the graph.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Input GeoDataFrame with geometry column.

    Returns
    -------
    tuple[nx.Graph, np.ndarray | None, list | None]
        Initialized graph, coordinate array, and node indices.
        Returns None values for coords and indices if data is empty.

    Raises
    ------
    TypeError
        If input is not a GeoDataFrame.
    ValueError
        If GeoDataFrame lacks geometry or all geometries are null.
    """
    if not isinstance(data, gpd.GeoDataFrame):
        msg = "Input data must be a GeoDataFrame."
        raise TypeError(msg)
    if not hasattr(data, "geometry") or data.geometry.isna().all():
        msg = "GeoDataFrame must contain geometry."
        raise ValueError(msg)

    # Initialize graph with CRS
    G = nx.Graph()
    if data.crs is not None:
        G.graph["crs"] = data.crs

    # Handle empty data
    if data.empty:
        return G, None, None

    # Extract coordinates and attributes, add nodes to graph
    coords, node_attrs = _extract_coords_and_attrs_from_gdf(data)
    node_indices = list(data.index)
    G.add_nodes_from(node_indices)
    nx.set_node_attributes(G, node_attrs)

    return G, coords, node_indices


def knn_graph(gdf: gpd.GeoDataFrame,
              k: int = 5,
              as_gdf: bool = False) -> nx.Graph | gpd.GeoDataFrame:
    """
    Generate k-nearest neighbor graph from points or polygon centroids.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input data as a GeoDataFrame. Centroids of geometries are used.
    k : int, default 5
        Number of nearest neighbors to connect to each node.
    as_gdf : bool, default False
        If True, return edges as a GeoDataFrame instead of NetworkX graph.

    Returns
    -------
    networkx.Graph or geopandas.GeoDataFrame
        Graph with nodes and k-nearest neighbor edges.
        Node attributes include original data and 'pos' coordinates.
        If as_gdf=True, returns GeoDataFrame of edges.
    """
    graph, coords, node_indices = _init_graph_and_nodes(gdf)

    # Early return for edge cases
    if k == 0 or coords is None or len(coords) <= 1:
        if as_gdf:
            return nx_to_gdf(graph, edges=True)
        return graph

    # Build k-nearest neighbor relationships
    n_neighbors = min(k + 1, len(coords))  # +1 to include self
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nbrs.fit(coords)
    _, indices = nbrs.kneighbors(coords)

    # Add edges to graph
    edges = _build_knn_edges(indices, node_indices)
    graph.add_edges_from(edges)

    if as_gdf:
        return nx_to_gdf(graph, edges=True)
    return graph


def delaunay_graph(gdf: gpd.GeoDataFrame,
                   as_gdf: bool = False) -> nx.Graph | gpd.GeoDataFrame:
    """
    Generate Delaunay graph from points or polygon centroids.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input data as a GeoDataFrame. Centroids of geometries are used.
    as_gdf : bool, default False
        If True, return edges as a GeoDataFrame instead of NetworkX graph.

    Returns
    -------
    networkx.Graph or geopandas.GeoDataFrame
        Graph with nodes and Delaunay triangulation edges.
        Node attributes include original data and 'pos' coordinates.
        If as_gdf=True, returns GeoDataFrame of edges.
    """
    graph, coords, node_indices = _init_graph_and_nodes(gdf)

    # Early return for insufficient points
    if coords is None or len(coords) < 3:
        if as_gdf:
            return nx_to_gdf(graph, edges=True)
        return graph

    # Build Delaunay triangulation edges
    edges = _build_delaunay_edges(coords, node_indices)
    graph.add_edges_from(edges)

    if as_gdf:
        return nx_to_gdf(graph, edges=True)
    return graph


def gilbert_graph(gdf: gpd.GeoDataFrame,
                  radius: float,
                  as_gdf: bool = False) -> nx.Graph | gpd.GeoDataFrame:
    """
    Generate Gilbert disc model graph from GeoDataFrame point geometries.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame with point geometries.
    radius : float
        Connection radius.
    as_gdf : bool, default False
        If True, return edges as a GeoDataFrame instead of NetworkX graph.

    Returns
    -------
    networkx.Graph or geopandas.GeoDataFrame
        Graph with nodes and edges connecting points within radius.
        If as_gdf=True, returns GeoDataFrame of edges.
    """
    graph, coords, node_indices = _init_graph_and_nodes(gdf)
    if coords is None or len(coords) < 2:
        if as_gdf:
            return nx_to_gdf(graph, edges=True)
        return graph

    # Vectorized distance computation and edge creation
    dists = squareform(pdist(coords))
    edge_mask = np.triu(dists <= radius, k=1)
    edge_indices = np.nonzero(edge_mask)

    # Convert to node indices without loops
    edges = [
        (node_indices[i], node_indices[j])
        for i, j in zip(edge_indices[0], edge_indices[1], strict=False)
    ]

    graph.add_edges_from(edges)
    graph.graph["radius"] = radius
    
    if as_gdf:
        return nx_to_gdf(graph, edges=True)
    return graph


def waxman_graph(
    gdf: gpd.GeoDataFrame,
    beta: float,
    r0: float,
    seed: int | None = None,
    as_gdf: bool = False) -> nx.Graph | gpd.GeoDataFrame:
    r"""
    Generate Waxman random geometric graph with $H_{ij} = \beta e^{-d_{ij}/r_0}}$.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame with point geometries.
    beta : float
        Probability scale factor.
    r0 : float
        Euclidean distance scale factor.
    seed : int | None, optional
        Random seed for reproducibility.
    as_gdf : bool, default False
        If True, return edges as a GeoDataFrame instead of NetworkX graph.

    Returns
    -------
    networkx.Graph or geopandas.GeoDataFrame
        Stochastic Waxman graph with 'beta' and 'r0' in graph attributes.
        If as_gdf=True, returns GeoDataFrame of edges.
    """
    graph, coords, node_indices = _init_graph_and_nodes(gdf)
    if coords is None or len(coords) < 2:
        if as_gdf:
            return nx_to_gdf(graph, edges=True)
        return graph

    # Vectorized distance computation and probability calculation
    dists = squareform(pdist(coords))
    probs = beta * np.exp(-dists / r0)
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    random_matrix = rng.random(probs.shape)

    # Create upper triangular mask for undirected edges
    edge_mask = np.triu(random_matrix <= probs, k=1)
    edge_indices = np.nonzero(edge_mask)

    # Convert to node indices without loops
    edges = [
        (node_indices[i], node_indices[j])
        for i, j in zip(edge_indices[0], edge_indices[1], strict=False)
    ]

    graph.add_edges_from(edges)
    graph.graph["beta"] = beta
    graph.graph["r0"] = r0
    
    if as_gdf:
        return nx_to_gdf(graph, edges=True)
    return graph
