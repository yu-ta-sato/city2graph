"""
Module for generating proximity-based graphs from geospatial data.

This module provides functions to create k-nearest neighbor and Delaunay
triangulation graphs from GeoDataFrame geometries.
"""

from itertools import combinations

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

__all__ = ["knn_graph", "delaunay_graph"]


def _build_knn_edges(indices: np.ndarray, node_indices: list | None = None) -> list[tuple]:
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
        return [(node_indices[i], node_indices[j]) 
                for i, neighbors in enumerate(indices) 
                for j in neighbors[1:]]  # Skip self (first neighbor)
    else:
        return [(i, j) 
                for i, neighbors in enumerate(indices) 
                for j in neighbors[1:]]  # Skip self (first neighbor)


def _build_delaunay_edges(coords: np.ndarray, node_indices: list) -> set[tuple]:
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
    tri = Delaunay(coords)
    return {(node_indices[i], node_indices[j]) 
            for simplex in tri.simplices 
            for i, j in combinations(simplex, 2)}


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
        for idx, (geom, centroid) in zip(gdf.index, zip(gdf.geometry, centroids))
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
        raise TypeError("Input data must be a GeoDataFrame.")
    if not hasattr(data, "geometry") or data.geometry.isna().all():
        raise ValueError("GeoDataFrame must contain geometry.")
    
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


def knn_graph(gdf: gpd.GeoDataFrame, k: int = 5) -> nx.Graph:
    """
    Generate k-nearest neighbor graph from points or polygon centroids.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input data as a GeoDataFrame. Centroids of geometries are used.
    k : int, default 5
        Number of nearest neighbors to connect to each node.

    Returns
    -------
    networkx.Graph
        Graph with nodes and k-nearest neighbor edges.
        Node attributes include original data and 'pos' coordinates.
    """
    graph, coords, node_indices = _init_graph_and_nodes(gdf)
    
    # Early return for edge cases
    if k == 0 or coords is None or len(coords) <= 1:
        return graph
    
    # Build k-nearest neighbor relationships
    n_neighbors = min(k + 1, len(coords))  # +1 to include self
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nbrs.fit(coords)
    _, indices = nbrs.kneighbors(coords)
    
    # Add edges to graph
    edges = _build_knn_edges(indices, node_indices)
    graph.add_edges_from(edges)
    
    return graph


def delaunay_graph(gdf: gpd.GeoDataFrame) -> nx.Graph:
    """
    Generate Delaunay graph from points or polygon centroids.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input data as a GeoDataFrame. Centroids of geometries are used.

    Returns
    -------
    networkx.Graph
        Graph with nodes and Delaunay triangulation edges.
        Node attributes include original data and 'pos' coordinates.
    """
    graph, coords, node_indices = _init_graph_and_nodes(gdf)
    
    # Early return for insufficient points
    if coords is None or len(coords) < 3:
        return graph
    
    # Build Delaunay triangulation edges
    edges = _build_delaunay_edges(coords, node_indices)
    graph.add_edges_from(edges)
    
    return graph
