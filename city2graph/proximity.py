"""
Module for processing contiguity networks.
"""

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors
import geopandas as gpd

__all__ = ['knn_graph']


def _extract_coords_and_attrs_from_gdf(gdf):
    """Extract centroid coordinates and prepare node attributes from GeoDataFrame."""
    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])
    
    # Prepare node attributes dictionary
    node_attrs = {}
    for idx, (geom, centroid) in zip(gdf.index, zip(gdf.geometry, centroids)):
        node_attrs[idx] = {
            'geometry': geom,
            'pos': (centroid.x, centroid.y)
        }
    
    return coords, node_attrs


def _build_knn_edges(indices, node_indices=None):
    """Build k-nearest neighbor edges."""
    edges = []
    
    if node_indices is not None:
        # For GeoDataFrame with custom indices
        for i, neighbors in enumerate(indices):
            source_node = node_indices[i]
            for neighbor_idx in neighbors[1:]:  # Skip self
                target_node = node_indices[neighbor_idx]
                edges.append((source_node, target_node))
    else:
        # For coordinate arrays with sequential indices
        for i, neighbors in enumerate(indices):
            for neighbor_idx in neighbors[1:]:  # Skip self
                edges.append((i, neighbor_idx))
    
    return edges


def knn_graph(data, k=5):
    """
    Generate k-nearest neighbor graph from points or polygon centroids.
    
    Parameters
    ----------
    data : geopandas.GeoDataFrame
        Input data as a GeoDataFrame. Centroids of geometries are used.
    k : int, default 5
        Number of nearest neighbors to connect to each node.
        
    Returns
    -------
    networkx.Graph
        Graph with nodes and k-nearest neighbor edges.
        Node attributes include original data and 'pos' coordinates.
    """
    # Validate input data type
    if isinstance(data, gpd.GeoDataFrame):
        if not hasattr(data, 'geometry') or data.geometry.isna().all():
            raise ValueError("GeoDataFrame must contain geometry.")
        
    elif isinstance(data, (list, np.ndarray)):
        data = gpd.GeoDataFrame(geometry=gpd.points_from_xy(data[:, 0], data[:, 1]))

    else:
        raise TypeError("Input data must be a GeoDataFrame of coordinates.")
    
    # Initialize an empty graph
    G = nx.Graph()
    
    # Early return for empty GeoDataFrame
    if data.empty:
        return G
        
    # Add CRS as graph attribute
    if data.crs is not None:
        G.graph['crs'] = data.crs
    
    # Use vectorized function to extract coordinates and attributes
    coords, node_attrs = _extract_coords_and_attrs_from_gdf(data)
    node_indices = list(data.index)
    
    # Add nodes with attributes in batch
    G.add_nodes_from(node_indices)
    
    # Set node attributes
    nx.set_node_attributes(G, node_attrs)
                    
    # Early return for edge cases
    if k == 0 or len(coords) <= 1:
        return G
    
    # Vectorized k-nearest neighbor computation
    n_neighbors = min(k + 1, len(coords))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
    nbrs.fit(coords)
    _, indices = nbrs.kneighbors(coords)
    
    # Use vectorized edge creation
    edges = _build_knn_edges(indices, node_indices)
    
    # Batch add edges
    G.add_edges_from(edges)
    
    return G