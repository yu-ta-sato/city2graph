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
from shapely.geometry import LineString
from sklearn.neighbors import NearestNeighbors

from .utils import gdf_to_nx
from .utils import nx_to_gdf

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


def _validate_network_compatibility(gdf: gpd.GeoDataFrame, 
                                   network_gdf: gpd.GeoDataFrame) -> None:
    """
    Validate that the input GeoDataFrame and network have compatible CRS.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with point/polygon geometries.
    network_gdf : gpd.GeoDataFrame
        Network GeoDataFrame for distance calculations.

    Raises
    ------
    ValueError
        If CRS don't match or network is invalid.
    """
    if gdf.crs != network_gdf.crs:
        msg = f"CRS mismatch: input data CRS {gdf.crs} != network CRS {network_gdf.crs}"
        raise ValueError(msg)

    if network_gdf.empty:
        msg = "Network GeoDataFrame is empty"
        raise ValueError(msg)


def _get_network_positions(network_graph: nx.Graph) -> dict:
    """Extract node positions from network graph."""
    pos_dict = nx.get_node_attributes(network_graph, "pos")
    if not pos_dict:
        # Fallback: use node coordinates if available
        node_attrs = dict(network_graph.nodes(data=True))
        pos_dict = {
            node_id: (attrs.get("x", 0), attrs.get("y", 0))
            for node_id, attrs in node_attrs.items()
            if "x" in attrs and "y" in attrs
        }

    if not pos_dict:
        msg = "Network graph missing node position information"
        raise ValueError(msg)

    return pos_dict


def _find_nearest_network_nodes(coords: np.ndarray,
                               network_graph: nx.Graph) -> list:
    """Find nearest network nodes for each input coordinate."""
    pos_dict = _get_network_positions(network_graph)

    # Create network coordinates array
    network_coords = np.array(list(pos_dict.values()))
    network_node_ids = list(pos_dict.keys())

    # Find nearest network nodes using sklearn
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nbrs.fit(network_coords)
    _, indices = nbrs.kneighbors(coords)

    return [network_node_ids[idx[0]] for idx in indices]


def _compute_network_distances(coords: np.ndarray,
                              node_indices: list,
                              network_graph: nx.Graph) -> tuple[np.ndarray, list]:
    """
    Compute network distances between all pairs of points.

    Returns
    -------
    tuple[np.ndarray, list]
        Distance matrix and list of nearest network nodes.
    """
    nearest_network_nodes = _find_nearest_network_nodes(coords, network_graph)

    # Initialize distance matrix
    n_points = len(node_indices)
    distance_matrix = np.full((n_points, n_points), np.inf)
    np.fill_diagonal(distance_matrix, 0)

    # Check if edges have length attribute
    has_length = any("length" in attrs for _, _, attrs in network_graph.edges(data=True))

    # Precompute distances from all unique network nodes
    unique_nodes = list(set(nearest_network_nodes))
    all_distances = {}

    for source_node in unique_nodes:
        try:
            if has_length:
                distances = nx.single_source_dijkstra_path_length(
                    network_graph, source_node, weight="length"
                )
            else:
                distances = nx.single_source_shortest_path_length(
                    network_graph, source_node
                )
            all_distances[source_node] = distances
        except nx.NetworkXNoPath:
            all_distances[source_node] = {}

    # Fill distance matrix using precomputed distances (vectorized)
    for i in range(n_points):
        source_net_node = nearest_network_nodes[i]
        source_distances = all_distances.get(source_net_node, {})
        if source_distances:
            # Vectorized assignment for all targets at once
            target_nodes = nearest_network_nodes[i + 1:]
            target_indices = np.arange(i + 1, n_points)

            # Get distances for all targets
            target_dists = np.array([
                source_distances.get(target_node, np.inf) 
                for target_node in target_nodes
            ])

            # Assign to both upper and lower triangular parts
            distance_matrix[i, target_indices] = target_dists
            distance_matrix[target_indices, i] = target_dists

    return distance_matrix, nearest_network_nodes


def _setup_network_computation(gdf: gpd.GeoDataFrame,
                              network_gdf: gpd.GeoDataFrame,
                              coords: np.ndarray,
                              node_indices: list) -> tuple[nx.Graph, np.ndarray, list]:
    """
    Set up network computation by validating network and computing distances.

    Returns
    -------
    tuple[nx.Graph, np.ndarray, list]
        Network graph, distance matrix, and nearest network nodes.
    """
    _validate_network_compatibility(gdf, network_gdf)
    network_graph = gdf_to_nx(edges=network_gdf)
    distance_matrix, nearest_network_nodes = _compute_network_distances(
        coords, node_indices, network_graph
    )
    return network_graph, distance_matrix, nearest_network_nodes


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
              distance_metric: str = "euclidean",
              network_gdf: gpd.GeoDataFrame | None = None,
              as_gdf: bool = False) -> nx.Graph | gpd.GeoDataFrame:
    """
    Generate k-nearest neighbor graph from points or polygon centroids.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input data as a GeoDataFrame. Centroids of geometries are used.
    k : int, default 5
        Number of nearest neighbors to connect to each node.
    distance_metric : str, default "euclidean"
        Distance metric to use. Options: "euclidean", "manhattan", or "network".
    network_gdf : geopandas.GeoDataFrame, optional
        Network GeoDataFrame for network distance calculations.
        Required when distance_metric="network".
    as_gdf : bool, default False
        If True, return edges as GeoDataFrame instead of NetworkX graph.

    Returns
    -------
    networkx.Graph or geopandas.GeoDataFrame
        Graph with nodes and k-nearest neighbor edges.
        If as_gdf=True, returns GeoDataFrame of edges.
        Node attributes include original data and 'pos' coordinates.
    """
    graph, coords, node_indices = _init_graph_and_nodes(gdf)

    # Early return for edge cases
    if k == 0 or coords is None or len(coords) <= 1:
        return graph

    # Initialize variables
    network_graph = None
    nearest_network_nodes = None

    if distance_metric == "network":
        if network_gdf is None:
            msg = "network_gdf is required when distance_metric='network'"
            raise ValueError(msg)

        # Setup network computation
        network_graph, distance_matrix, nearest_network_nodes = _setup_network_computation(
            gdf, network_gdf, coords, node_indices
        )

        # Find k nearest neighbors based on network distances
        k_nearest_indices = np.argsort(distance_matrix, axis=1)[:, :k+1]

        # Create edges list efficiently (vectorized)
        # Get all valid edges at once using vectorized operations
        valid_mask = distance_matrix < np.inf
        node_pairs = []
        for i in range(len(node_indices)):
            valid_neighbors = k_nearest_indices[i, 1:k+1][valid_mask[i, k_nearest_indices[i, 1:k+1]]]
            node_pairs.extend([(node_indices[i], node_indices[j]) for j in valid_neighbors])
        edges = node_pairs
    elif distance_metric == "manhattan":
        # Manhattan distance
        n_neighbors = min(k + 1, len(coords))  # +1 to include self
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto", metric="manhattan")
        nbrs.fit(coords)
        _, indices = nbrs.kneighbors(coords)
        edges = _build_knn_edges(indices, node_indices)
    else:
        # Euclidean distance (original implementation)
        n_neighbors = min(k + 1, len(coords))  # +1 to include self
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
        nbrs.fit(coords)
        _, indices = nbrs.kneighbors(coords)
        edges = _build_knn_edges(indices, node_indices)

    # Add edges to graph
    graph.add_edges_from(edges)

    # Add edge geometries
    _add_edge_geometries(graph, coords, node_indices, distance_metric,
                        network_graph, nearest_network_nodes)

    # Return as GeoDataFrame if requested
    if as_gdf:
        return nx_to_gdf(graph, edges=True)

    return graph


def delaunay_graph(gdf: gpd.GeoDataFrame,
                   distance_metric: str = "euclidean",
                   network_gdf: gpd.GeoDataFrame | None = None,
                   as_gdf: bool = False) -> nx.Graph | gpd.GeoDataFrame:
    """
    Generate Delaunay graph from points or polygon centroids.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input data as a GeoDataFrame. Centroids of geometries are used.
    distance_metric : str, default "euclidean"
        Distance metric to use. Options: "euclidean", "manhattan", or "network".
        Network metric only affects edge weights, not topology.
    network_gdf : geopandas.GeoDataFrame, optional
        Network GeoDataFrame for network distance calculations.
        Required when distance_metric="network".
    as_gdf : bool, default False
        If True, return edges as GeoDataFrame instead of NetworkX graph.

    Returns
    -------
    networkx.Graph or geopandas.GeoDataFrame
        Graph with nodes and Delaunay triangulation edges.
        If as_gdf=True, returns GeoDataFrame of edges.
        Node attributes include original data and 'pos' coordinates.
    """
    graph, coords, node_indices = _init_graph_and_nodes(gdf)

    # Early return for insufficient points
    if coords is None or len(coords) < 3:
        return graph

    # Build Delaunay triangulation edges (always uses Euclidean for topology)
    edges = _build_delaunay_edges(coords, node_indices)
    graph.add_edges_from(edges)

    # Calculate distance matrix and get network information if needed
    distance_matrix, network_graph, nearest_network_nodes = _calculate_distance_matrix(
        coords, node_indices, distance_metric, network_gdf, gdf
    )

    # Add distance weights to edges
    _add_distance_weights(graph, list(edges), node_indices, distance_matrix)

    # Add edge geometries based on distance metric
    _add_edge_geometries(graph, coords, node_indices, distance_metric,
                        network_graph, nearest_network_nodes)

    # Return as GeoDataFrame if requested
    if as_gdf:
        return nx_to_gdf(graph, edges=True)

    return graph


def gilbert_graph(gdf: gpd.GeoDataFrame,
                  radius: float,
                  distance_metric: str = "euclidean",
                  network_gdf: gpd.GeoDataFrame | None = None,
                  as_gdf: bool = False) -> nx.Graph | gpd.GeoDataFrame:
    """
    Generate Gilbert disc model graph from GeoDataFrame point geometries.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame with point geometries.
    radius : float
        Connection radius.
    distance_metric : str, default "euclidean"
        Distance metric to use. Options: "euclidean", "manhattan", or "network".
    network_gdf : geopandas.GeoDataFrame, optional
        Network GeoDataFrame for network distance calculations.
        Required when distance_metric="network".
    as_gdf : bool, default False
        If True, return edges as GeoDataFrame instead of NetworkX graph.

    Returns
    -------
    networkx.Graph or geopandas.GeoDataFrame
        Graph with nodes and edges connecting points within radius.
        If as_gdf=True, returns GeoDataFrame of edges.
    """
    graph, coords, node_indices = _init_graph_and_nodes(gdf)
    if coords is None or len(coords) < 2:
        return graph

    # Calculate distance matrix and get network information
    distance_matrix, network_graph, nearest_network_nodes = _calculate_distance_matrix(
        coords, node_indices, distance_metric, network_gdf, gdf
    )

    # Create edges within radius
    if distance_metric == "network":
        # Network distance edges
        within_radius_mask = (distance_matrix <= radius) & (distance_matrix < np.inf)
        upper_tri_mask = np.triu(within_radius_mask, k=1)
        edge_pairs = np.where(upper_tri_mask)
        edges = [
            (node_indices[i], node_indices[j])
            for i, j in zip(edge_pairs[0], edge_pairs[1], strict=False)
        ]
    else:
        # Euclidean or Manhattan distance edges
        edge_mask = np.triu(distance_matrix <= radius, k=1)
        edge_indices = np.nonzero(edge_mask)
        edges = [
            (node_indices[i], node_indices[j])
            for i, j in zip(edge_indices[0], edge_indices[1], strict=False)
        ]

    # Add edges with weights (vectorized)
    if edges:
        # Create node index mapping for faster lookups
        node_idx_map = {node: idx for idx, node in enumerate(node_indices)}

        # Vectorized edge weight calculation
        edge_weights = {}
        for u, v in edges:
            i = node_idx_map[u]
            j = node_idx_map[v]
            edge_weights[(u, v)] = distance_matrix[i, j]

        # Add edges with weights all at once
        graph.add_edges_from([(u, v, {"weight": w}) for (u, v), w in edge_weights.items()])

    graph.graph["radius"] = radius

    # Add edge geometries
    _add_edge_geometries(graph, coords, node_indices, distance_metric,
                        network_graph, nearest_network_nodes)

    if as_gdf:
        return nx_to_gdf(graph, edges=True)

    return graph


def waxman_graph(
    gdf: gpd.GeoDataFrame,
    beta: float,
    r0: float,
    seed: int | None = None,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
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
    distance_metric : str, default "euclidean"
        Distance metric to use. Options: "euclidean", "manhattan", or "network".
        Note: Waxman model uses distance for probability calculation.
    network_gdf : geopandas.GeoDataFrame, optional
        Network GeoDataFrame for network distance calculations.
        Required when distance_metric="network".
    as_gdf : bool, default False
        If True, return edges as GeoDataFrame instead of NetworkX graph.

    Returns
    -------
    networkx.Graph or geopandas.GeoDataFrame
        Stochastic Waxman graph with 'beta' and 'r0' in graph attributes.
        If as_gdf=True, returns GeoDataFrame of edges.
    """
    graph, coords, node_indices = _init_graph_and_nodes(gdf)
    if coords is None or len(coords) < 2:
        return graph

    # Initialize variables for geometry creation
    network_graph = None
    nearest_network_nodes = None

    if distance_metric == "network":
        if network_gdf is None:
            msg = "network_gdf is required when distance_metric='network'"
            raise ValueError(msg)

        # Setup network computation using the helper function
        network_graph, distance_matrix, nearest_network_nodes = _setup_network_computation(
            gdf, network_gdf, coords, node_indices
        )

        # Calculate probabilities based on network distances
        probs = beta * np.exp(-distance_matrix / r0)
        # Set infinite distances to zero probability
        probs[distance_matrix == np.inf] = 0
    elif distance_metric == "manhattan":
        # Manhattan distance
        dists = squareform(pdist(coords, metric="cityblock"))
        probs = beta * np.exp(-dists / r0)
        network_graph = None
        nearest_network_nodes = None
    else:
        # Euclidean distance (original implementation)
        dists = squareform(pdist(coords))
        probs = beta * np.exp(-dists / r0)
        network_graph = None
        nearest_network_nodes = None

    # Generate random connections based on probabilities
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

    # Add edge weights based on distance metric (vectorized)
    if distance_metric == "network":
        # Vectorized network distance weights
        edge_weights = {
            edge: distance_matrix[node_indices.index(edge[0]), node_indices.index(edge[1])]
            for edge in edges
        }
    elif distance_metric == "manhattan":
        # Vectorized Manhattan distance weights
        edge_weights = {
            edge: dists[node_indices.index(edge[0]), node_indices.index(edge[1])]
            for edge in edges
        }
    else:
        # Vectorized Euclidean distance weights
        edge_weights = {
            edge: dists[node_indices.index(edge[0]), node_indices.index(edge[1])]
            for edge in edges
        }

    # Set all weights at once
    nx.set_edge_attributes(graph, edge_weights, "weight")

    # Add edge geometries based on distance metric
    _add_edge_geometries(graph, coords, node_indices, distance_metric, network_graph, nearest_network_nodes)

    # Return as GeoDataFrame if requested
    if as_gdf:
        return nx_to_gdf(graph, edges=True)

    return graph


def _create_manhattan_linestring(coord1: tuple, coord2: tuple) -> LineString:
    """
    Create Manhattan distance LineString geometry between two coordinates.

    Parameters
    ----------
    coord1 : tuple
        First coordinate (x, y).
    coord2 : tuple
        Second coordinate (x, y).

    Returns
    -------
    LineString
        L-shaped path representing Manhattan distance.
    """
    x1, y1 = coord1
    x2, y2 = coord2

    # Create L-shaped path: horizontal first, then vertical
    return LineString([(x1, y1), (x2, y1), (x2, y2)])


def _create_network_linestring(source_node: str | int,
                              target_node: str | int,
                              network_graph: nx.Graph,
                              node_indices: list,
                              nearest_network_nodes: list) -> LineString | None:
    """
    Create network path LineString geometry between two nodes.

    Parameters
    ----------
    source_node : str | int
        Source node ID in the input data.
    target_node : str | int
        Target node ID in the input data.
    network_graph : nx.Graph
        NetworkX graph representation of the network.
    node_indices : list
        List of input node indices.
    nearest_network_nodes : list
        List mapping input nodes to nearest network nodes.

    Returns
    -------
    LineString | None
        Network path geometry, or None if no path exists.
    """
    try:
        # Get network node IDs for source and target
        source_idx = node_indices.index(source_node)
        target_idx = node_indices.index(target_node)
        source_net_node = nearest_network_nodes[source_idx]
        target_net_node = nearest_network_nodes[target_idx]

        # Get shortest path
        path = nx.shortest_path(network_graph, source_net_node, target_net_node)

        # Extract coordinates for path nodes
        pos_dict = nx.get_node_attributes(network_graph, "pos")
        if not pos_dict:
            # Fallback to x,y attributes
            node_attrs = dict(network_graph.nodes(data=True))
            pos_dict = {
                node_id: (attrs.get("x", 0), attrs.get("y", 0))
                for node_id, attrs in node_attrs.items()
                if "x" in attrs and "y" in attrs
            }

        if pos_dict:
            path_coords = [pos_dict[node] for node in path if node in pos_dict]
            if len(path_coords) >= 2:
                return LineString(path_coords)
    except (nx.NetworkXNoPath, ValueError, KeyError):
        pass

    return None


def _add_edge_geometries(graph: nx.Graph,
                        coords: np.ndarray,
                        node_indices: list,
                        distance_metric: str,
                        network_graph: nx.Graph | None = None,
                        nearest_network_nodes: list | None = None) -> None:
    """
    Add appropriate edge geometries based on distance metric.

    Parameters
    ----------
    graph : nx.Graph
        Graph to add geometries to.
    coords : np.ndarray
        Array of coordinates.
    node_indices : list
        List of node indices.
    distance_metric : str
        Distance metric used.
    network_graph : nx.Graph, optional
        Network graph for network distance geometries.
    nearest_network_nodes : list, optional
        Mapping to nearest network nodes.
    """
    coord_dict = {node_indices[i]: coords[i] for i in range(len(node_indices))}

    # Vectorized geometry creation
    edge_geometries = {}

    for u, v in graph.edges():
        if distance_metric == "network" and network_graph is not None and nearest_network_nodes:
            # Create network path geometry
            geom = _create_network_linestring(u, v, network_graph, node_indices, nearest_network_nodes)
            if geom is None:
                # Fallback to straight line if no network path
                geom = LineString([coord_dict[u], coord_dict[v]])
        elif distance_metric == "manhattan":
            # Create Manhattan distance geometry
            geom = _create_manhattan_linestring(coord_dict[u], coord_dict[v])
        else:
            # Default: straight line for Euclidean distance
            geom = LineString([coord_dict[u], coord_dict[v]])

        edge_geometries[(u, v)] = geom

    # Set all geometries at once
    nx.set_edge_attributes(graph, edge_geometries, "geometry")


def _calculate_distance_matrix(coords: np.ndarray,
                             node_indices: list,
                             distance_metric: str,
                             network_gdf: gpd.GeoDataFrame | None = None,
                             gdf: gpd.GeoDataFrame | None = None,
                             ) -> tuple[np.ndarray, nx.Graph | None, list | None]:
    """
    Calculate distance matrix based on the specified metric.

    Parameters
    ----------
    coords : np.ndarray
        Array of coordinates.
    node_indices : list
        List of node indices.
    distance_metric : str
        Distance metric to use.
    network_gdf : gpd.GeoDataFrame, optional
        Network GeoDataFrame for network distances.
    gdf : gpd.GeoDataFrame, optional
        Original GeoDataFrame for CRS validation.

    Returns
    -------
    tuple[np.ndarray, nx.Graph | None, list | None]
        Distance matrix, network graph (if used), and nearest network nodes (if used).
    """
    if distance_metric == "network":
        if network_gdf is None or gdf is None:
            msg = "network_gdf and gdf are required when distance_metric='network'"
            raise ValueError(msg)

        network_graph, distance_matrix, nearest_network_nodes = _setup_network_computation(
            gdf, network_gdf, coords, node_indices
        )
        return distance_matrix, network_graph, nearest_network_nodes

    if distance_metric == "manhattan":
        distance_matrix = squareform(pdist(coords, metric="cityblock"))
        return distance_matrix, None, None

    # Default to euclidean
    distance_matrix = squareform(pdist(coords))
    return distance_matrix, None, None


def _add_distance_weights(graph: nx.Graph,
                         edges: list,
                         node_indices: list,
                         distance_matrix: np.ndarray) -> None:
    """
    Add distance weights to graph edges using vectorized operations.

    Parameters
    ----------
    graph : nx.Graph
        Graph to add weights to.
    edges : list
        List of edges.
    node_indices : list
        List of node indices.
    distance_matrix : np.ndarray
        Precomputed distance matrix.
    """
    # Create node index mapping for O(1) lookup
    node_idx_map = {node: idx for idx, node in enumerate(node_indices)}

    # Vectorized weight calculation
    edge_weights = {
        (u, v): distance_matrix[node_idx_map[u], node_idx_map[v]]
        for u, v in edges
    }

    # Set all weights at once
    nx.set_edge_attributes(graph, edge_weights, "weight")
