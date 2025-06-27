"""Module for generating proximity-based graphs from geospatial data.

This module provides a suite of functions for constructing NetworkX graphs from
geospatial data (represented as GeoDataFrames) based on various spatial
proximity and connectivity models. These functions are essential for
translating spatial relationships into a graph structure suitable for network
analysis, spatial modeling, and as input for graph-based machine learning.

The module supports several common graph construction methods, including
k-nearest neighbors, Delaunay triangulation, fixed-radius connections (Gilbert
graph), and probabilistic connections (Waxman graph). It also offers advanced
functionality to compute distances along a provided network (e.g., streets)
instead of simple Euclidean or Manhattan distances.

Key Features
------------
- Generation of graphs from GeoDataFrame geometries (Points, Polygons, etc.).
- Support for multiple distance metrics: "euclidean", "manhattan", and "network".
- Construction of k-nearest neighbor (KNN) graphs.
- Construction of Delaunay triangulation graphs.
- Construction of random geometric graphs (Gilbert and Waxman models).
- Optional conversion of output graphs directly to node and edge GeoDataFrames.
- Preservation of Coordinate Reference Systems (CRS).
- Rich attribute generation, including edge weights and geometries.

Main Functions
--------------
knn_graph : Generate a k-nearest neighbor graph.
delaunay_graph : Generate a graph from Delaunay triangulation.
gilbert_graph : Generate a graph by connecting nodes within a fixed radius.
waxman_graph : Generate a probabilistic random geometric graph.

See Also
--------
city2graph.graph : Functions for converting graphs to PyTorch Geometric objects.
city2graph.utils : Utility functions for data validation and conversion.

Notes
-----
- All functions use the centroids of input geometries as node locations.
- The index of the input GeoDataFrame is used to identify nodes in the graph.
- When using `distance_metric="network"`, a `network_gdf` representing the
  underlying network (e.g., streets) must be provided.

Examples
--------
Basic usage of knn_graph:

>>> import geopandas as gpd
>>> from shapely.geometry import Point
>>> from city2graph.proximity import knn_graph
>>>
>>> # Create a sample GeoDataFrame of points
>>> d = {'geometry': [Point(0, 0), Point(1, 1), Point(0, 1), Point(1, 0)]}
>>> gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
>>>
>>> # Generate a 2-nearest neighbor graph
>>> G = knn_graph(gdf, k=2)
>>> print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
Graph has 4 nodes and 4 edges.
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

from .utils import GraphMetadata
from .utils import gdf_to_nx
from .utils import nx_to_gdf
from .utils import validate_gdf

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
                    network_graph, source_node, weight="length",
                )
            else:
                distances = nx.single_source_shortest_path_length(
                    network_graph, source_node,
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
        coords, node_indices, network_graph,
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

    # Prepare node attributes, preserving all original columns
    node_attrs_list = gdf.to_dict("records")
    node_attrs = {
        idx: {**attrs, "pos": (centroid.x, centroid.y)}
        for idx, attrs, centroid in zip(gdf.index, node_attrs_list, centroids, strict=False)
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
    validate_gdf(nodes_gdf=data)

    # Initialize graph with metadata
    G = nx.Graph()
    metadata = GraphMetadata(crs=data.crs, is_hetero=False)

    # Handle empty data
    if data.empty:
        G.graph.update(metadata.to_dict())
        return G, None, None

    # Extract coordinates and attributes, add nodes to graph
    coords, node_attrs = _extract_coords_and_attrs_from_gdf(data)
    node_indices = list(data.index)
    G.add_nodes_from(node_indices)
    nx.set_node_attributes(G, node_attrs)

    if data.index.nlevels > 1:
        metadata.node_index_names = list(data.index.names)
    else:
        metadata.node_index_names = data.index.name
    G.graph.update(metadata.to_dict())

    return G, coords, node_indices


def knn_graph(gdf: gpd.GeoDataFrame,
              k: int = 5,
              distance_metric: str = "euclidean",
              network_gdf: gpd.GeoDataFrame | None = None,
              as_gdf: bool = False) -> nx.Graph | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Generate k-nearest neighbor graph from points or polygon centroids.

    This function constructs a graph where each node (representing a geometry
    centroid from the input GeoDataFrame) is connected to its `k` nearest
    neighbors. The definition of "nearest" can be based on Euclidean,
    Manhattan, or on-network distance, providing flexibility for different
    spatial analysis contexts.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input data as a GeoDataFrame. The centroids of geometries are used as
        node locations. The index of the GeoDataFrame is used for node IDs.
    k : int, default 5
        The number of nearest neighbors to connect to each node.
    distance_metric : str, default "euclidean"
        The distance metric to use for finding neighbors.
        Options: "euclidean", "manhattan", or "network".
    network_gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame representing a street network (e.g., lines).
        This is required when `distance_metric` is "network".
    as_gdf : bool, default False
        If True, the function returns a tuple of GeoDataFrames (nodes, edges)
        instead of a NetworkX graph.

    Returns
    -------
    networkx.Graph or tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        A NetworkX graph representing the k-nearest neighbor connections.
        If `as_gdf` is True, returns a tuple of (nodes_gdf, edges_gdf).
        The graph nodes include attributes from the original `gdf` and a 'pos'
        attribute with coordinate tuples. Edges have a 'weight' attribute
        representing the distance and a 'geometry' attribute.

    Raises
    ------
    ValueError
        If `distance_metric` is "network" but `network_gdf` is not provided,
        or if the CRS of `gdf` and `network_gdf` do not match.
    TypeError
        If the input `gdf` is not a GeoDataFrame.

    See Also
    --------
    delaunay_graph : Create a graph based on Delaunay triangulation.
    gilbert_graph : Create a graph based on a fixed radius.
    city2graph.utils.nx_to_gdf : Convert a NetworkX graph to GeoDataFrames.

    Notes
    -----
    - The function uses the centroids of the input geometries as node positions.
    - When using "network" distance, the function finds the nearest network
      node for each point and computes shortest path distances on the network.
    - Edge geometries are straight lines for "euclidean", L-shaped lines for
      "manhattan", and network paths for "network" distance.

    Examples
    --------
    Create a simple KNN graph with Euclidean distance:

    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> from city2graph.proximity import knn_graph
    >>>
    >>> # Create a sample GeoDataFrame of points
    >>> d = {'col1': ['name1', 'name2', 'name3', 'name4'],
    ...      'geometry': [Point(0, 0), Point(1, 1), Point(0, 1), Point(1, 0)]}
    >>> gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    >>>
    >>> # Generate a 2-nearest neighbor graph
    >>> G = knn_graph(gdf, k=2)
    >>> print(G.number_of_nodes(), G.number_of_edges())
    4 4

    Generate a KNN graph and return as GeoDataFrames:

    >>> nodes_gdf, edges_gdf = knn_graph(gdf, k=1, as_gdf=True)
    >>> print(isinstance(nodes_gdf, gpd.GeoDataFrame))
    True
    >>> print(isinstance(edges_gdf, gpd.GeoDataFrame))
    True
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
            gdf, network_gdf, coords, node_indices,
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
        return nx_to_gdf(graph, nodes=True, edges=True)

    return graph


def delaunay_graph(gdf: gpd.GeoDataFrame,
                   distance_metric: str = "euclidean",
                   network_gdf: gpd.GeoDataFrame | None = None,
                   as_gdf: bool = False) -> nx.Graph | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Generate a Delaunay graph from points or polygon centroids.

    This function creates a graph based on the Delaunay triangulation of the
    input points. The triangulation connects points such that no point is
    inside the circumcircle of any triangle, resulting in a planar graph that
    is useful for representing spatial proximity and neighborhood relationships.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input data as a GeoDataFrame. The centroids of geometries are used as
        node locations. The index of the GeoDataFrame is used for node IDs.
    distance_metric : str, default "euclidean"
        The distance metric to use for calculating edge weights. The graph's
        topology is always based on Euclidean Delaunay triangulation.
        Options: "euclidean", "manhattan", or "network".
    network_gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame representing a street network. Required if
        `distance_metric` is "network" to calculate edge weights.
    as_gdf : bool, default False
        If True, the function returns a tuple of GeoDataFrames (nodes, edges)
        instead of a NetworkX graph.

    Returns
    -------
    networkx.Graph or tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        A NetworkX graph representing the Delaunay triangulation.
        If `as_gdf` is True, returns a tuple of (nodes_gdf, edges_gdf).
        The graph nodes include attributes from the original `gdf` and a 'pos'
        attribute. Edges have a 'weight' attribute with the calculated
        distance and a 'geometry' attribute.

    Raises
    ------
    ValueError
        If `distance_metric` is "network" but `network_gdf` is not provided,
        or if the CRS of `gdf` and `network_gdf` do not match.
    TypeError
        If the input `gdf` is not a GeoDataFrame.

    See Also
    --------
    knn_graph : Create a graph based on k-nearest neighbors.
    scipy.spatial.Delaunay : The underlying triangulation implementation.

    Notes
    -----
    - The topological structure of the graph (which nodes are connected) is
      always determined by Euclidean-based Delaunay triangulation. The
      `distance_metric` parameter only affects the 'weight' attribute of the
      edges.
    - The function requires at least 3 non-collinear points to generate a
      triangulation. For fewer points, an empty graph is returned.

    Examples
    --------
    Create a Delaunay graph from a set of points:

    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> from city2graph.proximity import delaunay_graph
    >>>
    >>> d = {'col1': ['name1', 'name2', 'name3', 'name4'],
    ...      'geometry': [Point(0, 0), Point(1, 1), Point(0, 1), Point(1, 0)]}
    >>> gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    >>>
    >>> G = delaunay_graph(gdf)
    >>> print(G.number_of_nodes(), G.number_of_edges())
    4 5
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
        coords, node_indices, distance_metric, network_gdf, gdf,
    )

    # Add distance weights to edges
    _add_distance_weights(graph, list(edges), node_indices, distance_matrix)

    # Add edge geometries based on distance metric
    _add_edge_geometries(graph, coords, node_indices, distance_metric,
                        network_graph, nearest_network_nodes)

    # Return as GeoDataFrame if requested
    if as_gdf:
        return nx_to_gdf(graph, nodes=True, edges=True)

    return graph


def gilbert_graph(gdf: gpd.GeoDataFrame,
                  radius: float,
                  distance_metric: str = "euclidean",
                  network_gdf: gpd.GeoDataFrame | None = None,
                  as_gdf: bool = False) -> nx.Graph | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Generate a Gilbert disc model graph from GeoDataFrame point geometries.

    This function creates a random geometric graph where two nodes are connected
    if the distance between them is less than or equal to a specified `radius`.
    This model is useful for representing connectivity based on a fixed-range
    threshold, such as wireless signal range or service area.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame with point or polygon geometries. Centroids are used
        as node locations. The index is used for node IDs.
    radius : float
        The connection radius. Nodes within this distance of each other will be
        connected by an edge.
    distance_metric : str, default "euclidean"
        The distance metric to use for checking the radius.
        Options: "euclidean", "manhattan", or "network".
    network_gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame representing a street network. Required if
        `distance_metric` is "network".
    as_gdf : bool, default False
        If True, the function returns a tuple of GeoDataFrames (nodes, edges)
        instead of a NetworkX graph.

    Returns
    -------
    networkx.Graph or tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        A NetworkX graph with nodes and edges connecting points within the radius.
        If `as_gdf` is True, returns a tuple of (nodes_gdf, edges_gdf).
        The graph has a 'radius' attribute, and edges have 'weight' and
        'geometry' attributes.

    Raises
    ------
    ValueError
        If `distance_metric` is "network" but `network_gdf` is not provided,
        or if the CRS of `gdf` and `network_gdf` do not match.
    TypeError
        If the input `gdf` is not a GeoDataFrame.

    See Also
    --------
    waxman_graph : Create a probabilistic random geometric graph.
    knn_graph : Create a graph based on a fixed number of neighbors.

    Notes
    -----
    - The `radius` should be in the same units as the coordinate system of the
      input `gdf`.
    - This is a deterministic model; for a probabilistic version, see
      `waxman_graph`.

    Examples
    --------
    Create a Gilbert graph with a specific radius:

    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> from city2graph.proximity import gilbert_graph
    >>>
    >>> d = {'col1': ['name1', 'name2', 'name3', 'name4'],
    ...      'geometry': [Point(0, 0), Point(1.1, 1.1), Point(0, 1), Point(2, 2)]}
    >>> gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    >>>
    >>> # Connect nodes within a distance of 1.5
    >>> G = gilbert_graph(gdf, radius=1.5)
    >>> print(sorted(list(G.edges)))
    [(0, 1), (0, 2), (1, 2)]
    """
    graph, coords, node_indices = _init_graph_and_nodes(gdf)
    if coords is None or len(coords) < 2:
        return graph

    # Calculate distance matrix and get network information
    distance_matrix, network_graph, nearest_network_nodes = _calculate_distance_matrix(
        coords, node_indices, distance_metric, network_gdf, gdf,
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
        return nx_to_gdf(graph, nodes=True, edges=True)

    return graph


def waxman_graph(
    gdf: gpd.GeoDataFrame,
    beta: float,
    r0: float,
    seed: int | None = None,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    as_gdf: bool = False) -> nx.Graph | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    r"""Generate a Waxman random geometric graph.

    This function creates a probabilistic graph where the likelihood of an edge
    between two nodes `i` and `j` is given by the Waxman model formula:
    $H_{ij} = \beta \exp(-d_{ij} / r_0)$, where $d_{ij}$ is the distance
    between the nodes. This model is useful for creating more realistic
    networks where connectivity is a decreasing function of distance.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame with point or polygon geometries. Centroids are used
        as node locations. The index is used for node IDs.
    beta : float
        A model parameter that controls the overall density of edges. Higher
        values increase the probability of connections. Typically $0 < \beta \le 1$.
    r0 : float
        A model parameter that controls the sensitivity to distance. It acts as
        a distance scale factor. Larger values make connections over longer
        distances more likely.
    seed : int, optional
        A random seed for the random number generator to ensure reproducibility.
    distance_metric : str, default "euclidean"
        The distance metric to use for $d_{ij}$ in the probability calculation.
        Options: "euclidean", "manhattan", or "network".
    network_gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame representing a street network. Required if
        `distance_metric` is "network".
    as_gdf : bool, default False
        If True, the function returns a tuple of GeoDataFrames (nodes, edges)
        instead of a NetworkX graph.

    Returns
    -------
    networkx.Graph or tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        A NetworkX graph representing the stochastic Waxman model.
        If `as_gdf` is True, returns a tuple of (nodes_gdf, edges_gdf).
        The graph includes 'beta' and 'r0' as graph-level attributes.

    Raises
    ------
    ValueError
        If `distance_metric` is "network" but `network_gdf` is not provided,
        or if the CRS of `gdf` and `network_gdf` do not match.
    TypeError
        If the input `gdf` is not a GeoDataFrame.

    See Also
    --------
    gilbert_graph : Create a deterministic geometric graph based on a fixed radius.

    Notes
    -----
    - The `r0` parameter should be in the same units as the coordinate system
      of the input `gdf`.
    - The connection probability decreases exponentially with distance.

    Examples
    --------
    Create a Waxman graph with a given seed for reproducibility:

    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> from city2graph.proximity import waxman_graph
    >>>
    >>> d = {'col1': ['name1', 'name2', 'name3', 'name4'],
    ...      'geometry': [Point(0, 0), Point(5, 5), Point(0, 1), Point(1, 0)]}
    >>> gdf = gpd.GeoDataFrame(d, crs="EPSG:4326")
    >>>
    >>> # Generate a graph with high probability for short distances
    >>> G = waxman_graph(gdf, beta=0.8, r0=1.0, seed=42)
    >>> print(sorted(list(G.edges)))
    [(0, 2), (0, 3), (2, 3)]
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
            gdf, network_gdf, coords, node_indices,
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
        return nx_to_gdf(graph, nodes=True, edges=True)

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
            gdf, network_gdf, coords, node_indices,
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
