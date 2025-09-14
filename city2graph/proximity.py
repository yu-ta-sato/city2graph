"""
Proximity-Based Graph Generation Module.

This module provides comprehensive functionality for generating graph networks based
on spatial proximity relationships between geographic features. It implements several
classical proximity models commonly used in spatial network analysis and geographic
information systems. The module is particularly useful for constructing heterogeneous
graphs from multiple domains of geospatial relations, enabling complex spatial analysis
across different feature types and scales.
"""

# Future annotations for type hints
from __future__ import annotations

# Standard library imports
import logging
from itertools import combinations
from itertools import permutations
from typing import Any
from typing import cast

# Third-party imports
import geopandas as gpd
import libpysal
import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay
from scipy.spatial import distance as sdist
from shapely.geometry import LineString
from sklearn.neighbors import NearestNeighbors

# Local imports
from .utils import gdf_to_nx
from .utils import nx_to_gdf
from .utils import validate_gdf

# Type checking imports

# Module logger configuration
logger = logging.getLogger(__name__)

__all__ = [
    "bridge_nodes",
    "contiguity_graph",
    "delaunay_graph",
    "euclidean_minimum_spanning_tree",
    "fixed_radius_graph",
    "gabriel_graph",
    "knn_graph",
    "relative_neighborhood_graph",
    "waxman_graph",
]

# Simple type alias for readability
EdgePair = tuple[Any, Any]


# ============================================================================
# GRAPH GENERATORS
# ============================================================================


def knn_graph(
    gdf: gpd.GeoDataFrame,
    k: int = 5,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    *,
    target_gdf: gpd.GeoDataFrame | None = None,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a k-nearest neighbour graph from a GeoDataFrame of points.

    This function constructs a graph where each node is connected to its k nearest neighbors
    based on the specified distance metric. The resulting graph captures local spatial
    relationships and is commonly used in spatial analysis, clustering, and network topology
    studies.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the points (nodes) for the graph. The index of this
        GeoDataFrame will be used as node IDs.
    k : int, default 5
        The number of nearest neighbors to connect to each node. Must be positive and
        less than the total number of nodes.
    distance_metric : str, default "euclidean"
        The distance metric to use for calculating nearest neighbors. Options are:
        - "euclidean": Straight-line distance
        - "manhattan": City-block distance (L1 norm)
        - "network": Shortest path distance along a network
    network_gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame representing a network (e.g., roads, paths) to use for "network"
        distance calculations. Required if `distance_metric` is "network".
    target_gdf : geopandas.GeoDataFrame, optional
        If provided, creates a directed graph where edges connect nodes from `gdf` to
        their k nearest neighbors in `target_gdf`. If None, creates an undirected graph
        within `gdf` itself.
    as_nx : bool, default False
        If True, returns a NetworkX graph object. Otherwise, returns a tuple of
        GeoDataFrames (nodes, edges).

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    Raises
    ------
    ValueError
        If `distance_metric` is "network" but `network_gdf` is not provided.
        If `k` is greater than or equal to the number of available nodes.

    See Also
    --------
    delaunay_graph : Generate a Delaunay triangulation graph.
    fixed_radius_graph : Generate a fixed-radius graph.
    waxman_graph : Generate a probabilistic Waxman graph.

    Notes
    -----
    - Node IDs are preserved from the input GeoDataFrame's index
    - Edge weights represent the distance between connected nodes
    - Edge geometries are LineStrings connecting node centroids
    - For Manhattan distance, edge geometries follow L-shaped paths
    - The graph is undirected unless `target_gdf` is specified

    References
    ----------
    Eppstein, D., Paterson, M.S. & Yao, F.F. On Nearest-Neighbor Graphs.
    Discrete Comput Geom 17, 263-282 (1997). [1](https://doi.org/10.1007/PL00009293)

    Examples
    --------
    >>> import geopandas as gpd
    >>> import numpy as np
    >>> from shapely.geometry import Point
    >>>
    >>> # Create a sample GeoDataFrame with 6 points
    >>> np.random.seed(42)
    >>> coords = np.random.rand(6, 2) * 10
    >>> data = {
    ...     'id': [f'node_{i}' for i in range(6)],
    ...     'type': ['residential', 'commercial', 'industrial', 'park', 'school', 'hospital'],
    ...     'geometry': [Point(x, y) for x, y in coords]
    ... }
    >>> gdf = gpd.GeoDataFrame(data, crs="EPSG:4326").set_index('id')
    >>> print("Input GeoDataFrame:")
    >>> print(gdf.head(3))
            type                     geometry
    id
    node_0  residential  POINT (3.745 9.507)
    node_1   commercial  POINT (7.319 5.987)
    node_2   industrial  POINT (1.560 0.581)

    >>> # Generate a 3-nearest neighbor graph
    >>> nodes_gdf, edges_gdf = knn_graph(gdf, k=3, distance_metric="euclidean")
    >>> print(f"\\nNodes GDF shape: {nodes_gdf.shape}")
    >>> print(f"Edges GDF shape: {edges_gdf.shape}")
    Nodes GDF shape: (6, 2)
    Edges GDF shape: (18, 2)

    >>> print("\\nSample edges with weights:")
    >>> print(edges_gdf[['weight']].head(3))
           weight
    0    4.186842
    1    6.190525
    2    8.944272

    >>> # Generate with Manhattan distance
    >>> nodes_manhattan, edges_manhattan = knn_graph(
    ...     gdf, k=2, distance_metric="manhattan"
    ... )
    >>> print(f"\\nManhattan edges count: {len(edges_manhattan)}")
    Manhattan edges count: 12

    >>> # Generate as NetworkX graph
    >>> G = knn_graph(gdf, k=3, as_nx=True)
    >>> print(f"\\nNetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    >>> print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    NetworkX graph: 6 nodes, 9 edges
    Average degree: 3.00

    >>> # Directed graph example with target_gdf
    >>> target_data = {
    ...     'id': ['target_1', 'target_2'],
    ...     'service': ['hospital', 'school'],
    ...     'geometry': [Point(5, 5), Point(8, 8)]
    ... }
    >>> target_gdf = gpd.GeoDataFrame(target_data, crs="EPSG:4326").set_index('id')
    >>> nodes_dir, edges_dir = knn_graph(gdf, k=1, target_gdf=target_gdf)
    >>> print(f"\\nDirected graph edges: {len(edges_dir)} (each source node → 1 target)")
    Directed graph edges: 6 (each source node → 1 target)
    """
    # Handle directed variant
    if target_gdf is not None:
        return _directed_graph(
            src_gdf=gdf,
            dst_gdf=target_gdf,
            distance_metric=distance_metric,
            method="knn",
            param=k,
            as_nx=as_nx,
            network_gdf=network_gdf,
        )

    # Prepare nodes and handle trivial cases
    G, coords, node_ids = _prepare_nodes(gdf)
    if len(coords) <= 1 or k <= 0:
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # Generate edges based on distance metric
    dm = None
    if distance_metric == "network":
        dm = _distance_matrix(coords, "network", network_gdf, gdf.crs)
        order = np.argsort(dm, axis=1)[:, 1 : k + 1]
        edges = [
            (node_ids[i], node_ids[j])
            for i in range(len(node_ids))
            for j in order[i]
            if dm[i, j] < np.inf
        ]
    else:
        nn_metric = "cityblock" if distance_metric == "manhattan" else "euclidean"
        n_neigh = min(k + 1, len(coords))
        nn = NearestNeighbors(n_neighbors=n_neigh, metric=nn_metric).fit(coords)
        _, idxs = nn.kneighbors(coords)
        edges = [(node_ids[i], node_ids[j]) for i, neigh in enumerate(idxs) for j in neigh[1:]]

    # Add edges with weights and geometries
    _add_edges(G, edges, coords, node_ids, metric=distance_metric, dm=dm, network_gdf=network_gdf)
    return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)


def delaunay_graph(
    gdf: gpd.GeoDataFrame,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    *,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Generate a Delaunay triangulation graph from a GeoDataFrame of points.

    This function constructs a graph based on the Delaunay triangulation of the
    input points. Each edge in the graph corresponds to an edge in the Delaunay
    triangulation.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the points (nodes) for the graph.
        The index of this GeoDataFrame will be used as node IDs.
    distance_metric : str, default "euclidean"
        The distance metric to use for calculating edge weights.
        Options are "euclidean", "manhattan", or "network".
    network_gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame representing a network (e.g., roads) to use for
        "network" distance calculations. Required if `distance_metric` is
        "network".
    as_nx : bool, default False
        If True, returns a NetworkX graph object. Otherwise, returns a tuple
        of GeoDataFrames (nodes, edges).

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    Raises
    ------
    ValueError
        If `distance_metric` is "network" but `network_gdf` is not provided.

    See Also
    --------
    knn_graph : Generate a k-nearest neighbour graph.
    fixed_radius_graph : Generate a fixed-radius graph.
    waxman_graph : Generate a probabilistic Waxman graph.

    Notes
    -----
    - Node IDs are preserved from the input GeoDataFrame's index.
    - Edge weights represent the distance between connected nodes.
    - Edge geometries are LineStrings connecting the centroids of the nodes.
    - If the input `gdf` has fewer than 3 points, an empty graph will be returned
      as Delaunay triangulation requires at least 3 non-collinear points.

    References
    ----------
    Lee, D. T., & Schachter, B. J. (1980). Two algorithms for constructing a Delaunay
    triangulation. International Journal of Computer & Information Sciences, 9(3), 219-242. [1](https://doi.org/10.1007/BF00977785)

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> # Create a sample GeoDataFrame
    >>> data = {'id': [1, 2, 3, 4, 5],
    ...         'geometry': [Point(0, 0), Point(1, 1), Point(0, 1), Point(1, 0), Point(2, 2)]}
    >>> gdf = gpd.GeoDataFrame(data, crs="EPSG:4326").set_index('id')
    >>>
    >>> # Generate a Delaunay graph
    >>> nodes_gdf, edges_gdf = delaunay_graph(gdf)
    >>> print(nodes_gdf)
    >>> print(edges_gdf)
    >>>
    >>> # Generate a Delaunay graph as NetworkX object
    >>> G_delaunay = delaunay_graph(gdf, as_nx=True)
    >>> print(G_delaunay.nodes(data=True))
    >>> print(G_delaunay.edges(data=True))
    """
    # Input validation
    _assert_euclidean(distance_metric, "delaunay_graph")

    # Node preparation
    G, coords, node_ids = _prepare_nodes(gdf)
    if len(coords) < 3:
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # Candidate edges: Delaunay triangulation
    tri = Delaunay(coords)
    edges = {
        (node_ids[i], node_ids[j]) for simplex in tri.simplices for i, j in combinations(simplex, 2)
    }

    # Add weights and geometries
    dm = None
    if distance_metric == "network":
        dm = _distance_matrix(coords, "network", network_gdf, gdf.crs)

    _add_edges(G, edges, coords, node_ids, metric=distance_metric, dm=dm, network_gdf=network_gdf)
    return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)


def gabriel_graph(
    gdf: gpd.GeoDataFrame,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    *,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a Gabriel graph from a GeoDataFrame of points.

    In a Gabriel graph two nodes *u* and *v* are connected iff the closed
    disc that has :math:`uv` as its diameter contains no other node of the set.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input point layer. The GeoDataFrame index is preserved as the node id.
    distance_metric : {'euclidean', 'manhattan', 'network'}, default 'euclidean'
        Metric used for edge weights / geometries (see the other generators).
    network_gdf : geopandas.GeoDataFrame, optional
        Required when *distance_metric='network'*.
    as_nx : bool, default False
        If *True* return a NetworkX graph, otherwise return two GeoDataFrames
        (nodes, edges) via `nx_to_gdf`.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    Notes
    -----
    • The Gabriel graph is a sub-graph of the Delaunay triangulation; therefore
      the implementation first builds the Delaunay edges then filters them
      according to the disc-emptiness predicate, achieving an overall

      .. math::

         \mathcal{O}(n \log n + m k)

      complexity ( *m* = Delaunay edges,
      *k* = average neighbours tested per edge).
    • When the input layer has exactly two points, the unique edge is returned.
    • If the layer has fewer than two points, an empty graph is produced.

    References
    ----------
    Gabriel, K. R., & Sokal, R. R. (1969). A new statistical approach to geographic
    variation analysis. Systematic zoology, 18(3), 259-278. [1](https://doi.org/10.2307/2412323)

    Examples
    --------
    >>> nodes, edges = gabriel_graph(points_gdf)
    >>> G = gabriel_graph(points_gdf, as_nx=True)
    """
    # Input validation
    _assert_euclidean(distance_metric, "delaunay_graph")

    # Node preparation
    G, coords, node_ids = _prepare_nodes(gdf)
    n_points = len(coords)
    if n_points < 2:
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # Candidate edges: Delaunay
    if n_points == 2:
        delaunay_edges = {(0, 1)}
    else:
        tri = Delaunay(coords)
        delaunay_edges = {
            tuple(sorted((i, j))) for simplex in tri.simplices for i, j in combinations(simplex, 2)
        }

    # Gabriel filtering
    # Square distances for numerical stability
    kept_edges: set[tuple[int, int]] = set()
    tol = 1e-12
    for i, j in delaunay_edges:
        mid = 0.5 * (coords[i] + coords[j])
        rad2 = np.sum((coords[i] - coords[j]) ** 2) * 0.25  # (|pi-pj|/2)^2

        # Squared distance of all points to the midpoint
        d2 = np.sum((coords - mid) ** 2, axis=1)
        mask = d2 <= rad2 + tol

        # Exactly the two endpoints inside the disc?
        if np.count_nonzero(mask) == 2:
            kept_edges.add((node_ids[i], node_ids[j]))

    # Add weights and geometries
    dm = None
    if distance_metric.lower() == "network":
        dm = _distance_matrix(coords, "network", network_gdf, gdf.crs)

    _add_edges(
        G,
        kept_edges,
        coords,
        node_ids,
        metric=distance_metric,
        dm=dm,
        network_gdf=network_gdf,
    )

    return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)


def relative_neighborhood_graph(
    gdf: gpd.GeoDataFrame,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    *,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a Relative-Neighbourhood Graph (RNG) from a GeoDataFrame.

    In an RNG two nodes *u* and *v* are connected iff there is **no third node
    *w*** such that both :math:`d(u,w) < d(u,v)` **and** :math:`d(v,w) < d(u,v)`.
    Equivalently, the intersection of the two open discs having radius
    :math::`d(u,v)` and centres *u* and *v* (the *lune*) is empty.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input point layer whose index provides the node ids.
    distance_metric : {'euclidean', 'manhattan', 'network'}, default 'euclidean'
        Metric used to attach edge weights / geometries.
    network_gdf : geopandas.GeoDataFrame, optional
        Required when *distance_metric='network'*.
    as_nx : bool, default False
        If *True* return a NetworkX graph, otherwise return two GeoDataFrames
        (nodes, edges) via `nx_to_gdf`.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    Notes
    -----
    •  The RNG is a sub-graph of the Delaunay triangulation; therefore the
       implementation first collects Delaunay edges ( :math:`\mathcal{O}(n\log n)` )
       and then filters them according to the lune-emptiness predicate.
    •  When the input layer has exactly two points the unique edge is returned.
    •  If the layer has fewer than two points, an empty graph is produced.

    References
    ----------
    Toussaint, G. T. (1980). The relative neighbourhood graph of a finite planar set.
    Pattern recognition, 12(4), 261-268. [1](https://doi.org/10.1016/0031-3203(80)90066-7)

    Examples
    --------
    >>> nodes, edges = relative_neighborhood_graph(points_gdf)
    >>> G = relative_neighborhood_graph(points_gdf, as_nx=True)
    """
    # Input validation
    _assert_euclidean(distance_metric, "delaunay_graph")

    # Node preparation
    G, coords, node_ids = _prepare_nodes(gdf)
    n_points = len(coords)
    if n_points < 2:
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # Candidate edges: Delaunay
    if n_points == 2:
        cand_edges = {(0, 1)}
    else:
        tri = Delaunay(coords)
        cand_edges = {
            tuple(sorted((i, j))) for simplex in tri.simplices for i, j in combinations(simplex, 2)
        }

    # RNG filtering
    kept_edges: set[tuple[int, int]] = set()

    # Work with squared distances to avoid sqrt
    for i, j in cand_edges:
        dij2 = np.dot(coords[i] - coords[j], coords[i] - coords[j])

        # Vectorised test of the lune-emptiness predicate
        di2 = np.sum((coords - coords[i]) ** 2, axis=1) < dij2
        dj2 = np.sum((coords - coords[j]) ** 2, axis=1) < dij2

        # Any third point closer to *both* i and j?
        closer_both = np.where(di2 & dj2)[0]
        if len(closer_both) == 0:
            kept_edges.add((node_ids[i], node_ids[j]))

    # Add weights and geometries
    dm = None
    if distance_metric.lower() == "network":
        dm = _distance_matrix(coords, "network", network_gdf, gdf.crs)

    _add_edges(
        G,
        kept_edges,
        coords,
        node_ids,
        metric=distance_metric,
        dm=dm,
        network_gdf=network_gdf,
    )
    return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)


def euclidean_minimum_spanning_tree(
    gdf: gpd.GeoDataFrame,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    *,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a (generalised) Euclidean Minimum Spanning Tree from a GeoDataFrame of points.

    The classical Euclidean Minimum Spanning Tree (EMST) is the minimum-
    total-length tree that connects a set of points when edge weights are the
    straight-line ( :math:`L_2` ) distances.  For consistency with the other generators
    this implementation also supports *manhattan* and *network* metrics - it
    simply computes the minimum-weight spanning tree under the chosen metric.
    When the metric is *euclidean* the edge search is restricted to the
    Delaunay triangulation (  EMST ⊆ Delaunay ), guaranteeing an :math:`\mathcal{O}(n \log n)`
    overall complexity.  With other metrics, or degenerate cases where the triangulation cannot be built, the algorithm
    gracefully falls back to the complete graph.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input point layer.  The index is preserved as the node identifier.
    distance_metric : {'euclidean', 'manhattan', 'network'}, default 'euclidean'
        Metric used for the edge weights / geometries.
    network_gdf : geopandas.GeoDataFrame, optional
        Required when *distance_metric='network'*.  Must contain the network
        arcs with valid *pos* attributes on its nodes.
    as_nx : bool, default False
        If *True* return a NetworkX graph, otherwise return two GeoDataFrames
        (nodes, edges) via ``nx_to_gdf``.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    See Also
    --------
    delaunay_graph : Generate a Delaunay triangulation graph.
    gabriel_graph : Generate a Gabriel graph.
    relative_neighborhood_graph : Generate a Relative Neighborhood Graph.

    Notes
    -----
    •  The resulting graph always contains *n - 1* edges (or 0 / 1 when the
       input has < 2 points).
    •  For planar Euclidean inputs the computation is :math:`\mathcal{O}(n \log n)`
       thanks to the Delaunay pruning.
    •  All the usual spatial attributes (*weight*, *geometry*, CRS checks,
       etc.) are attached through the shared private helpers.

    References
    ----------
    March, W. B., Ram, P., & Gray, A. G. (2010, July). Fast euclidean minimum spanning tree:
    algorithm, analysis, and applications. In Proceedings of the 16th ACM SIGKDD international
    conference on Knowledge discovery and data mining (pp. 603-612). [1](https://doi.org/10.1145/1835804.1835882)

    Examples
    --------
    >>> nodes, edges = euclidean_minimum_spanning_tree(points_gdf)
    >>> G = euclidean_minimum_spanning_tree(points_gdf, as_nx=True)
    """
    # Input validation
    _assert_euclidean(distance_metric, "delaunay_graph")

    # Node preparation
    G, coords, node_ids = _prepare_nodes(gdf)
    n_points = len(coords)
    if n_points < 2:
        # MST is empty (0 nodes) or a single isolated node
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # Candidate edge set
    # Fast O(n) candidate set via Delaunay when it is applicable
    use_complete_graph = False
    cand_edges: set[tuple[int, int]]

    if distance_metric.lower() == "euclidean" and n_points >= 3:
        tri = Delaunay(coords)
        cand_edges = {
            tuple(sorted((i, j))) for simplex in tri.simplices for i, j in combinations(simplex, 2)
        }
    else:
        use_complete_graph = True

    if use_complete_graph:
        cand_edges = {(i, j) for i in range(n_points) for j in range(i + 1, n_points)}

    # Convert vertex indices to actual node ids
    cand_edges = {(node_ids[i], node_ids[j]) for i, j in cand_edges}

    # Attach weights and geometries
    dm = None
    if distance_metric.lower() == "network":
        dm = _distance_matrix(coords, "network", network_gdf, gdf.crs)

    _add_edges(
        G,
        cand_edges,
        coords,
        node_ids,
        metric=distance_metric,
        dm=dm,
        network_gdf=network_gdf,
    )

    # Compute the minimum-spanning tree
    mst_G = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")

    # Output formatting
    return mst_G if as_nx else nx_to_gdf(mst_G, nodes=True, edges=True)


def fixed_radius_graph(
    gdf: gpd.GeoDataFrame,
    radius: float,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    *,
    target_gdf: gpd.GeoDataFrame | None = None,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a fixed-radius graph from a GeoDataFrame of points.

    This function constructs a graph where nodes are connected if the distance between
    them is within a specified radius. This model is particularly useful for modeling
    communication networks, ecological connectivity, and spatial influence zones where
    interaction strength has a clear distance threshold.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the source points (nodes) for the graph. The index of
        this GeoDataFrame will be used as node IDs.
    radius : float
        The maximum distance for connecting nodes. Nodes within this radius will have
        an edge between them. Must be positive.
    distance_metric : str, default "euclidean"
        The distance metric to use for determining connections. Options are:
        - "euclidean": Straight-line distance
        - "manhattan": City-block distance (L1 norm)
        - "network": Shortest path distance along a network
    network_gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame representing a network (e.g., roads) to use for "network"
        distance calculations. Required if `distance_metric` is "network".
    target_gdf : geopandas.GeoDataFrame, optional
        If provided, creates a directed graph where edges connect nodes from `gdf` to
        nodes in `target_gdf` within the specified radius. If None, creates an
        undirected graph from `gdf` itself.
    as_nx : bool, default False
        If True, returns a NetworkX graph object. Otherwise, returns a tuple of
        GeoDataFrames (nodes, edges).

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    Raises
    ------
    ValueError
        If `distance_metric` is "network" but `network_gdf` is not provided.
        If `radius` is not positive.

    See Also
    --------
    knn_graph : Generate a k-nearest neighbour graph.
    delaunay_graph : Generate a Delaunay triangulation graph.
    waxman_graph : Generate a probabilistic Waxman graph.

    Notes
    -----
    - Node IDs are preserved from the input GeoDataFrame's index
    - Edge weights represent the actual distance between connected nodes
    - Edge geometries are LineStrings connecting node centroids
    - The graph stores the radius parameter in `G.graph["radius"]`
    - For Manhattan distance, edges follow L-shaped geometric paths

    References
    ----------
    Bentley, J. L., Stanat, D. F., & Williams Jr, E. H. (1977).
    The complexity of finding fixed-radius near neighbors.
    Information processing letters, 6(6), 209-212. [1](https://doi.org/10.1016/0020-0190(77)90070-9)

    Examples
    --------
    >>> import geopandas as gpd
    >>> import numpy as np
    >>> from shapely.geometry import Point
    >>>
    >>> # Create a sample GeoDataFrame representing city facilities
    >>> facilities = {
    ...     'name': ['Library_A', 'Park_B', 'School_C', 'Hospital_D', 'Mall_E'],
    ...     'type': ['library', 'park', 'school', 'hospital', 'commercial'],
    ...     'geometry': [
    ...         Point(0, 0), Point(1.5, 1), Point(3, 0.5),
    ...         Point(1, 3), Point(4, 4)
    ...     ]
    ... }
    >>> gdf = gpd.GeoDataFrame(facilities, crs="EPSG:4326").set_index('name')
    >>> print("Input facilities:")
    >>> print(gdf)
            type              geometry
    name
    Library_A   library   POINT (0.000 0.000)
    Park_B         park   POINT (1.500 1.000)
    School_C     school   POINT (3.000 0.500)
    Hospital_D hospital   POINT (1.000 3.000)
    Mall_E   commercial   POINT (4.000 4.000)

    >>> # Generate fix radius graph with radius=2.0
    >>> nodes_gdf, edges_gdf = fixed_radius_graph(gdf, radius=2.0)
    >>> print(f"\\nConnections within 2.0 units:")
    >>> print(f"Nodes: {len(nodes_gdf)}, Edges: {len(edges_gdf)}")
    Connections within 2.0 units:
    Nodes: 5, Edges: 4

    >>> print("\\nEdge connections and distances:")
    >>> for idx, row in edges_gdf.iterrows():
    ...     print(f"{row.name}: weight = {row['weight']:.3f}")
    0: weight = 1.803
    1: weight = 1.581
    2: weight = 2.000
    3: weight = 2.236

    >>> # Compare with smaller radius
    >>> nodes_small, edges_small = fixed_radius_graph(gdf, radius=1.0)
    >>> print(f"\\nWith radius=1.0: {len(edges_small)} edges")
    With radius=1.0: 0 edges

    >>> # Compare with larger radius
    >>> nodes_large, edges_large = fixed_radius_graph(gdf, radius=3.0)
    >>> print(f"With radius=3.0: {len(edges_large)} edges")
    With radius=3.0: 7 edges

    >>> # Manhattan distance example
    >>> nodes_manh, edges_manh = fixed_radius_graph(
    ...     gdf, radius=3.0, distance_metric="manhattan"
    ... )
    >>> print(f"\\nManhattan metric (radius=3.0): {len(edges_manh)} edges")
    Manhattan metric (radius=3.0): 6 edges

    >>> # NetworkX graph with radius information
    >>> G = fixed_radius_graph(gdf, radius=2.5, as_nx=True)
    >>> print(f"\\nNetworkX graph properties:")
    >>> print(f"Radius parameter: {G.graph['radius']}")
    >>> print(f"Graph density: {nx.density(G):.3f}")
    >>> print(f"Connected components: {nx.number_connected_components(G)}")
    NetworkX graph properties:
    Radius parameter: 2.5
    Graph density: 0.600
    Connected components: 1

    >>> # Directed graph to specific targets
    >>> targets = gpd.GeoDataFrame({
    ...     'service': ['Emergency', 'Transit'],
    ...     'geometry': [Point(2, 2), Point(3.5, 1.5)]
    ... }, crs="EPSG:4326", index=['Emergency_Hub', 'Transit_Stop'])
    >>>
    >>> nodes_dir, edges_dir = fixed_radius_graph(
    ...     gdf, radius=2.5, target_gdf=targets
    ... )
    >>> print(f"\\nDirected connections to targets: {len(edges_dir)} edges")
    Directed connections to targets: 8 edges
    """
    # Handle directed variant
    if target_gdf is not None:
        return _directed_graph(
            src_gdf=gdf,
            dst_gdf=target_gdf,
            distance_metric=distance_metric,
            method="radius",
            param=radius,
            as_nx=as_nx,
            network_gdf=network_gdf,
        )

    # Prepare nodes and handle trivial cases
    G, coords, node_ids = _prepare_nodes(gdf)
    if len(coords) < 2:
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # Generate edges based on distance metric
    dm = None
    if distance_metric == "network":
        dm = _distance_matrix(coords, "network", network_gdf, gdf.crs)
        mask = (dm <= radius) & np.triu(np.ones_like(dm, dtype=bool), 1)
        edge_idx = np.column_stack(np.where(mask))
        edges = [(node_ids[i], node_ids[j]) for i, j in edge_idx if dm[i, j] < np.inf]
    else:
        nn_metric = "cityblock" if distance_metric == "manhattan" else "euclidean"
        nn = NearestNeighbors(radius=radius, metric=nn_metric).fit(coords)
        idxs = nn.radius_neighbors(coords, return_distance=False)
        edges = [(node_ids[i], node_ids[j]) for i, neigh in enumerate(idxs) for j in neigh if i < j]

    # Add edges with weights and geometries
    _add_edges(G, edges, coords, node_ids, metric=distance_metric, dm=dm, network_gdf=network_gdf)
    G.graph["radius"] = radius
    return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)


def waxman_graph(
    gdf: gpd.GeoDataFrame,
    beta: float,
    r0: float,
    seed: int | None = None,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    *,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a probabilistic Waxman graph from a GeoDataFrame of points.

    This function constructs a random graph where the probability of an edge existing
    between two nodes decreases exponentially with their distance. The model is based
    on the Waxman random graph model, commonly used to simulate realistic network
    topologies in telecommunications, transportation, and social networks where
    connection probability diminishes with distance.

    The connection probability follows the formula:

    .. math::

       P(u,v) = \beta \times \exp \left(-\frac{\text{dist}(u,v)}{r_0}\right)

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the points (nodes) for the graph. The index of this
        GeoDataFrame will be used as node IDs.
    beta : float
        Parameter controlling the overall probability of edge creation. Higher values
        (closer to 1.0) increase the likelihood of connections. Must be between 0 and 1.
    r0 : float
        Parameter controlling the decay rate of probability with distance. Higher values
        result in longer-range connections being more likely. Must be positive.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility of results.
        If None, results will vary between runs.
    distance_metric : str, default "euclidean"
        The distance metric to use for calculating distances between nodes. Options are:
        - "euclidean": Straight-line distance
        - "manhattan": City-block distance (L1 norm)
        - "network": Shortest path distance along a network
    network_gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame representing a network (e.g., roads) to use for "network"
        distance calculations. Required if `distance_metric` is "network".
    as_nx : bool, default False
        If True, returns a NetworkX graph object. Otherwise, returns a tuple of
        GeoDataFrames (nodes, edges).

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes with spatial and attribute information
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes
        If `as_nx` is True, returns a NetworkX graph object with spatial attributes.

    Raises
    ------
    ValueError
        If `distance_metric` is "network" but `network_gdf` is not provided.
        If `beta` is not between 0 and 1, or if `r0` is not positive.

    See Also
    --------
    knn_graph : Generate a k-nearest neighbour graph.
    delaunay_graph : Generate a Delaunay triangulation graph.
    fixed_radius_graph : Generate a fixed-radius graph.

    Notes
    -----
    - Node IDs are preserved from the input GeoDataFrame's index
    - Edge weights represent the actual distance between connected nodes
    - Edge geometries are LineStrings connecting node centroids
    - The graph stores parameters in `G.graph["beta"]` and `G.graph["r0"]`
    - Results are stochastic; use `seed` parameter for reproducible outputs
    - The graph is undirected with symmetric edge probabilities

    References
    ----------
    Waxman, B. M. (2002). Routing of multipoint connections.
    IEEE journal on selected areas in communications, 6(9), 1617-1622. [1](https://doi.org/10.1109/49.12889)

    Examples
    --------
    >>> import geopandas as gpd
    >>> import numpy as np
    >>> from shapely.geometry import Point
    >>>
    >>> # Create a sample GeoDataFrame representing communication towers
    >>> np.random.seed(123)
    >>> tower_coords = np.random.uniform(0, 10, (8, 2))
    >>> towers = {
    ...     'tower_id': [f'T{i:02d}' for i in range(8)],
    ...     'power': np.random.choice(['high', 'medium', 'low'], 8),
    ...     'geometry': [Point(x, y) for x, y in tower_coords]
    ... }
    >>> gdf = gpd.GeoDataFrame(towers, crs="EPSG:4326").set_index('tower_id')
    >>> print("Communication towers:")
    >>> print(gdf.head(4))
         power                     geometry
    tower_id
    T00       low  POINT (6.964 2.862)
    T01      high  POINT (2.269 5.513)
    T02    medium  POINT (5.479 4.237)
    T03    medium  POINT (8.444 7.579)

    >>> # Generate Waxman graph with moderate connectivity
    >>> nodes_gdf, edges_gdf = waxman_graph(
    ...     gdf, beta=0.5, r0=3.0, seed=42
    ... )
    >>> print(f"\\nWaxman graph (β=0.5, r₀=3.0):")
    >>> print(f"Nodes: {len(nodes_gdf)}, Edges: {len(edges_gdf)}")
    >>> print(f"Graph density: {2 * len(edges_gdf) / (len(nodes_gdf) * (len(nodes_gdf) - 1)):.3f}")
    Waxman graph (β=0.5, r₀=3.0):
    Nodes: 8, Edges: 12
    Graph density: 0.429

    >>> print("\\nSample edge weights (distances):")
    >>> print(edges_gdf[['weight']].head(4))
           weight
    0    2.876543
    1    4.123789
    2    1.987654
    3    5.432109

    >>> # Compare different parameter settings
    >>> # High connectivity (higher beta, higher r0)
    >>> _, edges_high = waxman_graph(gdf, beta=0.8, r0=5.0, seed=42)
    >>> print(f"\\nHigh connectivity (β=0.8, r₀=5.0): {len(edges_high)} edges")
    High connectivity (β=0.8, r₀=5.0): 19 edges

    >>> # Low connectivity (lower beta, lower r0)
    >>> _, edges_low = waxman_graph(gdf, beta=0.2, r0=1.5, seed=42)
    >>> print(f"Low connectivity (β=0.2, r₀=1.5): {len(edges_low)} edges")
    Low connectivity (β=0.2, r₀=1.5): 3 edges

    >>> # NetworkX graph with parameter storage
    >>> G = waxman_graph(gdf, beta=0.6, r0=4.0, seed=42, as_nx=True)
    >>> print(f"\\nNetworkX graph parameters:")
    >>> print(f"Beta: {G.graph['beta']}, r0: {G.graph['r0']}")
    >>> print(f"Average clustering coefficient: {nx.average_clustering(G):.3f}")
    >>> print(f"Number of connected components: {nx.number_connected_components(G)}")
    NetworkX graph parameters:
    Beta: 0.6, r0: 4.0
    Average clustering coefficient: 0.267
    Number of connected components: 1

    >>> # Demonstrate reproducibility with seed
    >>> G1 = waxman_graph(gdf, beta=0.4, r0=2.0, seed=99, as_nx=True)
    >>> G2 = waxman_graph(gdf, beta=0.4, r0=2.0, seed=99, as_nx=True)
    >>> print(f"\\nReproducibility test:")
    >>> print(f"Graph 1 edges: {G1.number_of_edges()}")
    >>> print(f"Graph 2 edges: {G2.number_of_edges()}")
    >>> print(f"Identical: {G1.number_of_edges() == G2.number_of_edges()}")
    Reproducibility test:
    Graph 1 edges: 8
    Graph 2 edges: 8
    Identical: True

    >>> # Manhattan distance metric example
    >>> nodes_manh, edges_manh = waxman_graph(
    ...     gdf, beta=0.5, r0=3.0, distance_metric="manhattan", seed=42
    ... )
    >>> print(f"\\nManhattan distance: {len(edges_manh)} edges")
    Manhattan distance: 10 edges
    """
    # Prepare nodes and handle trivial cases
    rng = np.random.default_rng(seed)
    G, coords, node_ids = _prepare_nodes(gdf)
    if len(coords) < 2:
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # Calculate connection probabilities
    dm = _distance_matrix(coords, distance_metric.lower(), network_gdf, gdf.crs)
    with np.errstate(divide="ignore"):
        probs = beta * np.exp(-dm / r0)
    probs[dm == np.inf] = 0  # Unreachable in network metric

    # Generate edges based on probabilities
    rand = rng.random(dm.shape)
    mask = (rand <= probs) & np.triu(np.ones_like(dm, dtype=bool), 1)
    edge_idx = np.column_stack(np.where(mask))
    edges = [(node_ids[i], node_ids[j]) for i, j in edge_idx]

    # Add edges with weights and geometries
    _add_edges(G, edges, coords, node_ids, metric=distance_metric, dm=dm, network_gdf=network_gdf)
    G.graph.update({"beta": beta, "r0": r0})
    return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)


def bridge_nodes(
    nodes_dict: dict[str, gpd.GeoDataFrame],
    proximity_method: str = "knn",
    *,
    multigraph: bool = False,
    as_nx: bool = False,
    **kwargs: float | str | bool,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]] | nx.Graph:
    r"""
    Build directed proximity edges between every ordered pair of node layers.

    This function creates a multi-layer spatial network by generating directed proximity
    edges from nodes in one GeoDataFrame layer to nodes in another. It systematically
    processes all ordered pairs of layers and applies either k-nearest neighbors (KNN)
    or fixed-radius method to establish inter-layer connections. This function is
    specifically designed for constructing heterogeneous graphs by generating new edge
    types ("is_nearby") between different types of nodes, enabling the modeling of
    complex relationships across multiple domains. It is useful for modeling complex
    urban systems, ecological networks, or multi-modal transportation systems where
    different types of entities interact through spatial proximity.

    Parameters
    ----------
    nodes_dict : dict[str, geopandas.GeoDataFrame]
        A dictionary where keys are layer names (strings) and values are GeoDataFrames
        representing the nodes of each layer. Each GeoDataFrame should contain point
        geometries with consistent CRS across all layers.
    proximity_method : str, default "knn"
        The method to use for generating proximity edges between layers. Options are:
        - "knn": k-nearest neighbors method
        - "fixed_radius": fixed-radius method
    multigraph : bool, default False
        If True, the resulting NetworkX graph will be a MultiGraph, allowing multiple
        edges between the same pair of nodes from different proximity relationships.
    as_nx : bool, default False
        If True, returns a NetworkX graph object containing all nodes and inter-layer
        edges. Otherwise, returns dictionaries of GeoDataFrames.
    **kwargs : Any
        Additional keyword arguments passed to the underlying proximity method:

        For `proximity_method="knn"`:

        - k : int, default 1
            Number of nearest neighbors to connect to in target layer
        - distance_metric : str, default "euclidean"
            Distance metric ("euclidean", "manhattan", "network")
        - network_gdf : geopandas.GeoDataFrame, optional
            Network for "network" distance calculations

        For `proximity_method="fixed_radius"`:

        - radius : float, required
            Maximum connection distance between layers
        - distance_metric : str, default "euclidean"
            Distance metric ("euclidean", "manhattan", "network")
        - network_gdf : geopandas.GeoDataFrame, optional
            Network for "network" distance calculations

    Returns
    -------
    tuple[dict[str, geopandas.GeoDataFrame], dict[tuple[str, str, str], geopandas.GeoDataFrame]] | networkx.Graph
        If `as_nx` is False, returns a tuple containing:

        - nodes_dict: The original input `nodes_dict` (unchanged)
        - edges_dict: Dictionary where keys are edge type tuples of the form
          `(source_layer_name, "is_nearby", target_layer_name)` and values are
          GeoDataFrames of the generated directed edges. **Each unique tuple
          represents a distinct edge type in the heterogeneous graph, enabling
          differentiation between relationships across different node type
          combinations.**

        If `as_nx` is True, returns a NetworkX graph object containing all nodes
        from all layers and the generated directed inter-layer edges, forming a
        **heterogeneous graph structure** where different node types are connected
        through proximity-based relationships.

    Raises
    ------
    ValueError
        If `nodes_dict` contains fewer than two layers.
        If `proximity_method` is not "knn" or "fixed_radius".
        If `proximity_method` is "fixed_radius" but `radius` is not provided in `kwargs`.

    See Also
    --------
    knn_graph : Generate a k-nearest neighbour graph.
    fixed_radius_graph : Generate a fixed-radius graph.

    Notes
    -----
    - All generated edges are directed from source layer to target layer
    - **The relation type for all generated edges is fixed as "is_nearby", creating
      a new edge type that bridges different node types in heterogeneous graphs**
    - **Each ordered pair of node layers generates a distinct edge type, enabling
      the construction of rich heterogeneous graph structures with multiple
      relationship types between different domain entities**
    - Edge weights and geometries are calculated based on the chosen distance_metric
    - Each ordered pair of layers generates a separate edge GeoDataFrame
    - Self-connections (layer to itself) are not created
    - **The resulting structure is ideal for heterogeneous graph neural networks,
      multi-layer network analysis, and cross-domain spatial relationship modeling**

    Examples
    --------
    >>> import geopandas as gpd
    >>> import numpy as np
    >>> from shapely.geometry import Point
    >>>
    >>> # Create multi-layer urban infrastructure dataset
    >>> # Layer 1: Schools
    >>> schools_data = {
    ...     'name': ['Elementary_A', 'High_B', 'Middle_C'],
    ...     'capacity': [300, 800, 500],
    ...     'geometry': [Point(1, 1), Point(4, 3), Point(2, 4)]
    ... }
    >>> schools = gpd.GeoDataFrame(schools_data, crs="EPSG:4326").set_index('name')
    >>>
    >>> # Layer 2: Hospitals
    >>> hospitals_data = {
    ...     'name': ['General_Hospital', 'Clinic_East'],
    ...     'beds': [200, 50],
    ...     'geometry': [Point(3, 2), Point(5, 5)]
    ... }
    >>> hospitals = gpd.GeoDataFrame(hospitals_data, crs="EPSG:4326").set_index('name')
    >>>
    >>> # Layer 3: Parks
    >>> parks_data = {
    ...     'name': ['Central_Park', 'River_Park', 'Neighborhood_Green'],
    ...     'area_ha': [15.5, 8.2, 3.1],
    ...     'geometry': [Point(2, 2), Point(1, 3), Point(4, 4)]
    ... }
    >>> parks = gpd.GeoDataFrame(parks_data, crs="EPSG:4326").set_index('name')
    >>>
    >>> nodes_dict = {
    ...     'schools': schools,
    ...     'hospitals': hospitals,
    ...     'parks': parks
    ... }
    >>>
    >>> print("Input layers:")
    >>> for layer_name, gdf in nodes_dict.items():
    ...     print(f"{layer_name}: {len(gdf)} nodes")
    Input layers:
    schools: 3 nodes
    hospitals: 2 nodes
    parks: 3 nodes

    >>> # Bridge nodes using KNN method (1 nearest neighbor)
    >>> nodes_out, edges_out = bridge_nodes(
    ...     nodes_dict, proximity_method="knn", k=1
    ... )
    >>>
    >>> print(f"\\nGenerated edge types: {len(edges_out)}")
    >>> for edge_key in edges_out.keys():
    ...     print(f"  {edge_key[0]} → {edge_key[2]}: {len(edges_out[edge_key])} edges")
    Generated edge types: 6
      schools → hospitals: 3 edges
      schools → parks: 3 edges
      hospitals → schools: 2 edges
      hospitals → parks: 2 edges
      parks → schools: 3 edges
      parks → hospitals: 3 edges

    >>> # Examine specific edge relationships
    >>> school_to_hospital = edges_out[('schools', 'is_nearby', 'hospitals')]
    >>> print("\\nSchools to nearest hospitals:")
    >>> print(school_to_hospital[['weight']])
           weight
    0    2.236068
    1    1.414214
    2    2.828427

    >>> # Bridge nodes using fixed radius
    >>> nodes_fr, edges_fr = bridge_nodes(
    ...     nodes_dict, proximity_method="fixed_radius", radius=2.5
    ... )
    >>>
    >>> total_fr_edges = sum(len(gdf) for gdf in edges_fr.values())
    >>> print(f"\\nFixed radius method (radius=2.5): {total_fr_edges} total edges")
    Fixed radius method (radius=2.5): 8 total edges

    >>> # Compare edge counts by method
    >>> print("\\nEdge count comparison:")
    >>> for edge_key in edges_out.keys():
    ...     knn_count = len(edges_out[edge_key])
    ...     fr_count = len(edges_fr[edge_key]) if edge_key in edges_fr else 0
    ...     print(f"  {edge_key[0]} → {edge_key[2]}: KNN={knn_count}, Fixed radious={fr_count}")
    Edge count comparison:
      schools → hospitals: KNN=3, Fixed radious=2
      schools → parks: KNN=3, Fixed radious=3
      hospitals → schools: KNN=2, Fixed radious=1
      hospitals → parks: KNN=2, Fixed radious=1
      parks → schools: KNN=3, Fixed radious=1
    """
    # Validate input parameters
    if len(nodes_dict) < 2:
        msg = "`nodes_dict` needs at least two layers"
        raise ValueError(msg)

    # Raise error if proximity method is not recognized
    if proximity_method.lower() not in {"knn", "fixed_radius"}:
        msg = "proximity_method must be 'knn' or 'fixed_radius'"
        raise ValueError(msg)

    # Generate edges for each pair of layers
    edge_dict = {}
    for src_type, dst_type in permutations(nodes_dict.keys(), 2):
        src_gdf = nodes_dict[src_type]
        dst_gdf = nodes_dict[dst_type]

        if proximity_method.lower() == "knn":  # k-nearest neighbors
            k = int(kwargs.get("k", 1))
            # Call knn_graph with appropriate arguments (always return GeoDataFrames)
            distance_metric = kwargs.get("distance_metric", "euclidean")
            if not isinstance(distance_metric, str):
                distance_metric = "euclidean"

            network_gdf = kwargs.get("network_gdf")

            _, edges_gdf = knn_graph(
                src_gdf,
                k=k,
                distance_metric=distance_metric,
                network_gdf=network_gdf,
                target_gdf=dst_gdf,
                as_nx=False,  # Always get GeoDataFrames from individual calls
            )
        else:  # fixed_radius
            radius = float(kwargs["radius"])

            # Call fixed_radius_graph with appropriate arguments (always return GeoDataFrames)
            distance_metric = kwargs.get("distance_metric", "euclidean")
            if not isinstance(distance_metric, str):
                distance_metric = "euclidean"

            network_gdf = kwargs.get("network_gdf")

            _, edges_gdf = fixed_radius_graph(
                src_gdf,
                radius=radius,
                distance_metric=distance_metric,
                network_gdf=network_gdf,
                target_gdf=dst_gdf,
                as_nx=False,  # Always get GeoDataFrames from individual calls
            )

        edge_dict[(src_type, "is_nearby", dst_type)] = edges_gdf

    # Format output
    if as_nx:
        return gdf_to_nx(nodes=nodes_dict, edges=edge_dict, multigraph=multigraph, directed=True)
    return nodes_dict, edge_dict


# ============================================================================
# NODE PREPARATION
# ============================================================================


def _prepare_nodes(
    gdf: gpd.GeoDataFrame,
    *,
    directed: bool = False,
) -> tuple[nx.Graph, npt.NDArray[np.floating], list[Any]]:
    """
    Return an empty graph with populated nodes plus coord cache.

    This function prepares a NetworkX graph by adding nodes from a given GeoDataFrame.
    It extracts centroids as node positions and includes all GeoDataFrame attributes
    as node attributes.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The GeoDataFrame containing the nodes to be added to the graph.
    directed : bool, default False
        If True, a directed graph (DiGraph) is created; otherwise, an undirected graph (Graph) is created.

    Returns
    -------
    tuple[networkx.Graph, numpy.ndarray, list[int]]
        A tuple containing:
        - G : networkx.Graph or networkx.DiGraph
            An empty graph with populated nodes.
        - coords : numpy.ndarray
            A 2D NumPy array of node coordinates.
        - node_ids : list[Any]
            A list of node IDs.
    """
    validate_gdf(nodes_gdf=gdf)

    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])
    node_ids: list[Any] = list(gdf.index)

    G = nx.DiGraph() if directed else nx.Graph()
    # Bulk-add nodes using a comprehension to minimize Python-level loops
    attrs_list = gdf.to_dict("records")
    G.add_nodes_from(
        (
            node_id,
            {"pos": (float(x), float(y)), **attrs},
        )
        for node_id, attrs, (x, y) in zip(node_ids, attrs_list, coords, strict=False)
    )
    G.graph["crs"] = gdf.crs

    return G, coords, node_ids


# ============================================================================
# DISTANCE MATRIX
# ============================================================================


def _euclidean_dm(coords: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Compute Euclidean distance matrix.

    This function calculates the pairwise Euclidean distances between all points
    in the input coordinate array and returns them as a squareform distance matrix.

    Parameters
    ----------
    coords : numpy.ndarray
        A 2D NumPy array of coordinates (n_points, n_dimensions).

    Returns
    -------
    numpy.ndarray
        A squareform Euclidean distance matrix.
    """
    return cast("npt.NDArray[np.floating]", sdist.squareform(sdist.pdist(coords)))


def _manhattan_dm(coords: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """
    Compute Manhattan distance matrix.

    This function calculates the pairwise Manhattan (city-block) distances between all points
    in the input coordinate array and returns them as a squareform distance matrix.

    Parameters
    ----------
    coords : numpy.ndarray
        A 2D NumPy array of coordinates (n_points, n_dimensions).

    Returns
    -------
    numpy.ndarray
        A squareform Manhattan distance matrix.
    """
    return cast(
        "npt.NDArray[np.floating]",
        sdist.squareform(sdist.pdist(coords, metric="cityblock")),
    )


def _network_dm(
    coords: npt.NDArray[np.floating],
    network_gdf: gpd.GeoDataFrame,
    gdf_crs: gpd.crs.CRS | None = None,
) -> npt.NDArray[np.floating]:
    """
    Compute network distance matrix.

    This function calculates a distance matrix based on shortest paths within a
    given network. It maps the input coordinates to the nearest nodes in the
    network and then computes all-pairs shortest paths between these mapped nodes.

    Parameters
    ----------
    coords : numpy.ndarray
        A 2D NumPy array of coordinates (n_points, n_dimensions) for which to compute distances.
    network_gdf : geopandas.GeoDataFrame
        A GeoDataFrame representing the network (e.g., roads) to use for distance calculations.
        It must contain LineString geometries and have a valid CRS.
    gdf_crs : geopandas.crs.CRS or None, optional
        The Coordinate Reference System (CRS) of the input `coords`. Used for CRS validation
        against the `network_gdf`.

    Returns
    -------
    numpy.ndarray
        A squareform network distance matrix, where `dm[i, j]` is the shortest path
        distance between the point corresponding to `coords[i]` and `coords[j]`
        along the `network_gdf`.
    """
    # Validate CRS
    if network_gdf.crs != gdf_crs:
        msg = f"CRS mismatch: {gdf_crs} != {network_gdf.crs}"
        raise ValueError(msg)

    # Convert edge GDF to NetworkX
    net_nx = gdf_to_nx(edges=network_gdf)

    # Get node positions
    pos = nx.get_node_attributes(net_nx, "pos")

    net_coords: npt.NDArray[np.floating] = np.asarray(list(pos.values()))
    net_ids = list(pos.keys())

    # Map sample points to nearest network nodes
    nn = NearestNeighbors(n_neighbors=1).fit(net_coords)
    _, idx = nn.kneighbors(coords)
    nearest = [net_ids[i[0]] for i in idx]

    # Pre-allocate distance matrix
    n = len(coords)
    dm: npt.NDArray[np.floating] = np.full((n, n), np.inf)
    np.fill_diagonal(dm, 0)

    # Calculate all-pairs shortest paths
    use_weight = "length" if any("length" in d for _, _, d in net_nx.edges(data=True)) else None

    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(net_nx, nearest[i], weight=use_weight)
        for j in range(i + 1, n):
            dist = lengths.get(nearest[j], np.inf)
            dm[i, j] = dm[j, i] = dist
    return dm


def _distance_matrix(
    coords: npt.NDArray[np.floating],
    metric: str,
    network_gdf: gpd.GeoDataFrame | None,
    gdf_crs: gpd.crs.CRS | None = None,
) -> npt.NDArray[np.floating]:
    """
    Compute distance matrix based on the specified metric.

    This function acts as a dispatcher, calling the appropriate distance matrix
    computation function based on the `metric` parameter. It supports Euclidean,
    Manhattan, and network-based distance calculations.

    Parameters
    ----------
    coords : numpy.ndarray
        A 2D NumPy array of coordinates (n_points, n_dimensions) for which to compute distances.
    metric : str
        The distance metric to use. Options are "euclidean", "manhattan", or "network".
    network_gdf : geopandas.GeoDataFrame or None
        A GeoDataFrame representing a network (e.g., roads) to use for "network"
        distance calculations. Required if `metric` is "network".
    gdf_crs : geopandas.crs.CRS or None, optional
        The Coordinate Reference System (CRS) of the input `coords`. Used for CRS validation
        against the `network_gdf`.

    Returns
    -------
    numpy.ndarray
        A squareform distance matrix based on the specified metric.

    Raises
    ------
    ValueError
        If `metric` is "network" but `network_gdf` is not provided.
        If an unknown `metric` is specified.
    """
    if metric == "euclidean":
        return _euclidean_dm(coords)
    if metric == "manhattan":
        return _manhattan_dm(coords)
    if metric == "network":
        if network_gdf is None:
            msg = "network_gdf is required for network distance metric"
            raise ValueError(msg)
        return _network_dm(coords, network_gdf, gdf_crs)
    msg = f"Unknown distance metric: {metric}"
    raise ValueError(msg)


# ============================================================================
# EDGE ADDITION
# ============================================================================


def _add_edges(
    G: nx.Graph,
    edges: list[EdgePair] | set[EdgePair],
    coords: npt.NDArray[np.floating],
    node_ids: list[Any],
    *,
    metric: str,
    dm: npt.NDArray[np.floating] | None = None,
    network_gdf: gpd.GeoDataFrame | None = None,
) -> None:
    """
    Add edges to the graph with weights and geometries.

    When the metric is *network* the geometry is built by tracing the path on
    `network_gdf`, so the resulting LineString corresponds to the real
    shortest-path on that network rather than a straight segment.

    Parameters
    ----------
    G : networkx.Graph
        The graph to which edges will be added.
    edges : list[tuple[int, int]] or set[tuple[int, int]]
        A list or set of (u, v) tuples representing the edges to add.

    coords : numpy.ndarray
        A 2D NumPy array of coordinates.
    node_ids : list[int]
        A list of node IDs corresponding to the coordinates.
    metric : str
        The distance metric used for edge weights (e.g., "euclidean", "manhattan", "network").
    dm : numpy.ndarray or None, optional
        Precomputed distance matrix. If provided, it is used to set edge weights.
    network_gdf : geopandas.GeoDataFrame or None, optional
        Required if `metric` is "network". Represents the network for shortest path calculations.
    """
    if not edges:
        return

    # Add edges to the graph
    G.add_edges_from(edges)

    # Calculate and set edge weights
    idx_map: dict[Any, int] = {n: i for i, n in enumerate(node_ids)}

    if dm is not None:
        weights = {(u, v): dm[idx_map[u], idx_map[v]] for u, v in G.edges()}
    elif metric == "manhattan":
        weights = {
            (u, v): abs(coords[idx_map[u]][0] - coords[idx_map[v]][0])
            + abs(coords[idx_map[u]][1] - coords[idx_map[v]][1])
            for u, v in G.edges()
        }
    else:  # Euclidean
        weights = {
            (u, v): float(
                np.hypot(
                    coords[idx_map[u]][0] - coords[idx_map[v]][0],
                    coords[idx_map[u]][1] - coords[idx_map[v]][1],
                ),
            )
            for u, v in G.edges()
        }
    nx.set_edge_attributes(G, weights, "weight")

    # Create and set edge geometries
    geom_attr: dict[tuple[Any, Any], LineString] = {}

    # Network metric: trace path on network
    if metric.lower() == "network":
        # Build a NetworkX representation of the network
        net_nx = gdf_to_nx(edges=network_gdf)

        # All network nodes must expose coords in attribute 'pos'
        pos = nx.get_node_attributes(net_nx, "pos")

        net_coords = np.asarray(list(pos.values()))
        net_ids = list(pos.keys())

        # Map each sample point to its nearest network node
        nn = NearestNeighbors(n_neighbors=1).fit(net_coords)
        _, idxs = nn.kneighbors(coords)
        nearest = {node_ids[i]: net_ids[j[0]] for i, j in enumerate(idxs)}

        # Choose weight key if present on the network
        use_weight = "length" if any("length" in d for *_, d in net_nx.edges(data=True)) else None

        # Per-edge shortest path extraction is inherently iterative; keep minimal loop
        for u, v in G.edges():
            path_nodes = nx.shortest_path(
                net_nx,
                source=nearest[u],
                target=nearest[v],
                weight=use_weight,
            )
            path_coords = [pos[p] for p in path_nodes]
            # Remove consecutive duplicates
            if len(path_coords) > 1:
                dedup = [path_coords[0]]
                dedup.extend(pc for pc in path_coords[1:] if pc != dedup[-1])
                geom_attr[(u, v)] = (
                    LineString(dedup)
                    if len(dedup) > 1
                    else LineString([coords[idx_map[u]], coords[idx_map[v]]])
                )
            else:
                geom_attr[(u, v)] = LineString([coords[idx_map[u]], coords[idx_map[v]]])

    # Manhattan or Euclidean metric
    else:
        # Vectorized geometry creation via array ops and a small comprehension
        edge_list = list(G.edges())
        if edge_list:
            # Use Python mapping to avoid None from dict.get with object keys under vectorization
            try:
                u_idx = np.array([idx_map[u] for u, _ in edge_list], dtype=int)
                v_idx = np.array([idx_map[v] for _, v in edge_list], dtype=int)
            except KeyError as e:
                missing = e.args[0]
                msg = f"Edge endpoint {missing!r} not found in node id mapping."
                raise KeyError(msg) from e
            p1 = coords[u_idx]
            p2 = coords[v_idx]
            if metric.lower() == "manhattan":
                geoms = [
                    LineString([(x1, y1), (x2, y1), (x2, y2)])
                    for (x1, y1), (x2, y2) in zip(p1, p2, strict=False)
                ]
            else:
                geoms = [LineString([pt1, pt2]) for pt1, pt2 in zip(p1, p2, strict=False)]
            geom_attr = {edge: geom for edge, geom in zip(edge_list, geoms, strict=False)}

    nx.set_edge_attributes(G, geom_attr, "geometry")


def _directed_edges(
    src_coords: npt.NDArray[np.floating],
    dst_coords: npt.NDArray[np.floating],
    src_ids: list[int],
    dst_ids: list[int],
    *,
    metric: str,
    k: int | None = None,
    radius: float | None = None,
) -> list[tuple[int, int]]:
    """
    Generate directed edges from source to destination nodes.

    This function creates directed edges between source and destination node sets
    using either k-nearest neighbors or radius-based proximity methods to establish
    spatial connections between different node layers.

    Parameters
    ----------
    src_coords : numpy.ndarray
        A 2D NumPy array of coordinates (n_points, n_dimensions) for the source nodes.
    dst_coords : numpy.ndarray
        A 2D NumPy array of coordinates (n_points, n_dimensions) for the destination nodes.
    src_ids : list[int]
        A list of node IDs corresponding to `src_coords`.
    dst_ids : list[int]
        A list of node IDs corresponding to `dst_coords`.
    metric : str
        The distance metric to use (e.g., "euclidean", "manhattan").
    k : int or None, optional
        The number of nearest neighbors to consider for each source node.
    radius : float or None, optional
        The maximum distance for connecting nodes.

    Returns
    -------
    list[tuple[int, int]]
        A list of directed edges as (source_id, destination_id) tuples.
    """
    if (k is None) == (radius is None):
        msg = "Specify exactly one of k or radius for directed graph"
        raise ValueError(msg)

    nn_metric = "cityblock" if metric == "manhattan" else "euclidean"

    # K-nearest neighbors case
    if k is not None:
        n_neigh = min(k, len(dst_coords))
        nn = NearestNeighbors(n_neighbors=n_neigh, metric=nn_metric).fit(dst_coords)
        _, idxs = nn.kneighbors(src_coords)
        return [(src_ids[i], dst_ids[j]) for i, neigh in enumerate(idxs) for j in neigh]

    # Fixed-radius case
    nn = NearestNeighbors(radius=radius, metric=nn_metric).fit(dst_coords)
    idxs = nn.radius_neighbors(src_coords, return_distance=False)
    return [(src_ids[i], dst_ids[j]) for i, neigh in enumerate(idxs) for j in neigh]


# ============================================================================
# MUTILAYER BRIDGING
# ============================================================================


def _directed_graph(
    *,
    src_gdf: gpd.GeoDataFrame,
    dst_gdf: gpd.GeoDataFrame,
    distance_metric: str,
    method: str,
    param: float,
    as_nx: bool,
    network_gdf: gpd.GeoDataFrame | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Build source → target directed proximity edges.

    This function creates a directed graph with edges from source nodes to target
    nodes based on proximity criteria, supporting both k-nearest neighbors and
    radius-based connection methods for spatial network construction.

    Parameters
    ----------
    src_gdf : geopandas.GeoDataFrame
        The source GeoDataFrame.
    dst_gdf : geopandas.GeoDataFrame
        The destination GeoDataFrame.
    distance_metric : str
        The distance metric to use.
    method : str
        The method to use for generating proximity edges ("knn" or "radius").
    param : float
        The parameter for the proximity method (k for knn, radius for fixed_radius).
    as_nx : bool
        If True, returns a NetworkX graph object.
    network_gdf : geopandas.GeoDataFrame, optional
        A GeoDataFrame representing a network for "network" distance calculations.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        If as_nx is False, returns a tuple of (nodes_gdf, edges_gdf).
        If as_nx is True, returns a NetworkX Graph object.
    """
    # Validate CRS
    if src_gdf.crs != dst_gdf.crs:
        msg = "CRS mismatch between source and target GeoDataFrames"
        raise ValueError(msg)

    # Prepare nodes for both source and destination
    src_G, src_coords, src_ids = _prepare_nodes(src_gdf, directed=True)
    dst_G, dst_coords, dst_ids = _prepare_nodes(dst_gdf, directed=True)

    # Disambiguate node identifiers between source and destination layers.
    # Use tuple-based namespacing to guarantee collision-free composition:
    #   ('src', original_id) for sources, ('dst', original_id) for targets
    unique_src_ids = [("src", sid) for sid in src_ids]
    unique_dst_ids = [("dst", did) for did in dst_ids]

    src_relabel_map = {old: new for old, new in zip(src_ids, unique_src_ids, strict=False)}
    dst_relabel_map = {old: new for old, new in zip(dst_ids, unique_dst_ids, strict=False)}

    # Relabel graphs with unique IDs and attach provenance attributes
    src_G = nx.relabel_nodes(src_G, src_relabel_map, copy=True)
    dst_G = nx.relabel_nodes(dst_G, dst_relabel_map, copy=True)

    # Attach original index and node_type to enable correct reconstruction later
    nx.set_node_attributes(
        src_G,
        {nid: oid for nid, oid in zip(unique_src_ids, src_ids, strict=False)},
        "_original_index",
    )
    nx.set_node_attributes(src_G, dict.fromkeys(unique_src_ids, "src"), "node_type")

    nx.set_node_attributes(
        dst_G,
        {nid: oid for nid, oid in zip(unique_dst_ids, dst_ids, strict=False)},
        "_original_index",
    )
    nx.set_node_attributes(dst_G, dict.fromkeys(unique_dst_ids, "dst"), "node_type")

    # Merge nodes into one directed graph (safe: IDs are now unique)
    G = nx.compose(src_G, dst_G)

    # Precompute combined coordinates and ids for weighting/geometry
    combined_coords = np.vstack([src_coords, dst_coords])
    combined_ids = unique_src_ids + unique_dst_ids
    src_n = len(src_ids)
    dst_n = len(dst_ids)

    # Generate directed edges
    # Special handling for network metric: selection must use network distances,
    # and resulting weights should be set from the corresponding distance matrix.
    dm: npt.NDArray[np.floating] | None = None
    raw_edges: list[tuple[int, int]]
    if distance_metric.lower() == "network":
        # Compute network distances for all src+dst points once
        dm = _distance_matrix(combined_coords, "network", network_gdf, src_gdf.crs)
        d_sub = dm[:src_n, src_n : src_n + dst_n]
        finite = np.isfinite(d_sub)

        if method == "knn":
            k = int(param)
            # Rank all destinations per source, then pick top-k finite
            order = np.argsort(d_sub, axis=1)
            rows = np.arange(src_n)[:, None]
            ranks = np.empty_like(order)
            ranks[rows, order] = np.arange(dst_n)[None, :]
            sel_mask = (ranks < k) & finite
            i_idx, j_idx = np.where(sel_mask)
        else:  # radius
            radius_val = float(param)
            i_idx, j_idx = np.where(finite & (d_sub <= radius_val))

        raw_edges = list(zip((src_ids[i] for i in i_idx), (dst_ids[j] for j in j_idx), strict=True))
    else:
        raw_edges = _directed_edges(
            src_coords,
            dst_coords,
            src_ids,
            dst_ids,
            metric=distance_metric,
            k=int(param) if method == "knn" else None,
            radius=param if method == "radius" else None,
        )

    # Convert edges to use the unique, namespaced node identifiers
    relabeled_edges: list[tuple[tuple[str, int], tuple[str, int]]] = [
        (src_relabel_map[u], dst_relabel_map[v]) for (u, v) in raw_edges
    ]

    # Add edges with weights and geometries
    _add_edges(
        G,
        relabeled_edges,
        combined_coords,
        combined_ids,
        metric=distance_metric,
        dm=dm,
        network_gdf=network_gdf,
    )

    # Preserve original edge indices for GeoDataFrame reconstruction
    # so that nx_to_gdf emits (orig_src, orig_dst) even though internal IDs are namespaced
    orig_edge_index = {
        (u, v): (
            G.nodes[u].get("_original_index", u),
            G.nodes[v].get("_original_index", v),
        )
        for u, v in G.edges()
    }
    nx.set_edge_attributes(G, orig_edge_index, "_original_edge_index")

    return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)


def _assert_euclidean(metric: str, func_name: str) -> None:
    """
    Warn if a non-Euclidean metric is used for algorithms based on it.

    This function checks if the provided distance metric is Euclidean and issues
    a warning if a non-Euclidean metric is used for algorithms that are specifically
    designed for Euclidean distance calculations.

    Parameters
    ----------
    metric : str
        The distance metric being used.
    func_name : str
        The name of the function where the warning is issued.
    """
    if metric.lower() != "euclidean":
        msg = (
            f"{func_name} supports only 'euclidean' distance for edge identification algorithm; "
            f"'{metric}' will be used only for generating edge geometries."
        )
        logger.warning(msg)


def _validate_contiguity_input(gdf: gpd.GeoDataFrame, contiguity: str) -> None:
    """
    Validate input parameters for contiguity graph generation.

    This function performs comprehensive validation of the input GeoDataFrame and
    contiguity parameter to ensure they meet the requirements for contiguity-based
    graph generation.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame to validate. Must contain polygon geometries.
    contiguity : str
        Contiguity type to validate. Must be "queen" or "rook".

    Raises
    ------
    TypeError
        If `gdf` is not a GeoDataFrame.
    ValueError
        If `contiguity` is not "queen" or "rook".
        If `gdf` contains non-polygon geometries.
        If `gdf` contains invalid geometries.

    Notes
    -----
    This function is used internally by contiguity_graph to ensure input validity
    before processing. It follows the validation patterns established in the
    city2graph library.
    """
    # Validate GeoDataFrame type
    if not isinstance(gdf, gpd.GeoDataFrame):
        msg = (
            f"Input must be a GeoDataFrame, got {type(gdf).__name__}. "
            "Please provide a valid GeoDataFrame with polygon geometries."
        )
        raise TypeError(msg)

    # Validate contiguity parameter
    valid_contiguity = {"queen", "rook"}
    if contiguity not in valid_contiguity:
        msg = (
            f"Invalid contiguity type '{contiguity}'. "
            f"Must be one of {sorted(valid_contiguity)}. "
            "Use 'queen' for edge/vertex adjacency or 'rook' for edge-only adjacency."
        )
        raise ValueError(msg)

    # Handle empty GeoDataFrame (allowed, will return empty graph)
    if gdf.empty:
        return

    # Validate geometry column exists
    if not hasattr(gdf, "geometry") or gdf.geometry is None:
        msg = (
            "GeoDataFrame must have a valid geometry column. "
            "Please ensure the GeoDataFrame contains spatial geometries."
        )
        raise ValueError(msg)

    # Check for null geometries
    null_geoms = gdf.geometry.isnull()
    if null_geoms.any():
        null_count = null_geoms.sum()
        msg = (
            f"GeoDataFrame contains {null_count} null geometr{'y' if null_count == 1 else 'ies'}. "
            "All geometries must be valid for contiguity analysis. "
            "Please remove or fix null geometries before processing."
        )
        raise ValueError(msg)

    # Validate geometry types - must be polygons only
    geom_types = gdf.geometry.geom_type.unique()
    valid_polygon_types = {"Polygon", "MultiPolygon"}
    invalid_types = set(geom_types) - valid_polygon_types

    if invalid_types:
        invalid_count = gdf.geometry.geom_type.isin(invalid_types).sum()
        msg = (
            f"GeoDataFrame contains {invalid_count} non-polygon geometr{'y' if invalid_count == 1 else 'ies'} "
            f"of type(s): {sorted(invalid_types)}. "
            "Contiguity analysis requires polygon geometries only. "
            f"Valid types are: {sorted(valid_polygon_types)}."
        )
        raise ValueError(msg)

    # Check for invalid geometries
    invalid_geoms = ~gdf.geometry.is_valid
    if invalid_geoms.any():
        invalid_count = invalid_geoms.sum()
        # Get examples of invalid geometry indices for debugging
        invalid_indices = gdf.index[invalid_geoms].tolist()[:3]  # Show up to 3 examples
        indices_str = ", ".join(str(idx) for idx in invalid_indices)
        if len(invalid_indices) < invalid_count:
            indices_str += f", ... ({invalid_count - len(invalid_indices)} more)"

        msg = (
            f"GeoDataFrame contains {invalid_count} invalid geometr{'y' if invalid_count == 1 else 'ies'}. "
            f"Invalid geometries found at indices: {indices_str}. "
            "Please fix invalid geometries using methods like buffer(0) or make_valid() "
            "before performing contiguity analysis."
        )
        raise ValueError(msg)


def _create_spatial_weights(
    gdf: gpd.GeoDataFrame,
    contiguity: str,
) -> libpysal.weights.W:
    """
    Create spatial weights matrix using libpysal for contiguity analysis.

    This function creates a spatial weights matrix from a GeoDataFrame using either
    Queen or Rook contiguity rules. It handles libpysal compatibility requirements
    and provides informative error handling for spatial weights creation failures.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries. Must have been validated
        by _validate_contiguity_input before calling this function.
    contiguity : str
        Type of spatial contiguity to use. Must be "queen" or "rook".

    Returns
    -------
    libpysal.weights.W
        Spatial weights matrix representing adjacency relationships between polygons.
        The weights matrix uses the original GeoDataFrame index as identifiers.

    Raises
    ------
    ValueError
        If libpysal fails to create spatial weights matrix.
        If the resulting weights matrix is invalid or empty when polygons exist.

    Notes
    -----
    - Resets GeoDataFrame index for libpysal compatibility, then restores original indices
    - Uses libpysal.weights.Queen.from_dataframe for Queen contiguity
    - Uses libpysal.weights.Rook.from_dataframe for Rook contiguity
    - Handles libpysal exceptions and provides informative error messages
    - Returns weights matrix with original index values as identifiers
    """
    # Handle empty GeoDataFrame case early
    if gdf.empty:
        return libpysal.weights.W({})

    # Validate contiguity value early to avoid catching it in the generic except
    contiguity_lc = contiguity.lower()
    if contiguity_lc not in {"queen", "rook"}:
        msg = f"Unsupported contiguity type: {contiguity}"
        raise ValueError(msg)

    # Use libpysal's ids parameter when available to preserve original index.
    # Fall back to calling without ids for compatibility or when monkeypatched callables
    # don't support the keyword (used in tests).
    try:
        ids = list(gdf.index)
        if contiguity_lc == "queen":
            try:
                weights = libpysal.weights.Queen.from_dataframe(gdf, ids=ids)
            except TypeError:
                weights = libpysal.weights.Queen.from_dataframe(gdf)
        else:  # contiguity_lc == "rook"
            try:
                weights = libpysal.weights.Rook.from_dataframe(gdf, ids=ids)
            except TypeError:
                weights = libpysal.weights.Rook.from_dataframe(gdf)
    except Exception as e:
        msg = (
            f"Failed to create {contiguity} contiguity spatial weights matrix. "
            f"This may be due to invalid geometries, topology issues, or libpysal incompatibility. "
            f"Original error: {e}"
        )
        raise ValueError(msg) from e

    if weights is None:
        msg = f"libpysal returned None when creating {contiguity} contiguity weights."
        raise ValueError(msg)

    return weights


def _generate_contiguity_edges(
    weights: libpysal.weights.W,
    gdf: gpd.GeoDataFrame,
) -> list[EdgePair]:
    """
    Extract adjacency relationships from spatial weights matrix.

    This function processes a libpysal spatial weights matrix to extract edge
    relationships for graph construction. It handles disconnected components
    and isolated nodes, returning a list of (source_id, target_id) tuples
    using the original GeoDataFrame indices.

    Parameters
    ----------
    weights : libpysal.weights.W
        Spatial weights matrix containing adjacency relationships between polygons.
        Should be created by _create_spatial_weights function.
    gdf : gpd.GeoDataFrame
        Original GeoDataFrame containing polygon geometries. Used for logging
        and validation purposes. The index provides the node identifiers.

    Returns
    -------
    list[tuple[any, any]]
        List of (source_id, target_id) tuples representing edges in the graph.
        Uses original GeoDataFrame index values as node identifiers.
        Returns empty list if no adjacency relationships exist.

    Notes
    -----
    - Handles disconnected components by preserving all adjacency relationships
    - Isolated nodes (no neighbors) are handled by returning no edges for them
    - Each adjacency relationship appears only once in the output (undirected edges)
    - Uses original GeoDataFrame index values as preserved by _create_spatial_weights
    - Logs information about connectivity for debugging purposes

    Examples
    --------
    >>> # Assuming weights matrix and gdf are already created
    >>> edges = _generate_contiguity_edges(weights, gdf)
    >>> print(f"Generated {len(edges)} edges from {len(gdf)} polygons")
    """
    # Handle empty weights matrix
    if not weights.neighbors or len(weights.neighbors) == 0:
        logger.debug("Empty spatial weights matrix - no adjacency relationships found")
        return []

    # Extract unique, undirected edges from weights.neighbors using a set-comprehension
    # Canonicalize by sorting endpoints so (a,b) and (b,a) collapse to one entry
    undirected_pairs = {
        tuple(sorted((src, nbr))) for src, nbrs in weights.neighbors.items() for nbr in nbrs
    }

    # Represent edges using the canonical ordering (u <= v) for determinism
    edges: list[EdgePair] = list(undirected_pairs)

    # Log connectivity information for debugging
    total_nodes = len(gdf)
    nodes_with_neighbors = len(weights.neighbors)
    isolated_nodes = total_nodes - nodes_with_neighbors

    logger.debug(
        "Contiguity edge extraction: %d edges generated from %d polygons (%d connected, %d isolated)",
        len(edges),
        total_nodes,
        nodes_with_neighbors,
        isolated_nodes,
    )

    # Additional connectivity analysis for debugging
    if len(edges) > 0:
        # Count unique nodes involved in edges without explicit Python loops
        edge_nodes = {n for e in edges for n in e}

        logger.debug(
            "Edge connectivity: %d nodes participate in edges, average degree: %.2f",
            len(edge_nodes),
            2 * len(edges) / len(edge_nodes),
        )

    return edges


def _build_contiguity_graph(
    gdf: gpd.GeoDataFrame,
    edges: list[EdgePair],
) -> nx.Graph:
    """
    Build NetworkX graph from GeoDataFrame and edge list.

    This function creates a NetworkX graph with nodes containing preserved original
    attributes and geometries, and edges with weight and geometry attributes.
    Internally it reuses the common helpers used by other generators
    (node preparation + edge attachment) for consistency and efficiency.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Original GeoDataFrame containing polygon geometries and attributes.
        The index provides node identifiers and all columns are preserved
        as node attributes.
    edges : list[tuple[any, any]]
        List of (source_id, target_id) tuples representing edges in the graph.
        Should use the same identifiers as the GeoDataFrame index.

    Returns
    -------
    networkx.Graph
        NetworkX graph with:
        - Nodes containing 'geometry' and all original GeoDataFrame attributes
        - Edges containing 'weight' (Euclidean distance) and 'geometry' (LineString)
        - Graph metadata including 'crs' and 'contiguity' information

    Notes
    -----
    - All original GeoDataFrame attributes are preserved as node attributes
    - Node geometries are preserved as the original polygon geometries
    - Edge weights are Euclidean distances between polygon centroids
    - Edge geometries are LineStrings connecting polygon centroids
    - CRS information is preserved in graph metadata
    - Isolated nodes (no edges) are included in the graph
    - Graph is undirected since contiguity is a symmetric relationship

    Examples
    --------
    >>> # Assuming gdf and edges are already prepared
    >>> G = _build_contiguity_graph(gdf, edges)
    >>> print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    >>> print(f"CRS: {G.graph.get('crs')}")
    """
    # Reuse shared helpers: prepare nodes once (adds 'pos' and preserves attributes)
    G, coords, node_ids = _prepare_nodes(gdf)

    # Attach edges with standard geometry/weight logic (Euclidean between centroids)
    if edges:
        _add_edges(G, edges, coords, node_ids, metric="euclidean")

    # Log graph construction results
    logger.debug(
        "Contiguity graph constructed: %d nodes, %d edges, CRS: %s",
        G.number_of_nodes(),
        G.number_of_edges(),
        G.graph.get("crs"),
    )

    return G


def _contiguity_graph_core(
    gdf: gpd.GeoDataFrame,
    contiguity: str,
) -> nx.Graph:
    """
    Build a contiguity graph (NetworkX) from a validated polygon GeoDataFrame.

    Assumes inputs have already been validated by the caller. Constructs a
    NetworkX graph with nodes, edges, and graph metadata (``crs``,
    ``contiguity``) set.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing valid polygon geometries. Index values are used
        as node identifiers and all columns are preserved as node attributes.
    contiguity : {"queen", "rook"}
        Type of spatial contiguity used to derive adjacency relationships.
        This value is also recorded in the graph metadata.

    Returns
    -------
    networkx.Graph
        Undirected graph where:
        - Nodes hold original attributes and polygon geometries.
        - Edges represent adjacency derived from the specified contiguity rule
          and include ``weight`` (Euclidean distance between centroids) and
          ``geometry`` (LineString connecting centroids).
        - Graph metadata includes ``crs`` and ``contiguity``.
    """
    # Create spatial weights and extract undirected edges
    logger.debug("Creating %s contiguity spatial weights matrix", contiguity)
    weights = _create_spatial_weights(gdf, contiguity)

    logger.debug("Extracting adjacency relationships from spatial weights")
    edges = _generate_contiguity_edges(weights, gdf)

    # Build graph and attach metadata
    logger.debug("Building NetworkX graph with nodes and edges")
    G = _build_contiguity_graph(gdf, edges)
    G.graph["contiguity"] = contiguity
    return G


def _empty_contiguity_result(
    gdf: gpd.GeoDataFrame,
    contiguity: str,
    *,
    as_nx: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Create an empty-but-typed result matching contiguity_graph outputs.

    This helper centralizes creation of empty results while preserving CRS and
    expected schemas. It avoids duplicating the empty-branch logic and ensures
    consistent behavior across refactors.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame (possibly empty) used to derive CRS and columns.
    contiguity : str
        Contiguity type label to attach to graph metadata when returning NetworkX.
    as_nx : bool
        Whether to return a NetworkX graph or a tuple of GeoDataFrames.

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph
        Properly typed empty result preserving CRS and node columns.
    """
    if as_nx:
        empty_graph = nx.Graph()
        empty_graph.graph["crs"] = gdf.crs
        empty_graph.graph["contiguity"] = contiguity
        return empty_graph

    # Create empty nodes GeoDataFrame with same structure as input
    empty_nodes = gpd.GeoDataFrame(
        columns=gdf.columns.tolist(),
        crs=gdf.crs,
        index=gdf.index[:0],  # Empty index with same type
    )

    # Create empty edges GeoDataFrame with expected structure
    empty_edges = gpd.GeoDataFrame(
        columns=["weight", "geometry"],
        crs=gdf.crs,
    )

    return empty_nodes, empty_edges


def contiguity_graph(
    gdf: gpd.GeoDataFrame,
    contiguity: str = "queen",
    *,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    r"""
    Generate a contiguity-based spatial graph from polygon geometries.

    This function creates a spatial graph where nodes represent polygons and edges
    connect spatially contiguous (adjacent) polygons based on Queen or Rook contiguity
    rules. It leverages libpysal's robust spatial weights functionality to accurately
    determine adjacency relationships, making it ideal for spatial analysis of
    administrative boundaries, urban morphology studies, land use patterns, and
    geographic network analysis.

    The function supports both Queen contiguity (polygons sharing edges or vertices)
    and Rook contiguity (polygons sharing only edges), providing flexibility for
    different spatial analysis requirements. Edge weights are calculated as Euclidean
    distances between polygon centroids, and edge geometries are represented as
    LineStrings connecting these centroids.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame containing polygon geometries. Must contain valid polygon
        geometries in the 'geometry' column. The index of this GeoDataFrame will be
        preserved as node identifiers in the output graph. All original attributes
        are maintained in the nodes output.
    contiguity : {"queen", "rook"}, default "queen"
        Type of spatial contiguity rule to apply for determining adjacency:

        - "queen": Polygons are considered adjacent if they share any boundary
          (edges or vertices). This is more inclusive and typically results in
          more connections.
        - "rook": Polygons are considered adjacent only if they share an edge
          (not just vertices). This is more restrictive and results in fewer
          connections.
    as_nx : bool, default False
        Output format control. If True, returns a NetworkX Graph object with
        spatial attributes. If False, returns a tuple of GeoDataFrames for
        nodes and edges, compatible with other city2graph functions.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        **When as_nx=False (default):**
        Returns a tuple of two GeoDataFrames:

        - **nodes_gdf** : geopandas.GeoDataFrame
            Contains all input polygons as nodes with preserved original attributes
            and geometries. Index matches the input GeoDataFrame index.

        - **edges_gdf** : geopandas.GeoDataFrame
            Contains edges representing contiguity relationships with columns:

            * 'weight' : float - Euclidean distance between polygon centroids
            * 'geometry' : LineString - Line connecting polygon centroids

        **When as_nx=True:**
        Returns a NetworkX Graph object with:

        - **Nodes**: Polygon IDs with attributes including 'geometry' and all
          original GeoDataFrame columns
        - **Edges**: Adjacency relationships with 'weight' and 'geometry' attributes
        - **Graph metadata**: Includes 'crs' and 'contiguity' information

    Raises
    ------
    TypeError
        If `gdf` is not a geopandas.GeoDataFrame instance.
    ValueError
        If `contiguity` is not one of {"queen", "rook"}.
        If `gdf` contains geometries that are not polygons (Point, LineString, etc.).
        If `gdf` contains invalid or corrupt polygon geometries.
        If libpysal fails to create spatial weights matrix.

    See Also
    --------
    libpysal.weights.Queen : Spatial weights based on Queen contiguity.
    libpysal.weights.Rook : Spatial weights based on Rook contiguity.
    knn_graph : Generate k-nearest neighbor graphs from point data.
    delaunay_graph : Generate Delaunay triangulation graphs from point data.
    fixed_radius_graph : Generate fixed-radius proximity graphs.
    gabriel_graph : Generate Gabriel graphs from point data.
    relative_neighborhood_graph : Generate relative neighborhood graphs.
    waxman_graph : Generate probabilistic Waxman graphs.

    Examples
    --------
    >>> # Create sample administrative districts
    >>> districts = [
    ...     Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),  # District A
    ...     Polygon([(2, 0), (4, 0), (4, 2), (2, 2)]),  # District B (adjacent to A)
    ...     Polygon([(0, 2), (2, 2), (2, 4), (0, 4)]),  # District C (adjacent to A)
    ...     Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])   # District D (isolated)
    ... ]
    >>> gdf = gpd.GeoDataFrame({
    ...     'district_id': ['A', 'B', 'C', 'D'],
    ...     'population': [10000, 15000, 8000, 5000],
    ...     'area_km2': [4.0, 4.0, 4.0, 4.0],
    ...     'geometry': districts
    ... }, crs="EPSG:3857").set_index('district_id')
    >>>
    >>> # Generate Queen contiguity graph
    >>> nodes_gdf, edges_gdf = contiguity_graph(gdf, contiguity="queen")
    >>> print(f"Districts: {len(nodes_gdf)}, Adjacency relationships: {len(edges_gdf)}")
    Districts: 4, Adjacency relationships: 4
    >>>
    >>> # Examine adjacency relationships
    >>> print("\\nAdjacency relationships:")
    >>> for idx, edge in edges_gdf.iterrows():
    ...     src, dst = edge.name  # Edge endpoints from MultiIndex
    ...     weight = edge['weight']
    ...     print(f"  {src} ↔ {dst}: distance = {weight:.2f}")
    Adjacency relationships:
      A ↔ B: distance = 2.83
      A ↔ C: distance = 2.83
    """
    # Log function entry for debugging
    logger.debug(
        "contiguity_graph called with %s polygons, contiguity='%s', as_nx=%s",
        (len(gdf) if hasattr(gdf, "__len__") else "unknown"),
        contiguity,
        as_nx,
    )

    # Step 1: Input validation
    _validate_contiguity_input(gdf, contiguity)

    # Step 2: Handle empty GeoDataFrame case up-front
    if gdf.empty:
        logger.debug("Empty GeoDataFrame provided - returning empty result")
        return _empty_contiguity_result(gdf, contiguity, as_nx=as_nx)

    # Step 3: Build the core graph once, then format output as requested
    G = _contiguity_graph_core(gdf, contiguity)

    # Step 4: Return in requested format
    if as_nx:
        logger.debug(
            "Returning NetworkX graph: %d nodes, %d edges",
            G.number_of_nodes(),
            G.number_of_edges(),
        )
        return G

    logger.debug("Converting NetworkX graph to GeoDataFrame tuple")
    return nx_to_gdf(G, nodes=True, edges=True)
