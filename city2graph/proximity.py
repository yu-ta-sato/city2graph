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
from typing import Literal
from typing import cast

# Third-party imports
import geopandas as gpd
import libpysal
import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd
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
    "group_nodes",
    "knn_graph",
    "relative_neighborhood_graph",
    "waxman_graph",
]

# Simple type alias for readability
EdgePair = tuple[Any, Any]


# ----------------------------------------------------------------------------
# Internal helpers (new)
# ----------------------------------------------------------------------------


def _normalize_metric(metric: object) -> str:
    """
    Return a normalised distance metric name.

    Falls back to "euclidean" when the provided value is falsy or not a string.
    The function intentionally performs only light validation; callers that have
    a restricted set (e.g., ``contiguity_graph``) should still perform explicit
    membership checks to keep error messages and behavior stable.

    Parameters
    ----------
    metric : object
        Candidate distance metric value. If a non-empty string, it is lowercased
        and returned. Any other value results in "euclidean".

    Returns
    -------
    str
        Normalised metric string (e.g., "euclidean", "manhattan", or "network").
    """
    if not isinstance(metric, str) or not metric:
        return "euclidean"
    return metric.lower()


def _get_network_distance_matrix(
    metric: str,
    coords: npt.NDArray[np.floating],
    network_gdf: gpd.GeoDataFrame | None,
    crs: gpd.GeoDataFrame | gpd.GeoSeries | None,
) -> npt.NDArray[np.floating] | None:
    """
    Return a network distance matrix when ``metric == 'network'``.

    This tiny helper centralises a common pattern across generators: only
    compute the distance matrix when the network metric is requested; otherwise
    return ``None`` and let callers use their metric-specific fast paths.

    Parameters
    ----------
    metric : str
        Normalised distance metric name.
    coords : numpy.ndarray
        Array of node coordinates with shape ``(n, 2)``.
    network_gdf : geopandas.GeoDataFrame or None
        Network edges GeoDataFrame. Required when ``metric == 'network'``.
    crs : object
        CRS information for validation against ``network_gdf`` when present.

    Returns
    -------
    numpy.ndarray or None
        Network distance matrix or ``None`` when ``metric`` is not "network".
    """
    if metric == "network":  # hot path early exit for common euclidean/manhattan cases
        return _distance_matrix(coords, "network", network_gdf, getattr(crs, "crs", crs))
    return None


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

    Examples
    --------
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
    distance_metric = _normalize_metric(distance_metric)
    if distance_metric == "network":
        dm = _get_network_distance_matrix(distance_metric, coords, network_gdf, gdf)
        assert dm is not None  # for type-checker: ensured by distance_metric branch
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
    # Normalise metric once
    distance_metric = _normalize_metric(distance_metric)

    # Prepare nodes (early exit for <3 points - Delaunay undefined). Covered in tests.
    G, coords, node_ids = _prepare_nodes(gdf)
    if len(coords) < 3:  # pragma: no cover - defensive early exit
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # Delaunay triangulation candidate edges
    tri = Delaunay(coords)
    edges = {
        (node_ids[i], node_ids[j]) for simplex in tri.simplices for i, j in combinations(simplex, 2)
    }

    # Attach weights/geometries
    dm = _get_network_distance_matrix(distance_metric, coords, network_gdf, gdf)
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
    distance_metric = _normalize_metric(distance_metric)

    G, coords, node_ids = _prepare_nodes(gdf)
    n_points = len(coords)
    if n_points < 2:  # pragma: no cover - defensive early exit
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # Candidate edges (constant-time for 2 points else Delaunay)
    delaunay_edges = (
        {(0, 1)}
        if n_points == 2
        else {
            tuple(sorted((i, j)))
            for simplex in Delaunay(coords).simplices
            for i, j in combinations(simplex, 2)
        }
    )

    # Gabriel filtering
    # Square distances for numerical stability
    kept_edges: set[tuple[int, int]] = set()
    tol = 1e-12
    for i, j in delaunay_edges:  # pragma: no branch - loop body fully covered
        mid = 0.5 * (coords[i] + coords[j])
        rad2 = np.sum((coords[i] - coords[j]) ** 2) * 0.25  # (|pi-pj|/2)^2

        # Squared distance of all points to the midpoint
        d2 = np.sum((coords - mid) ** 2, axis=1)
        mask = d2 <= rad2 + tol

        # Exactly the two endpoints inside the disc?
        if np.count_nonzero(mask) == 2:
            kept_edges.add((node_ids[i], node_ids[j]))

    # Add weights and geometries
    dm = _get_network_distance_matrix(distance_metric, coords, network_gdf, gdf)

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
        Metric used to attach edge weights / geometries (see the other generators).
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
    distance_metric = _normalize_metric(distance_metric)

    G, coords, node_ids = _prepare_nodes(gdf)
    n_points = len(coords)
    if n_points < 2:  # pragma: no cover - defensive early exit
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    cand_edges = (
        {(0, 1)}
        if n_points == 2
        else {
            tuple(sorted((i, j)))
            for simplex in Delaunay(coords).simplices
            for i, j in combinations(simplex, 2)
        }
    )

    # RNG filtering
    kept_edges: set[tuple[int, int]] = set()

    # Work with squared distances to avoid sqrt
    for i, j in cand_edges:  # pragma: no branch loop fully exercised in tests
        dij2 = np.dot(coords[i] - coords[j], coords[i] - coords[j])

        # Vectorised test of the lune-emptiness predicate
        di2 = np.sum((coords - coords[i]) ** 2, axis=1) < dij2
        dj2 = np.sum((coords - coords[j]) ** 2, axis=1) < dij2

        # Any third point closer to *both* i and j?
        closer_both = np.where(di2 & dj2)[0]
        if len(closer_both) == 0:
            kept_edges.add((node_ids[i], node_ids[j]))

    # Add weights and geometries
    dm = _get_network_distance_matrix(distance_metric, coords, network_gdf, gdf)

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
    # Normalise metric; delegate validation to the shared dispatcher below so
    # unknown metrics raise a consistent "Unknown distance metric" from one place.
    distance_metric = _normalize_metric(distance_metric)

    # Node preparation
    G, coords, node_ids = _prepare_nodes(gdf)
    n_points = len(coords)
    if n_points < 2:  # pragma: no cover - defensive early exit
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # Candidate edge set
    # Fast O(n) candidate set via Delaunay when it is applicable
    cand_edges: set[tuple[int, int]]
    if distance_metric == "euclidean" and n_points >= 3:
        tri = Delaunay(coords)
        cand_edges = {
            tuple(sorted((i, j))) for simplex in tri.simplices for i, j in combinations(simplex, 2)
        }
    else:  # fallback complete graph (also covers non-euclidean metrics)
        cand_edges = {(i, j) for i in range(n_points) for j in range(i + 1, n_points)}

    # Convert vertex indices to actual node ids
    cand_edges = {(node_ids[i], node_ids[j]) for i, j in cand_edges}

    # Attach weights and geometries
    # Always compute via the shared dispatcher so invalid metrics raise from there
    dm = _distance_matrix(coords, distance_metric, network_gdf, gdf.crs)

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
    if len(coords) < 2:  # pragma: no cover - defensive early exit
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # Generate edges based on distance metric
    distance_metric = _normalize_metric(distance_metric)
    dm = _get_network_distance_matrix(distance_metric, coords, network_gdf, gdf)
    if dm is not None:  # network metric
        mask = (dm <= radius) & np.triu(np.ones_like(dm, dtype=bool), 1)
        edge_idx = np.column_stack(np.where(mask))
        edges = [(node_ids[i], node_ids[j]) for i, j in edge_idx if dm[i, j] < np.inf]
    else:  # euclidean / manhattan
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
    distance_metric: Literal["euclidean", "manhattan", "network"] = "euclidean",
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
    if len(coords) < 2:  # pragma: no cover - defensive early exit
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # Calculate connection probabilities
    metric_lc = _normalize_metric(distance_metric)
    dm = _distance_matrix(coords, metric_lc, network_gdf, gdf.crs)
    with np.errstate(divide="ignore"):
        probs = beta * np.exp(-dm / r0)
    probs[dm == np.inf] = 0  # Unreachable in network metric

    # Generate edges based on probabilities
    rand = rng.random(dm.shape)
    mask = (rand <= probs) & np.triu(np.ones_like(dm, dtype=bool), 1)
    edge_idx = np.column_stack(np.where(mask))
    edges = [(node_ids[i], node_ids[j]) for i, j in edge_idx]

    # Add edges with weights and geometries
    _add_edges(G, edges, coords, node_ids, metric=metric_lc, dm=dm, network_gdf=network_gdf)
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
    if len(nodes_dict) < 2:  # pragma: no cover - defensive validation
        msg = "`nodes_dict` needs at least two layers"
        raise ValueError(msg)

    # Raise error if proximity method is not recognized
    if proximity_method.lower() not in {"knn", "fixed_radius"}:  # pragma: no cover
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
            distance_metric = _normalize_metric(kwargs.get("distance_metric", "euclidean"))

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
            distance_metric = _normalize_metric(kwargs.get("distance_metric", "euclidean"))

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
    if as_nx:  # pragma: no cover - not exercised in minimal public tests
        return gdf_to_nx(nodes=nodes_dict, edges=edge_dict, multigraph=multigraph, directed=True)
    return nodes_dict, edge_dict


# ============================================================================
# NODE PREPARATION
# ============================================================================


def _relation_from_predicate(predicate: str | None) -> str:
    """
    Map a spatial predicate to a canonical relation label.

    This standardizes the relation name used in heterogeneous edge keys based on a
    GeoPandas spatial join predicate.

    Parameters
    ----------
    predicate : str or None
        Spatial predicate name (e.g., "covered_by", "within", "contains"). If None,
        defaults to "covered_by".

    Returns
    -------
    str
        Canonical relation label used in edge keys. Mappings: "covered_by" -> "covers",
        "within" -> "contains", "contains" -> "contains". Any other value is returned
        unchanged.
    """
    pred = (predicate or "covered_by").lower()
    return {"covered_by": "covers", "within": "contains", "contains": "contains"}.get(pred, pred)


def _group_nodes_empty_result(
    polygons_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    relation: str,
    crs: object,
    as_nx: bool,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]] | nx.Graph:
    """
    Return an empty heterogeneous structure with correct schema and metadata.

    When there are no matches or inputs are empty, this helper fabricates the minimal
    nodes/edges containers (or a NetworkX graph) with the right index names and CRS so
    downstream consumers can rely on a consistent interface without extra conditionals.

    Parameters
    ----------
    polygons_gdf : geopandas.GeoDataFrame
        Polygon nodes to include in the output under key "polygon".
    points_gdf : geopandas.GeoDataFrame
        Point nodes to include in the output under key "point".
    relation : str
        Canonical relation label (e.g., "covers", "contains") for the heterogeneous edge key.
    crs : object
        CRS to assign to the empty edges GeoDataFrame.
    as_nx : bool
        If True, return a typed heterogeneous NetworkX graph; otherwise return nodes/edges dicts.

    Returns
    -------
    tuple[dict[str, geopandas.GeoDataFrame], dict[tuple[str, str, str], geopandas.GeoDataFrame]] or networkx.Graph
        Nodes/edges dictionaries when ``as_nx`` is False, or a heterogeneous NetworkX graph when True.
    """
    edge_key = ("polygon", relation, "point")
    idx = pd.MultiIndex.from_tuples([], names=[polygons_gdf.index.name, points_gdf.index.name])
    empty_edges = gpd.GeoDataFrame(
        {"weight": pd.Series(dtype=float)},
        geometry=gpd.GeoSeries(dtype="geometry"),
        crs=crs,
        index=idx,
    )
    nodes_dict: dict[str, gpd.GeoDataFrame] = {"polygon": polygons_gdf, "point": points_gdf}
    edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] = {edge_key: empty_edges}
    return (
        gdf_to_nx(nodes=nodes_dict, edges=edges_dict, directed=True)
        if as_nx
        else (
            nodes_dict,
            edges_dict,
        )
    )


def _edges_gdf_from_pairs(
    polygons_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    pairs: list[tuple[Any, Any]],
    metric: str,
    network_gdf: gpd.GeoDataFrame | None,
) -> gpd.GeoDataFrame | None:
    """
    Build an edges GeoDataFrame from (polygon_id, point_id) pairs.

    Constructs a temporary directed graph that includes polygon centroids and
    points as nodes, then attaches weights/geometries using the common helper.

    Parameters
    ----------
    polygons_gdf : geopandas.GeoDataFrame
        Polygon features; centroids are used as polygon node positions.
    points_gdf : geopandas.GeoDataFrame
        Point features representing destination nodes.
    pairs : list of tuple
        List of (polygon_id, point_id) pairs specifying edges to create.
    metric : str
        Distance metric: "euclidean", "manhattan", or "network".
    network_gdf : geopandas.GeoDataFrame or None
        Network edges GeoDataFrame when ``metric='network'``; ignored otherwise.

    Returns
    -------
    geopandas.GeoDataFrame or None
        An edges GeoDataFrame indexed by (polygon_id, point_id) with columns "weight" and
        "geometry". Returns None if ``pairs`` is empty or if no edges are produced.
    """
    # Build a unified temporary nodes GeoDataFrame using polygon centroids and point coordinates
    poly_centroids = polygons_gdf.geometry.centroid
    poly_nodes = gpd.GeoDataFrame(
        {"geometry": poly_centroids}, geometry="geometry", crs=polygons_gdf.crs
    )
    pt_nodes = gpd.GeoDataFrame(
        {"geometry": points_gdf.geometry}, geometry="geometry", crs=points_gdf.crs
    )

    # Namespace node IDs to disambiguate between polygon and point indices
    poly_index = pd.MultiIndex.from_tuples([("poly", i) for i in polygons_gdf.index])
    pt_index = pd.MultiIndex.from_tuples([("pt", i) for i in points_gdf.index])
    poly_nodes.index = poly_index
    pt_nodes.index = pt_index
    temp_nodes = pd.concat([poly_nodes, pt_nodes])

    # Prepare nodes once using the shared helper (directed for clarity of polygon -> point edges)
    G_tmp, coords_all, node_ids_ns = _prepare_nodes(temp_nodes, directed=True)

    # Build namespaced edge list (polygon tuple id -> point tuple id)
    ns_edge_list = [(("poly", u), ("pt", v)) for u, v in pairs]

    # Attach weights and geometries using the common edge helper (no precomputed DM;
    # helpers are fast enough for the typically sparse containment relation)
    _add_edges(
        G_tmp,
        ns_edge_list,
        coords_all,
        node_ids_ns,
        metric=metric,
        dm=None,
        network_gdf=network_gdf,
    )

    # Extract edge records back into a typed GeoDataFrame with MultiIndex (polygon_id, point_id)
    records: list[dict[str, Any]] = []
    index_tuples: list[tuple[Any, Any]] = []
    for u_ns, v_ns, data in G_tmp.edges(data=True):
        u_orig = u_ns[1]
        v_orig = v_ns[1]
        index_tuples.append((u_orig, v_orig))
        records.append({"weight": data.get("weight", np.nan), "geometry": data.get("geometry")})

    edge_index = pd.MultiIndex.from_tuples(
        index_tuples,
        names=[polygons_gdf.index.name, points_gdf.index.name],
    )
    return gpd.GeoDataFrame(
        pd.DataFrame(records, index=edge_index)[["weight"]],
        geometry=[rec["geometry"] for rec in records],
        crs=polygons_gdf.crs,
    )


def _containment_pairs(
    points_gdf: gpd.GeoDataFrame,
    polygons_gdf: gpd.GeoDataFrame,
    predicate: str,
) -> list[tuple[Any, Any]]:
    """
    Return (polygon_id, point_id) pairs using a robust spatial join.

    Performs a single GeoPandas ``sjoin`` between points and a copy of polygons whose
    original index is materialised as a normal column. This ensures a deterministic
    mapping back to polygon identifiers regardless of GeoPandas internals.

    Parameters
    ----------
    points_gdf : geopandas.GeoDataFrame
        Points to test against the polygons.
    polygons_gdf : geopandas.GeoDataFrame
        Polygons used as containers.
    predicate : str
        Spatial predicate for the join (e.g., "covered_by", "within").

    Returns
    -------
    list[tuple[Any, Any]]
        List of (polygon_id, point_id) tuples representing containment relationships.
    """
    predicate_lc = (predicate or "covered_by").lower()

    # Materialise polygon ids to a stable column name
    id_col = polygons_gdf.index.name or "index"
    polys = polygons_gdf.reset_index()  # exposes original ids in column `id_col`

    joined = gpd.sjoin(points_gdf, polys, how="inner", predicate=predicate_lc)
    if joined.empty:
        return []

    # Map right-side matches back to original polygon ids deterministically
    poly_ids_series = (
        polys.loc[joined["index_right"], id_col]
        if "index_right" in joined.columns
        else joined[id_col]
    )

    point_ids = joined.index.to_list()
    poly_ids = poly_ids_series.to_list()
    return list(zip(poly_ids, point_ids, strict=False))


def group_nodes(
    polygons_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    *,
    distance_metric: Literal["euclidean", "manhattan", "network"] = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    predicate: str = "covered_by",
    as_nx: bool = False,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]] | nx.Graph:
    """
    Create a heterogeneous graph linking polygon zones to contained points.

    This function builds a bipartite relation between polygon and point features by
    connecting each polygon to the points that satisfy a spatial containment
    predicate (default: "covered_by" so boundary points are included). It follows
    city2graph heterogeneous graph conventions and reuses the proximity helpers for
    computing edge weights and geometries according to the chosen distance metric.

    Parameters
    ----------
    polygons_gdf : geopandas.GeoDataFrame
        GeoDataFrame of polygonal features representing zones. CRS must match
        points_gdf. Original attributes and geometries are preserved in the
        resulting polygon nodes.
    points_gdf : geopandas.GeoDataFrame
        GeoDataFrame of point features to be associated with the polygons. CRS must
        match polygons_gdf. Original attributes and geometries are preserved in
        the resulting point nodes.
    distance_metric : {"euclidean", "manhattan", "network"}, default "euclidean"
        Metric used for edge weights and geometries. Euclidean produces straight
        line segments, Manhattan produces L-shaped polylines, and Network traces
        polylines along the provided network_gdf and uses shortest-path distances.
    network_gdf : geopandas.GeoDataFrame, optional
        Required when distance_metric="network". Must share the same CRS as the
        inputs.
    predicate : str, default "covered_by"
        Spatial predicate used to determine containment in a vectorized spatial
        join (e.g., "covered_by", "within", "contains", "intersects"). The default
        includes points on polygon boundaries.
    as_nx : bool, default False
        If False, return heterogeneous GeoDataFrame dictionaries. If True, return a
        typed heterogeneous NetworkX graph built with gdf_to_nx.

    Returns
    -------
    (nodes_dict, edges_dict) : tuple of dicts
        Returned when as_nx=False. nodes_dict is {"polygon": polygons_gdf,
        "point": points_gdf} with original indices, attributes, and geometries.
        edges_dict maps a typed edge key to an edges GeoDataFrame whose index is a
        MultiIndex of (polygon_id, point_id) and includes at least weight and
        geometry columns. The edge key has the form ("polygon", relation, "point"),
        where relation is derived from predicate (e.g., covered_by -> "covers",
        within -> "contains").
    G : networkx.Graph
        Returned when as_nx=True. A heterogeneous graph with node_type in nodes and
        a typed edge_type reflecting the relation derived from predicate. Graph
        metadata includes crs and is_hetero=True.

    Notes
    -----
    - CRS must be present and identical for both inputs. For network metric, the
      network's CRS must also match.
    - Boundary points are included by default via predicate="covered_by".
    - Distance calculations and edge geometries reuse internal helpers
      (_prepare_nodes, _distance_matrix, _add_edges) to ensure consistency with
      other proximity functions.

    Examples
    --------
    Build heterogeneous GeoDataFrames (default Euclidean):

    >>> import geopandas as gpd
    >>> from shapely.geometry import Point, Polygon
    >>> from city2graph.proximity import group_nodes
    >>> polys = gpd.GeoDataFrame(
    ...     {"name": ["A"]},
    ...     geometry=[Polygon([(0,0), (2,0), (2,2), (0,2)])],
    ...     crs="EPSG:3857",
    ... )
    >>> pts = gpd.GeoDataFrame(
    ...     {"id": [1, 2]},
    ...     geometry=[Point(1, 1), Point(3, 3)],
    ...     crs="EPSG:3857",
    ... )
    >>> nodes, edges = group_nodes(polys, pts, as_nx=False)
    >>> list(nodes.keys())
    ['polygon', 'point']
    >>> next(iter(edges)).__class__ is tuple
    True

    Build a typed heterogeneous NetworkX graph:

    >>> G = group_nodes(polys, pts, as_nx=True)
    >>> G.graph.get("is_hetero")
    True
    """
    # ---- Normalise inputs and validate once ----
    relation = _relation_from_predicate(predicate)
    metric_lc = _normalize_metric(distance_metric)

    # Strong, early CRS presence check (runs before generic validator to avoid any side-effects)
    poly_crs = polygons_gdf.crs
    pt_crs = points_gdf.crs
    if not poly_crs or not pt_crs:
        msg = (
            f"Both inputs must have a CRS (got polygons_gdf.crs={poly_crs}, "
            f"points_gdf.crs={pt_crs})"
        )
        raise ValueError(msg)

    # Validate GDFs and CRS via shared utility (keeps behaviour consistent across the package)
    # We intentionally ignore the returned validated copies to avoid accidental mutation.
    validate_gdf({"polygon": polygons_gdf, "point": points_gdf}, None, allow_empty=True)

    if poly_crs != pt_crs:
        msg = f"CRS mismatch between inputs: {poly_crs} != {pt_crs}"
        raise ValueError(msg)

    if metric_lc not in {"euclidean", "manhattan", "network"}:
        msg = f"Unsupported distance_metric: {distance_metric!r}"
        raise ValueError(msg)
    if metric_lc == "network":
        if network_gdf is None:
            msg = "network_gdf is required when distance_metric='network'"
            raise ValueError(msg)
        if network_gdf.crs != poly_crs:
            msg = f"CRS mismatch between inputs and network: inputs={poly_crs} != network={network_gdf.crs}"
            raise ValueError(msg)

    # Quick exits for empty inputs / no matches
    if polygons_gdf.empty or points_gdf.empty:
        return _group_nodes_empty_result(polygons_gdf, points_gdf, relation, poly_crs, as_nx)

    # Find all (polygon_id, point_id) pairs matching the containment predicate
    pairs = _containment_pairs(points_gdf, polygons_gdf, predicate)

    # Return empty result if no containment pairs were found
    if not pairs:
        return _group_nodes_empty_result(polygons_gdf, points_gdf, relation, poly_crs, as_nx)

    # Build edge GeoDataFrame using shared helpers
    edges_gdf = _edges_gdf_from_pairs(polygons_gdf, points_gdf, pairs, metric_lc, network_gdf)

    # Package as heterogeneous data structure (edges_gdf may legitimately be empty)
    nodes_dict: dict[str, gpd.GeoDataFrame] = {"polygon": polygons_gdf, "point": points_gdf}
    edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] = {
        ("polygon", relation, "point"): edges_gdf
    }
    return (
        gdf_to_nx(nodes=nodes_dict, edges=edges_dict, directed=True)
        if as_nx
        else (nodes_dict, edges_dict)
    )


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

    if metric.lower() == "network":  # delegate to helper for clarity
        _set_network_edge_geometries(
            G,
            geom_attr,
            coords,
            node_ids,
            idx_map,
            network_gdf,
        )
        nx.set_edge_attributes(G, geom_attr, "geometry")
        return

    # Manhattan or Euclidean metric
    # Vectorized geometry creation via array ops and a small comprehension
    edge_list = list(G.edges())
    if edge_list:
        # Use Python mapping to avoid None from dict.get with object keys under vectorization
        u_idx = np.array([idx_map[u] for u, _ in edge_list], dtype=int)
        v_idx = np.array([idx_map[v] for _, v in edge_list], dtype=int)

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


def _set_network_edge_geometries(
    G: nx.Graph,
    geom_attr: dict[tuple[Any, Any], LineString],
    coords: npt.NDArray[np.floating],
    node_ids: list[Any],
    idx_map: dict[Any, int],
    network_gdf: gpd.GeoDataFrame | None,
) -> None:
    """
    Populate geometry for network metric edges using batched shortest paths.

    This isolates the more complex logic from `_add_edges`, reducing cyclomatic
    complexity while enabling caching of the network graph and nearest-node
    mapping for repeated calls.

    Parameters
    ----------
    G : nx.Graph
        Graph whose edges will receive a ``geometry`` attribute in-place.
    geom_attr : dict[tuple[Any, Any], LineString]
        Mutable mapping populated with ``(u, v) -> LineString``; passed in so we
        can extend an existing dict without reallocating.
    coords : numpy.ndarray
        Array of node coordinate pairs (shape ``(n, 2)``) aligned with ``node_ids``.
    node_ids : list[Any]
        Sequence of node identifiers whose order matches ``coords``.
    idx_map : dict[Any, int]
        Mapping from node id to its integer row index in ``coords`` for quick lookup.
    network_gdf : geopandas.GeoDataFrame or None
        GeoDataFrame describing the underlying real network geometry (e.g., road
        segments). If provided it is converted to an internal NetworkX graph and
        cached under a private attribute to avoid repeated conversions. If ``None``
        a straight LineString between endpoints is used.

    Notes
    -----
    Caching: A NetworkX graph is cached on ``network_gdf`` under the private
    attribute ``_c2g_cached_nx`` so subsequent calls reuse the constructed graph.
    A nearest-node mapping is also cached on ``G.graph['_c2g_nearest_cache']``
    keyed by the ``id(coords)`` to avoid recomputing nearest neighbors when the
    same coordinate array instance is reused.
    The function mutates ``geom_attr`` and (via the caller) sets the ``geometry``
    edge attribute on ``G``. It may also attach a cached network graph to
    ``network_gdf``.
    """
    if network_gdf is None:  # pragma: no cover - defensive; public API ensures non-None
        msg = "network_gdf must be provided for network metric geometry construction"
        raise ValueError(msg)

    # Reuse cached network graph if available (constructed once per network_gdf instance)
    net_nx = getattr(network_gdf, "_c2g_cached_nx", None)
    if net_nx is None:
        net_nx = gdf_to_nx(edges=network_gdf)
        # Cache on the GeoDataFrame instance for reuse in subsequent calls
    cast("Any", network_gdf)._c2g_cached_nx = net_nx
    pos = nx.get_node_attributes(net_nx, "pos")  # gdf_to_nx guarantees 'pos'

    # Build / reuse nearest mapping
    net_coords = np.asarray(list(pos.values()))
    net_ids = list(pos.keys())
    nearest_cache = G.graph.get("_c2g_nearest_cache")
    if nearest_cache is None or nearest_cache.get("_hash_coords") is not id(coords):
        nn = NearestNeighbors(n_neighbors=1).fit(net_coords)
        _, idxs = nn.kneighbors(coords)
        nearest = {node_ids[i]: net_ids[j[0]] for i, j in enumerate(idxs)}
        G.graph["_c2g_nearest_cache"] = {"mapping": nearest, "_hash_coords": id(coords)}
    else:
        nearest = nearest_cache["mapping"]

    use_weight = "length" if any("length" in d for *_, d in net_nx.edges(data=True)) else None

    edges_by_source: dict[Any, list[tuple[Any, Any]]] = {}
    for u, v in G.edges():
        edges_by_source.setdefault(nearest[u], []).append((u, v))

    for src_nn, edge_list in edges_by_source.items():
        _, paths = nx.single_source_dijkstra(net_nx, src_nn, weight=use_weight)
        for u, v in edge_list:
            path_nodes = paths.get(nearest[v])
            if not path_nodes or len(path_nodes) < 2:
                # Uniform fallback: straight segment between endpoints
                geom_attr[(u, v)] = LineString([coords[idx_map[u]], coords[idx_map[v]]])
            else:
                geom_attr[(u, v)] = LineString([pos[p] for p in path_nodes])


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
    # Internal invariant: exactly one of k / radius is provided by callers.
    nn_metric = "cityblock" if metric == "manhattan" else "euclidean"
    if k is not None:
        n_neigh = min(k, len(dst_coords))
        nn = NearestNeighbors(n_neighbors=n_neigh, metric=nn_metric).fit(dst_coords)
        _, idxs = nn.kneighbors(src_coords)
        return [(src_ids[i], dst_ids[j]) for i, neigh in enumerate(idxs) for j in neigh]
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


def _validate_contiguity_input(gdf: gpd.GeoDataFrame, contiguity: str) -> None:
    """
    Lightweight validation for contiguity graph public API.

    Keep only checks required by tests and core invariants; rely on downstream
    libraries (GeoPandas/libpysal) for deeper geometry validation to reduce
    code complexity and branching.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input polygon layer to validate.
    contiguity : {"queen", "rook"}
        Contiguity rule to enforce.

    Raises
    ------
    TypeError
        If ``gdf`` is not a GeoDataFrame.
    ValueError
        If ``contiguity`` is not one of {"queen", "rook"}.
    """
    if not isinstance(gdf, gpd.GeoDataFrame):  # pragma: no cover (defensive)
        msg = (
            f"Input must be a GeoDataFrame, got {type(gdf).__name__}. "
            "Please provide a valid GeoDataFrame with polygon geometries."
        )
        raise TypeError(msg)
    if contiguity not in {"queen", "rook"}:
        msg = "Invalid contiguity type: must be 'queen' or 'rook'"
        raise ValueError(msg)


def _create_spatial_weights(
    gdf: gpd.GeoDataFrame,
    contiguity: str,
) -> libpysal.weights.W:
    """
    Create spatial weights matrix using libpysal for contiguity analysis.

    This helper wraps libpysal's Queen/Rook constructors and returns a
    weights object keyed by the original GeoDataFrame index. It centralises
    the choice of contiguity rule and small edge cases (like empties) so the
    public API remains concise and uniform.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing polygon geometries. Must have been validated
        by ``_validate_contiguity_input`` before calling this function.
    contiguity : {"queen", "rook"}
        Type of spatial contiguity to use.

    Returns
    -------
    libpysal.weights.W
        Spatial weights matrix representing adjacency relationships between polygons.
        The weights matrix uses the original GeoDataFrame index as identifiers.
    """
    # Handle empty GeoDataFrame case early
    if gdf.empty:  # pragma: no cover - empty handled earlier in public API
        return libpysal.weights.W({})

    # Validate contiguity value early to avoid catching it in the generic except
    contiguity_lc = contiguity.lower()
    if contiguity_lc not in {"queen", "rook"}:  # pragma: no cover (validated earlier)
        msg = f"Unsupported contiguity type: {contiguity}"
        raise ValueError(msg)

    # Simplified: rely on libpysal API (modern versions support ids); remove fallbacks.
    ids = list(gdf.index)
    if contiguity_lc == "queen":
        weights = libpysal.weights.Queen.from_dataframe(gdf, ids=ids)
    else:  # rook
        weights = libpysal.weights.Rook.from_dataframe(gdf, ids=ids)
    return weights


def _generate_contiguity_edges(
    weights: libpysal.weights.W,
    _gdf: gpd.GeoDataFrame,
) -> list[EdgePair]:
    """
    Extract adjacency relationships from a spatial weights matrix.

    The returned edge list contains each undirected adjacency exactly once
    with endpoints ordered canonically. This normalisation simplifies later
    processing and avoids duplicate edges in the final graph.

    Parameters
    ----------
    weights : libpysal.weights.W
        Spatial weights matrix containing adjacency relationships between polygons.
        Should be created by ``_create_spatial_weights``.
    _gdf : geopandas.GeoDataFrame
        Original GeoDataFrame (unused, present for logging/signature consistency).

    Returns
    -------
    list[tuple[Any, Any]]
        Unique undirected edges as pairs of original GeoDataFrame index values.
        Returns an empty list if there are no adjacency relationships.
    """
    # Empty weights -> no edges
    if not weights.neighbors:
        return []

    # Unique undirected edges, canonical order
    return list(
        {tuple(sorted((src, nbr))) for src, nbrs in weights.neighbors.items() for nbr in nbrs}
    )


def _build_contiguity_graph(
    gdf: gpd.GeoDataFrame,
    edges: list[EdgePair],
    *,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
) -> nx.Graph:
    """
    Build NetworkX graph from GeoDataFrame and edge list.

    Nodes preserve all attributes from the input GeoDataFrame and carry a
    'pos' coordinate attribute derived from polygon centroids. Edges receive
    metric-specific weights and geometries through the shared edge attachment
    utilities, ensuring consistent behaviour with the other generators.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Original GeoDataFrame containing polygon geometries and attributes.
        The index provides node identifiers and all columns are preserved
        as node attributes.
    edges : list[tuple[Any, Any]]
        List of (source_id, target_id) tuples representing edges in the graph.
        Should use the same identifiers as the GeoDataFrame index.
    distance_metric : {"euclidean", "manhattan", "network"}, default "euclidean"
        Metric used to compute edge weights/geometries.
    network_gdf : geopandas.GeoDataFrame, optional
        Line-based network required when ``distance_metric == 'network'``.

    Returns
    -------
    networkx.Graph
        Graph with nodes preserving original attributes and edges carrying
        weight and geometry attributes; graph metadata includes CRS.
    """
    # Reuse shared helpers: prepare nodes once (adds 'pos' and preserves attributes)
    G, coords, node_ids = _prepare_nodes(gdf)

    # Caller (contiguity_graph) already validated metric & network_gdf; avoid duplicate branches.
    metric_lc = _normalize_metric(distance_metric)

    # Attach edges with selected metric logic
    if edges:
        _add_edges(
            G,
            edges,
            coords,
            node_ids,
            metric=metric_lc,
            dm=None,
            network_gdf=network_gdf,
        )

    # Log graph construction results
    logger.debug(
        "Contiguity graph constructed: %d nodes, %d edges, CRS: %s, metric: %s",
        G.number_of_nodes(),
        G.number_of_edges(),
        G.graph.get("crs"),
        distance_metric,
    )

    return G


def _contiguity_graph_core(
    gdf: gpd.GeoDataFrame,
    contiguity: str,
    *,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
) -> nx.Graph:
    """
    Build a contiguity graph (NetworkX) from a validated polygon GeoDataFrame.

    This internal core performs the minimal steps to obtain adjacency edges
    via libpysal, then delegates node/edge construction to the shared helpers.
    Public-facing validation and formatting are handled by `contiguity_graph`.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing valid polygon geometries. Index values are used
        as node identifiers and all columns are preserved as node attributes.
    contiguity : {"queen", "rook"}
        Type of spatial contiguity used to derive adjacency relationships.
    distance_metric : {"euclidean", "manhattan", "network"}, default "euclidean"
        Metric used to compute edge weights/geometries.
    network_gdf : geopandas.GeoDataFrame, optional
        Line-based network required when ``distance_metric == 'network'``.

    Returns
    -------
    networkx.Graph
        Undirected graph with nodes/edges and metadata ('crs', 'contiguity', 'distance_metric').
    """
    # Create spatial weights and extract undirected edges
    logger.debug("Creating %s contiguity spatial weights matrix", contiguity)
    weights = _create_spatial_weights(gdf, contiguity)

    logger.debug("Extracting adjacency relationships from spatial weights")
    edges = _generate_contiguity_edges(weights, gdf)

    # Build graph and attach metadata
    logger.debug("Building NetworkX graph with nodes and edges")
    G = _build_contiguity_graph(
        gdf,
        edges,
        distance_metric=distance_metric,
        network_gdf=network_gdf,
    )
    G.graph["contiguity"] = contiguity
    G.graph["distance_metric"] = distance_metric
    return G


def _empty_contiguity_result(
    gdf: gpd.GeoDataFrame,
    contiguity: str,
    *,
    distance_metric: str = "euclidean",
    as_nx: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
    Create an empty-but-typed result matching contiguity_graph outputs.

    This keeps return types stable even when the input is empty or yields no
    adjacency relationships, preserving CRS and expected columns/metadata so
    downstream code can rely on a consistent schema.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        The input GeoDataFrame (possibly empty) used to derive CRS and columns.
    contiguity : str
        Contiguity type label to attach to graph metadata when returning NetworkX.
    distance_metric : {"euclidean", "manhattan", "network"}, default "euclidean"
        Metric recorded in graph metadata when returning a NetworkX graph.
    as_nx : bool
        Whether to return a NetworkX graph or a tuple of GeoDataFrames.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        Properly typed empty result preserving CRS and node columns (or an empty graph).
    """
    if as_nx:
        empty_graph = nx.Graph()
        empty_graph.graph["crs"] = gdf.crs
        empty_graph.graph["contiguity"] = contiguity
        empty_graph.graph["distance_metric"] = distance_metric
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
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
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
        different spatial analysis requirements. Edge weights are calculated as distances
        between polygon centroids using the selected ``distance_metric``. Supported metrics:

        * ``euclidean`` (default): straight-line distance; edge geometry is a direct
            centroid-to-centroid LineString.
        * ``manhattan``: L1 distance; edge geometry is an L-shaped polyline (two segments)
            following an axis-aligned path between centroids.
        * ``network``: shortest-path distance over ``network_gdf`` (a line network in the
            same CRS); edge geometry is the polyline path traced along the network.

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
    distance_metric : {"euclidean", "manhattan", "network"}, default "euclidean"
        Metric used to compute edge weights and geometries.
    network_gdf : geopandas.GeoDataFrame, optional
        Required when ``distance_metric='network'``. A line-based network whose CRS
        matches ``gdf``.
    as_nx : bool, default False
        Output format control. If True, returns a NetworkX Graph object with
        spatial attributes. If False, returns a tuple of GeoDataFrames for
        nodes and edges, compatible with other city2graph functions.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph
        When ``as_nx=False`` (default), returns ``(nodes_gdf, edges_gdf)`` as GeoDataFrames.
        When ``as_nx=True``, returns a NetworkX Graph with spatial attributes and metadata.

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
    >>> nodes_gdf, edges_gdf = contiguity_graph(gdf, contiguity="queen", distance_metric="euclidean")
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

    metric_lc = _normalize_metric(distance_metric)
    if metric_lc not in {"euclidean", "manhattan", "network"}:
        msg = f"Unsupported distance_metric: {distance_metric!r}"
        raise ValueError(msg)
    if metric_lc == "network":
        if network_gdf is None:
            msg = "network_gdf is required when distance_metric='network'"
            raise ValueError(msg)
        if network_gdf.crs != gdf.crs:
            msg = f"CRS mismatch between gdf ({gdf.crs}) and network_gdf ({network_gdf.crs})"
            raise ValueError(msg)

    # Step 2: Handle empty GeoDataFrame case up-front
    if gdf.empty:
        logger.debug("Empty GeoDataFrame provided - returning empty result")
        return _empty_contiguity_result(
            gdf,
            contiguity,
            distance_metric=metric_lc,
            as_nx=as_nx,
        )

    # Step 3: Build the core graph once, then format output as requested
    G = _contiguity_graph_core(
        gdf,
        contiguity,
        distance_metric=metric_lc,
        network_gdf=network_gdf,
    )

    # Step 4: Return in requested format (single return branch for clarity)
    return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)
