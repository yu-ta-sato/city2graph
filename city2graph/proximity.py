"""Proximity based graph generators.

This module offers four classical proximity models that all share the same
internal workflow:

1. Validate the input GeoDataFrame(s)
2. Build a base (NetworkX) graph and cache node coordinates
3. Compute a distance matrix (Euclidean, Manhattan, or network shortest path)
4. Derive an edge list according to the chosen proximity model
5. Attach edge weights and geometries
6. Return either the NetworkX object or GeoDataFrames via `nx_to_gdf`

The public API remains unchanged:
    • knn_graph
    • delaunay_graph
    • fixed_radius_graph
    • waxman_graph
    • bridge_nodes
No classes are introduced everything is implemented with plain functions.
"""

from __future__ import annotations

from itertools import combinations
from itertools import permutations
from typing import TYPE_CHECKING
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import distance as sdist
from scipy.spatial.qhull import QhullError
from shapely.geometry import LineString
from sklearn.neighbors import NearestNeighbors

from .utils import gdf_to_nx
from .utils import nx_to_gdf
from .utils import validate_gdf

if TYPE_CHECKING:
    from collections.abc import Iterable

    import geopandas as gpd

__all__ = [
    "bridge_nodes",
    "delaunay_graph",
    "euclidean_minimum_spanning_tree",
    "fixed_radius_graph",
    "gabriel_graph",
    "knn_graph",
    "relative_neighborhood_graph",
    "waxman_graph",
]

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
    r"""Generate a k-nearest neighbour graph from a GeoDataFrame of points.

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
    delaunay_graph : Generate a Delaunay triangulation graph
    fixed_radius_graph : Generate a fixed-radius graph
    waxman_graph : Generate a probabilistic Waxman graph

    Notes
    -----
    - Node IDs are preserved from the input GeoDataFrame's index
    - Edge weights represent the distance between connected nodes
    - Edge geometries are LineStrings connecting node centroids
    - For Manhattan distance, edge geometries follow L-shaped paths
    - The graph is undirected unless `target_gdf` is specified

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
    if target_gdf is not None:  # directed variant
        return _directed_graph(
            src_gdf=gdf,
            dst_gdf=target_gdf,
            distance_metric=distance_metric,
            method="knn",
            param=k,
            as_nx=as_nx,
            network_gdf=network_gdf,
        )

    G, coords, node_ids = _prepare_nodes(gdf)
    if len(coords) <= 1 or k <= 0:
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    dm = None
    if distance_metric == "network":
        dm = _distance_matrix(coords, "network", network_gdf, gdf.crs)
        order = np.argsort(dm, axis=1)[:, 1 : k + 1]
        edges = [(node_ids[i], node_ids[j]) for i in range(len(node_ids)) for j in order[i] if dm[i, j] < np.inf]
    else:
        nn_metric = "cityblock" if distance_metric == "manhattan" else "euclidean"
        n_neigh = min(k + 1, len(coords))
        nn = NearestNeighbors(n_neighbors=n_neigh, metric=nn_metric).fit(coords)
        _, idxs = nn.kneighbors(coords)
        edges = [
            (node_ids[i], node_ids[j])
            for i, neigh in enumerate(idxs)
            for j in neigh[1:]
        ]

    _add_edges(G, edges, coords, node_ids, metric=distance_metric, dm=dm, network_gdf=network_gdf)
    return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)


def delaunay_graph(
    gdf: gpd.GeoDataFrame,
    distance_metric: str = "euclidean",
    network_gdf: gpd.GeoDataFrame | None = None,
    *,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """Generate a Delaunay triangulation graph from a GeoDataFrame of points.

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
    geopandas.GeoDataFrame or networkx.Graph
        If `as_nx` is False, returns a tuple of GeoDataFrames:
        - nodes_gdf: GeoDataFrame of nodes (same as input `gdf` with added attributes).
        - edges_gdf: GeoDataFrame of edges with 'weight' and 'geometry' attributes.
        If `as_nx` is True, returns a NetworkX graph object.

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
    G, coords, node_ids = _prepare_nodes(gdf)
    if len(coords) < 3:
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    try:
        tri = Delaunay(coords)
        edges = {(node_ids[i], node_ids[j]) for simplex in tri.simplices for i, j in combinations(simplex, 2)}
    except QhullError:
        edges = set()

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
    r"""Generate a Gabriel graph from a GeoDataFrame of points.

    In a Gabriel graph two nodes *u* and *v* are connected iff the closed
    disc that has $uv$ as its diameter contains no other node of the set.

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
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | networkx.Graph
        Same convention as the other graph-generator functions.

    Notes
    -----
    • The Gabriel graph is a sub-graph of the Delaunay triangulation; therefore
      the implementation first builds the Delaunay edges then filters them
      according to the disc-emptiness predicate, achieving an overall

      $$\mathcal O(n \log n + m k)$$ complexity ( *m* = Delaunay edges,
      *k* = average neighbours tested per edge).
    • When the input layer has exactly two points, the unique edge is returned.
    • If the layer has fewer than two points, an empty graph is produced.

    Examples
    --------
    >>> nodes, edges = gabriel_graph(points_gdf)
    >>> G = gabriel_graph(points_gdf, as_nx=True)
    """
    # ---- node preparation ---------------------------------------------------
    G, coords, node_ids = _prepare_nodes(gdf)
    n_points = len(coords)
    if n_points < 2:
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # ---- candidate edges: Delaunay ------------------------------------------
    if n_points == 2:
        delaunay_edges = {(0, 1)}
    else:
        tri = Delaunay(coords)
        delaunay_edges = {
            tuple(sorted((i, j)))
            for simplex in tri.simplices
            for i, j in combinations(simplex, 2)
        }

    # ---- Gabriel filtering --------------------------------------------------
    # Square distances for numerical stability
    kept_edges: set[tuple[Any, Any]] = set()
    tol = 1e-12
    for i, j in delaunay_edges:
        mid = 0.5 * (coords[i] + coords[j])
        rad2 = np.sum((coords[i] - coords[j]) ** 2) * 0.25  # (|pi-pj|/2)^2
        if rad2 == 0.0:   # coincident points – ignore
            continue
        # squared distance of *all* points to the midpoint
        d2 = np.sum((coords - mid) ** 2, axis=1)
        mask = d2 <= rad2 + tol
        # exactly the two endpoints inside the disc?
        if np.count_nonzero(mask) == 2:
            kept_edges.add((node_ids[i], node_ids[j]))

    # ---- weights + geometries ----------------------------------------------
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
    r"""Generate a Relative-Neighbourhood Graph (RNG) from a GeoDataFrame.

    In an RNG two nodes *u* and *v* are connected iff there is **no third node
    *w*** such that both $$d(u,w) < d(u,v)$$ **and** $$d(v,w) < d(u,v)$$.
    Equivalently, the intersection of the two open discs having radius

    $$d(u,v)$$ and centres *u* and *v* (the *lune*) is empty.

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
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | networkx.Graph
        Same convention as the other graph-generator functions.

    Notes
    -----
    •  The RNG is a sub-graph of the Delaunay triangulation; therefore the
       implementation first collects Delaunay edges ( $$\mathcal O(n\log n)$$ )
       and then filters them according to the lune-emptiness predicate.
    •  When the input layer has exactly two points the unique edge is returned.
    •  If the layer has fewer than two points, an empty graph is produced.

    Examples
    --------
    >>> nodes, edges = relative_neighborhood_graph(points_gdf)
    >>> G = relative_neighborhood_graph(points_gdf, as_nx=True)
    """
    # ---- node preparation -------------------------------------------------
    G, coords, node_ids = _prepare_nodes(gdf)
    n_points = len(coords)
    if n_points < 2:
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # ---- candidate edges: Delaunay ---------------------------------------
    if n_points == 2:
        cand_edges = {(0, 1)}
    else:
        tri = Delaunay(coords)
        cand_edges = {
            tuple(sorted((i, j)))
            for simplex in tri.simplices
            for i, j in combinations(simplex, 2)
        }

    # ---- RNG filtering ----------------------------------------------------
    kept_edges: set[tuple[Any, Any]] = set()
    # work with squared distances to avoid sqrt
    for i, j in cand_edges:
        dij2 = np.dot(coords[i] - coords[j], coords[i] - coords[j])
        if dij2 == 0.0:  # coincident points → ignore
            continue
        # vectorised test of the lune-emptiness predicate
        di2 = np.sum((coords - coords[i]) ** 2, axis=1) < dij2
        dj2 = np.sum((coords - coords[j]) ** 2, axis=1) < dij2
        # any third point closer to *both* i and j?
        closer_both = np.where(di2 & dj2)[0]
        if len(closer_both) == 0:
            kept_edges.add((node_ids[i], node_ids[j]))

    # ---- weights + geometries --------------------------------------------
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
    r"""Generate a (generalised) Euclidean Minimum Spanning Tree from a GeoDataFrame of points.

    The classical Euclidean Minimum Spanning Tree (EMST) is the minimum-
    total-length tree that connects a set of points when edge weights are the
    straight-line ( L₂ ) distances.  For consistency with the other generators
    this implementation also supports *manhattan* and *network* metrics - it
    simply computes the minimum-weight spanning tree under the chosen metric.
    When the metric is *euclidean* the edge search is restricted to the
    Delaunay triangulation (  EMST ⊆ Delaunay ), guaranteeing an

    $$\mathcal O(n \log n)$$ overall complexity.  With other metrics, or
    degenerate cases where the triangulation cannot be built, the algorithm
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
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | networkx.Graph
        Same convention as the other graph-generator functions.

    Examples
    --------
    >>> nodes, edges = euclidean_minimum_spanning_tree(points_gdf)
    >>> G = euclidean_minimum_spanning_tree(points_gdf, as_nx=True)

    Notes
    -----
    •  The resulting graph always contains *n - 1* edges (or 0 / 1 when the
       input has < 2 points).
    •  For planar Euclidean inputs the computation is

       $$\mathcal O(n \log n)$$ thanks to the Delaunay pruning.
    •  All the usual spatial attributes (*weight*, *geometry*, CRS checks,
       etc.) are attached through the shared private helpers.
    """
    # ---- node preparation -------------------------------------------------
    G, coords, node_ids = _prepare_nodes(gdf)
    n_points = len(coords)
    if n_points < 2:
        # MST is empty (0 nodes) or a single isolated node
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    # ---- candidate edge set ----------------------------------------------
    # Fast O(n) candidate set via Delaunay when it is applicable
    use_complete_graph = False
    cand_edges: set[tuple[Any, Any]]

    if distance_metric.lower() == "euclidean" and n_points >= 3:
        try:
            tri = Delaunay(coords)
            cand_edges = {
                tuple(sorted((i, j)))
                for simplex in tri.simplices
                for i, j in combinations(simplex, 2)
            }
        except QhullError:
            # Collinear or otherwise degenerate set – fall back
            use_complete_graph = True
    else:
        use_complete_graph = True

    if use_complete_graph:
        cand_edges = {
            (i, j) for i in range(n_points) for j in range(i + 1, n_points)
        }

    # Convert vertex indices → actual node ids
    cand_edges = {(node_ids[i], node_ids[j]) for i, j in cand_edges}

    # ---- attach weights / geometries -------------------------------------
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

    # ---- compute the minimum-spanning tree -------------------------------
    mst_G = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")

    # ---- output formatting ------------------------------------------------
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
    r"""Generate a fixed-radius graph from a GeoDataFrame of points.

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
    knn_graph : Generate a k-nearest neighbour graph
    delaunay_graph : Generate a Delaunay triangulation graph
    waxman_graph : Generate a probabilistic Waxman graph

    Notes
    -----
    - Node IDs are preserved from the input GeoDataFrame's index
    - Edge weights represent the actual distance between connected nodes
    - Edge geometries are LineStrings connecting node centroids
    - The graph stores the radius parameter in `G.graph["radius"]`
    - For Manhattan distance, edges follow L-shaped geometric paths

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
    if target_gdf is not None:  # directed
        return _directed_graph(
            src_gdf=gdf,
            dst_gdf=target_gdf,
            distance_metric=distance_metric,
            method="radius",
            param=radius,
            as_nx=as_nx,
            network_gdf=network_gdf,
        )

    G, coords, node_ids = _prepare_nodes(gdf)
    if len(coords) < 2:
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

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
        edges = [
            (node_ids[i], node_ids[j])
            for i, neigh in enumerate(idxs)
            for j in neigh
            if i < j
        ]

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
    r"""Generate a probabilistic Waxman graph from a GeoDataFrame of points.

    This function constructs a random graph where the probability of an edge existing
    between two nodes decreases exponentially with their distance. The model is based
    on the Waxman random graph model, commonly used to simulate realistic network
    topologies in telecommunications, transportation, and social networks where
    connection probability diminishes with distance.

    The connection probability follows the formula:

    $P(u,v) = \beta \times \exp \left(-\frac{\text{dist}(u,v)}{r_0}\right)$

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
    knn_graph : Generate a k-nearest neighbour graph
    delaunay_graph : Generate a Delaunay triangulation graph
    fixed_radius_graph : Generate a fixed-radius graph

    Notes
    -----
    - Node IDs are preserved from the input GeoDataFrame's index
    - Edge weights represent the actual distance between connected nodes
    - Edge geometries are LineStrings connecting node centroids
    - The graph stores parameters in `G.graph["beta"]` and `G.graph["r0"]`
    - Results are stochastic; use `seed` parameter for reproducible outputs
    - The graph is undirected with symmetric edge probabilities

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
    rng = np.random.default_rng(seed)
    G, coords, node_ids = _prepare_nodes(gdf)
    if len(coords) < 2:
        return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)

    dm = _distance_matrix(coords, distance_metric.lower(), network_gdf, gdf.crs)
    with np.errstate(divide="ignore"):
        probs = beta * np.exp(-dm / r0)
    probs[dm == np.inf] = 0  # unreachable in network metric

    rand = rng.random(dm.shape)
    mask = (rand <= probs) & np.triu(np.ones_like(dm, dtype=bool), 1)
    edge_idx = np.column_stack(np.where(mask))
    edges = [(node_ids[i], node_ids[j]) for i, j in edge_idx]

    _add_edges(G, edges, coords, node_ids, metric=distance_metric, dm=dm, network_gdf=network_gdf)
    G.graph.update({"beta": beta, "r0": r0})
    return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)


def bridge_nodes(
    nodes_dict: dict[str, gpd.GeoDataFrame],
    proximity_method: str = "knn",
    *,
    multigraph: bool = False,
    as_nx: bool = False,
    **kwargs: Any,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]] | nx.Graph:
    r"""Build directed proximity edges between every ordered pair of node layers.

    This function creates a multi-layer spatial network by generating directed proximity
    edges from nodes in one GeoDataFrame layer to nodes in another. It systematically
    processes all ordered pairs of layers and applies either k-nearest neighbors (KNN)
    or fixed-radius method to establish inter-layer connections. This is
    particularly useful for modeling complex urban systems, ecological networks, or
    multi-modal transportation systems where different types of entities interact.

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
        If `as_nx` is False, returns a tuple:
        - nodes_dict: The original input `nodes_dict` (unchanged)
        - edges_dict: Dictionary where keys are edge type tuples 
          `(source_layer_name, "is_nearby", target_layer_name)` and values are
          GeoDataFrames of the generated directed edges
        If `as_nx` is True, returns a NetworkX graph object containing all nodes
        from all layers and the generated directed inter-layer edges.

    Raises
    ------
    ValueError
        If `nodes_dict` contains fewer than two layers.
        If `proximity_method` is not "knn" or "fixed_radius".
        If `proximity_method` is "fixed_radius" but `radius` is not provided in `kwargs`.

    See Also
    --------
    knn_graph : Generate a k-nearest neighbour graph
    fixed_radius_graph : Generate a fixed-radius graph

    Notes
    -----
    - All generated edges are directed from source layer to target layer
    - The relation type for all generated edges is fixed as "is_nearby"
    - Edge weights and geometries are calculated based on the chosen distance_metric
    - Each ordered pair of layers generates a separate edge GeoDataFrame
    - Self-connections (layer to itself) are not created

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
    if len(nodes_dict) < 2:
        msg = "`nodes_dict` needs at least two layers"
        raise ValueError(msg)
    if proximity_method.lower() not in {"knn", "fixed_radius"}:
        msg = "proximity_method must be 'knn' or 'fixed_radius'"
        raise ValueError(msg)

    edge_dict = {}
    for src_type, dst_type in permutations(nodes_dict.keys(), 2):
        src_gdf = nodes_dict[src_type]
        dst_gdf = nodes_dict[dst_type]

        if proximity_method.lower() == "knn":
            k = int(kwargs.get("k", 1))
            _, edges_gdf = knn_graph(
                src_gdf,
                k=k,
                target_gdf=dst_gdf,
                **{k_: v for k_, v in kwargs.items() if k_ not in {"k", "radius"}},
            )
        else:  # fixed_radius
            radius = float(kwargs["radius"])
            _, edges_gdf = fixed_radius_graph(
                src_gdf,
                radius=radius,
                target_gdf=dst_gdf,
                **{k_: v for k_, v in kwargs.items() if k_ != "radius"},
            )

        edge_dict[(src_type, "is_nearby", dst_type)] = edges_gdf

    if as_nx:
        return gdf_to_nx(nodes=nodes_dict, edges=edge_dict, multigraph=multigraph)
    return nodes_dict, edge_dict

# ============================================================================
# NODE PREPARATION
# ============================================================================

def _prepare_nodes(
    gdf: gpd.GeoDataFrame,
    *,
    directed: bool = False,
) -> tuple[nx.Graph, np.ndarray, list]:
    """Return an empty graph with populated nodes plus coord cache."""
    validate_gdf(nodes_gdf=gdf)

    centroids = gdf.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])
    node_ids = list(gdf.index)

    G = nx.DiGraph() if directed else nx.Graph()

    for node_id, attrs, (x, y) in zip(node_ids, gdf.to_dict("records"), coords, strict=False):
        G.add_node(node_id, **attrs, pos=(x, y))

    return G, coords, node_ids


# ============================================================================
# DISTANCE MATRIX
# ============================================================================

def _euclidean_dm(coords: np.ndarray) -> np.ndarray:
    return sdist.squareform(sdist.pdist(coords))


def _manhattan_dm(coords: np.ndarray) -> np.ndarray:
    return sdist.squareform(sdist.pdist(coords, metric="cityblock"))


def _network_dm(
    coords: np.ndarray,
    network_gdf: gpd.GeoDataFrame,
    gdf_crs: gpd.crs.CRS | None = None,
) -> np.ndarray:
    if network_gdf.crs != gdf_crs:
        msg = f"CRS mismatch: {gdf_crs} != {network_gdf.crs}"
        raise ValueError(msg)

    # Convert edge GDF → NetworkX
    net_nx = gdf_to_nx(edges=network_gdf)

    # Get node positions
    pos = nx.get_node_attributes(net_nx, "pos")
    if not pos:
        msg = "The supplied network lacks node 'pos' attributes"
        raise ValueError(msg)

    net_coords = np.asarray(list(pos.values()))
    net_ids = list(pos.keys())

    # Map sample points to nearest network nodes
    nn = NearestNeighbors(n_neighbors=1).fit(net_coords)
    _, idx = nn.kneighbors(coords)
    nearest = [net_ids[i[0]] for i in idx]

    # Pre-allocate matrix
    n = len(coords)
    dm = np.full((n, n), np.inf)
    np.fill_diagonal(dm, 0)

    use_weight = "length" if any("length" in d for _, _, d in net_nx.edges(data=True)) else None

    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(net_nx, nearest[i], weight=use_weight)
        for j in range(i + 1, n):
            d = lengths.get(nearest[j], np.inf)
            dm[i, j] = dm[j, i] = d

    return dm


def _distance_matrix(
    coords: np.ndarray,
    metric: str,
    network_gdf: gpd.GeoDataFrame | None,
    gdf_crs: gpd.crs.CRS | None = None,
) -> np.ndarray:
    """Return a distance matrix for the given coordinates and metric."""
    metric = metric.lower()
    if metric == "euclidean":
        return _euclidean_dm(coords)
    if metric == "manhattan":
        return _manhattan_dm(coords)
    if metric == "network":
        if network_gdf is None:
            msg = "`network_gdf` must be supplied for network metric"
            raise ValueError(msg)
        return _network_dm(coords, network_gdf, gdf_crs)
    msg = "distance_metric must be 'euclidean', 'manhattan', or 'network'"
    raise ValueError(msg)

# ============================================================================
# EDGE ADDITION
# ============================================================================

def _add_edges(
    G: nx.Graph,
    edges: Iterable[tuple[Any, Any]],
    coords: np.ndarray,
    node_ids: list,
    *,
    metric: str,
    dm: np.ndarray | None = None,
    network_gdf: gpd.GeoDataFrame | None = None,
) -> None:
    """
    Insert *edges* into *G* and attach both weights and geometries.

    When `metric == "network"` the geometry is traced along the input
    `network_gdf`, so the resulting LineString corresponds to the real
    shortest-path on that network rather than a straight segment.
    """
    if not edges:
        return

    # ---- 1. add the edges --------------------------------------------------
    G.add_edges_from(edges)

    # ---- 2. weights --------------------------------------------------------
    idx_map = {n: i for i, n in enumerate(node_ids)}

    if dm is not None:
        weights = {(u, v): dm[idx_map[u], idx_map[v]] for u, v in G.edges()}
    elif metric == "manhattan": # Manhattan
        weights = {
            (u, v): abs(coords[idx_map[u]][0] - coords[idx_map[v]][0])
            + abs(coords[idx_map[u]][1] - coords[idx_map[v]][1])
            for u, v in G.edges()
        }
    else: # Euclidean
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

    # ---- 3. geometries -----------------------------------------------------
    geom_attr: dict[tuple[Any, Any], LineString] = {}

    # 3-a: network metric  ..................................................
    if metric.lower() == "network":
        if network_gdf is None:
            msg = (
                "`network_gdf` must be supplied when distance_metric='network' "
                "so that path geometries can be traced."
            )
            raise ValueError(
                msg,
            )

        # Build a NetworkX representation of the network
        net_nx = gdf_to_nx(edges=network_gdf)

        # All network nodes must expose coords in attribute 'pos'
        pos = nx.get_node_attributes(net_nx, "pos")
        if not pos:
            msg = "The supplied network lacks node 'pos' attributes"
            raise ValueError(msg)

        net_coords = np.asarray(list(pos.values()))
        net_ids = list(pos.keys())

        # Map each sample point to its nearest network node
        nn = NearestNeighbors(n_neighbors=1).fit(net_coords)
        _, idxs = nn.kneighbors(coords)
        nearest = {node_ids[i]: net_ids[j[0]] for i, j in enumerate(idxs)}

        # Choose weight key if present on the network
        use_weight = (
            "length"
            if any("length" in d for *_, d in net_nx.edges(data=True))
            else None
        )

        for u, v in G.edges():
            # shortest path in network
            path_nodes = nx.shortest_path(
                net_nx,
                source=nearest[u],
                target=nearest[v],
                weight=use_weight,
            )
            path_coords = [pos[p] for p in path_nodes]

            # remove consecutive duplicates and make sure at least 2 points
            path_coords = [path_coords[i]
                           for i in range(len(path_coords))
                           if i == 0 or path_coords[i] != path_coords[i - 1]]

            if len(path_coords) < 2:
                p_start = coords[idx_map[u]]
                p_end   = coords[idx_map[v]]
                path_coords = [p_start, p_end]

            geom_attr[(u, v)] = LineString(path_coords)

    # 3-b: manhattan / euclidean .............................................
    else:
        for u, v in G.edges():
            p1 = coords[idx_map[u]]
            p2 = coords[idx_map[v]]
            if metric.lower() == "manhattan":
                geom = LineString(
                    [(p1[0], p1[1]), (p2[0], p1[1]), (p2[0], p2[1])]
                )
            else:  # euclidean
                geom = LineString([p1, p2])
            geom_attr[(u, v)] = geom

    nx.set_edge_attributes(G, geom_attr, "geometry")


def _directed_edges(
    src_coords: np.ndarray,
    dst_coords: np.ndarray,
    src_ids: list,
    dst_ids: list,
    *,
    metric: str,
    k: int | None = None,
    radius: float | None = None,
) -> list[tuple[Any, Any]]:
    """Return edge tuples from *src* to *dst* according to KNN or radius."""
    if (k is None) == (radius is None):
        msg = "Specify exactly one of k or radius for directed graph"
        raise ValueError(msg)

    nn_metric = "cityblock" if metric == "manhattan" else "euclidean"

    if k is not None:
        n_neigh = min(k, len(dst_coords))
        nn = NearestNeighbors(n_neighbors=n_neigh, metric=nn_metric).fit(dst_coords)
        _, idxs = nn.kneighbors(src_coords)
        return [
            (src_ids[i], dst_ids[j])
            for i, neigh in enumerate(idxs)
            for j in neigh
        ]

    # radius case
    nn = NearestNeighbors(radius=radius, metric=nn_metric).fit(dst_coords)
    idxs = nn.radius_neighbors(src_coords, return_distance=False)
    return [
        (src_ids[i], dst_ids[j])
        for i, neigh in enumerate(idxs)
        for j in neigh
    ]

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
    """Build source → target directed proximity edges."""
    if src_gdf.crs != dst_gdf.crs:
        msg = "CRS mismatch between source and target GeoDataFrames"
        raise ValueError(msg)

    src_G, src_coords, src_ids = _prepare_nodes(src_gdf, directed=True)
    dst_G, dst_coords, dst_ids = _prepare_nodes(dst_gdf, directed=True)
    # Merge nodes into one directed graph
    G = nx.compose(src_G, dst_G)

    edges = _directed_edges(
        src_coords,
        dst_coords,
        src_ids,
        dst_ids,
        metric=distance_metric,
        k=param if method == "knn" else None,
        radius=param if method == "radius" else None,
    )

    # src_G was created first – use its coords for weight calc
    combined_coords = np.vstack([src_coords, dst_coords])
    combined_ids = src_ids + dst_ids
    _add_edges(G, edges, combined_coords, combined_ids, metric=distance_metric, network_gdf=network_gdf)

    return G if as_nx else nx_to_gdf(G, nodes=True, edges=True)
