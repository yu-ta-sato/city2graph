"""Module for creating morphological graphs from urban data."""

import logging
import math
import warnings

import geopandas as gpd
import networkx as nx
import libpysal
import pandas as pd
from shapely.geometry import LineString, Point

from .utils import create_tessellation
from .utils import dual_graph
from .utils import filter_graph_by_distance
from .utils import gdf_to_nx
from .utils import _create_nodes_gdf
from .utils import _extract_node_positions
from .utils import _get_nearest_node

# Define the public API for this module
__all__ = [
    "morphological_graph",
    "private_to_private_graph",
    "private_to_public_graph",
    "public_to_public_graph",
]

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN MORPHOLOGICAL GRAPH FUNCTION
# ============================================================================


def morphological_graph(
    buildings_gdf: gpd.GeoDataFrame,
    segments_gdf: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    distance: float | None = None,
    private_id_col: str | None = None,
    public_id_col: str | None = None,
    tessellation_distance: float = math.inf,
    public_geom_col: str | None = "barrier_geometry",
    contiguity: str = "queen",
    keep_buildings: bool = False,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]:
    """
    Create a morphological graph from buildings and street segments.

    This function creates a comprehensive morphological graph that captures relationships
    between private spaces (building tessellations) and public spaces (street segments).
    The graph includes three types of relationships: private-to-private adjacency,
    public-to-public connectivity, and private-to-public interfaces.

    Parameters
    ----------
    buildings_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing building polygons. Should contain Polygon or MultiPolygon geometries.
    segments_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing street segments. Should contain LineString geometries.
    center_point : Union[gpd.GeoSeries, gpd.GeoDataFrame], optional
        Center point(s) for spatial filtering. If provided with distance parameter,
        only segments within the specified distance will be included.
    distance : float, optional
        Maximum distance from ``center_point`` for spatial filtering. When
        specified, street segments beyond this shortest-path distance are
        removed and tessellation cells are kept only if their own distance via
        these segments does not exceed this value.
    private_id_col : str, optional
        Column name to use for private space identifiers. If None, uses "tess_id".
        If the column doesn't exist, it will be created using row indices.
    public_id_col : str, optional
        Column name to use for public space identifiers. If None, uses "id".
        If the column doesn't exist, it will be created using row indices.
    tessellation_distance : float, default=math.inf
        Maximum allowed distance between tessellation cells and street segments.
        Distances are evaluated within each ``enclosure_index`` group
        if that column exists, otherwise globally. The distance is measured
        between tessellation centroids and the nearest street segment.
        The default of ``math.inf`` retains all cells. When ``center_point`` and
        ``distance`` are provided, tessellation cells are additionally filtered
        by their shortest-path distance from ``center_point`` via public
        segments so that only cells within ``distance`` are kept.
    public_geom_col : str, optional
        Column name containing alternative geometry for public spaces. If specified and exists,
        this geometry will be used instead of the main geometry column for tessellation barriers.
        Default is "barrier_geometry".
    contiguity : str, default="queen"
        Type of spatial contiguity for private-to-private connections.
        Must be either "queen" or "rook".
    keep_buildings : bool, default=False
        If True, preserves building information in the tessellation output.

    Returns
    -------
    tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
        A tuple containing:
        - nodes: Dictionary with keys "private" and "public" containing node GeoDataFrames
        - edges: Dictionary with relationship type keys containing edge GeoDataFrames

    Raises
    ------
    TypeError
        If buildings_gdf or segments_gdf are not GeoDataFrames.
    ValueError
        If contiguity parameter is not "queen" or "rook".

    Notes
    -----
    The function creates tessellations from buildings and optionally uses street segments
    as barriers. It then establishes three types of relationships:

    1. Private-to-private: Spatial adjacency between tessellation polygons
    2. Public-to-public: Topological connectivity between street segments
    3. Private-to-public: Spatial interfaces between tessellations and streets

    The output follows a heterogeneous graph structure suitable for network analysis
    of urban morphology.
    """
    # Validate input GeoDataFrames
    _validate_input_gdfs(buildings_gdf, segments_gdf)

    # Ensure CRS consistency
    segments_gdf = _ensure_crs_consistency(buildings_gdf, segments_gdf)

    # Set default column names
    private_id_col = private_id_col or "tess_id"
    public_id_col = public_id_col or "id"

    # Prepare barriers and create tessellation
    barriers = _prepare_barriers(segments_gdf, public_geom_col)
    tessellation = create_tessellation(
        buildings_gdf,
        primary_barriers=None if barriers.empty else barriers,
    )
    tessellation, private_id_col = _ensure_id_column(tessellation, private_id_col, "tess_id")

    # Apply spatial filtering if requested
    if center_point is not None and distance is not None and not segments_gdf.empty:
        segs = filter_graph_by_distance(segments_gdf, center_point, distance)
    else:
        segs = segments_gdf
    segs, public_id_col = _ensure_id_column(segs, public_id_col, "id")

    # Filter tessellation to only include areas adjacent to segments
    tessellation = _filter_adjacent_tessellation(
        tessellation,
        segs,
        max_distance=tessellation_distance,
    )

    if center_point is not None and distance is not None:
        tessellation = _filter_tessellation_by_network_distance(
            tessellation,
            segs,
            center_point,
            distance,
        )

    # Optionally preserve building information
    if keep_buildings:
        tessellation = _add_building_info(tessellation, buildings_gdf)

    # Create all three graph relationships
    priv_priv = private_to_private_graph(
        tessellation,
        private_id_col=private_id_col,
        group_col="enclosure_index",
        contiguity=contiguity,
    )
    pub_pub = public_to_public_graph(segs, public_id_col=public_id_col)
    priv_pub = private_to_public_graph(
        tessellation,
        segs,
        private_id_col=private_id_col,
        public_id_col=public_id_col,
        public_geom_col=public_geom_col,
    )

    # Log warning if no private-public connections found
    if priv_pub.empty:
        logger.warning("No private to public connections found")

    # Organize output as heterogeneous graph structure
    nodes = {
        "private": _set_index_if_exists(tessellation, private_id_col),
        "public": _set_index_if_exists(segs, public_id_col),
    }
    edges = {
        ("private", "touched_to", "private"): _set_edge_index(
            priv_priv, "from_private_id", "to_private_id",
        ),
        ("public", "connected_to", "public"): _set_edge_index(
            pub_pub, "from_public_id", "to_public_id",
        ),
        ("private", "faced_to", "public"): _set_edge_index(
            priv_pub, "private_id", "public_id",
        ),
    }
    return nodes, edges


# ============================================================================
# PRIVATE TO PRIVATE GRAPH FUNCTIONS
# ============================================================================


def private_to_private_graph(
    private_gdf: gpd.GeoDataFrame,
    private_id_col: str | None = None,
    group_col: str | None = None,
    contiguity: str = "queen",
) -> gpd.GeoDataFrame:
    """
    Create edges between contiguous private polygons based on spatial adjacency.

    This function identifies spatial adjacency relationships between private polygons
    (e.g., tessellation cells) using either Queen or Rook contiguity criteria.
    Optionally groups connections within specified groups (e.g., enclosures).

    Parameters
    ----------
    private_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing private space polygons. Should contain Polygon geometries.
    private_id_col : str, optional
        Column name to use for private space identifiers. If None, uses "tess_id".
        If the column doesn't exist, it will be created using row indices.
    group_col : str, optional
        Column name for grouping connections. Only polygons within the same group
        will be connected. If None, all polygons are considered as one group.
    contiguity : str, default="queen"
        Type of spatial contiguity to use. Must be either "queen" or "rook".
        Queen contiguity includes vertex neighbors, Rook includes only edge neighbors.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing edge geometries between adjacent polygons.
        Columns include from_private_id, to_private_id, group column, and geometry.

    Raises
    ------
    TypeError
        If private_gdf is not a GeoDataFrame.
    ValueError
        If contiguity is not "queen" or "rook", or if group_col doesn't exist.

    Notes
    -----
    The function uses libpysal's spatial weights to determine adjacency relationships.
    Edge geometries are created as LineStrings connecting polygon centroids.
    Self-connections and duplicate edges are automatically filtered out.
    """
    # Input validation
    _validate_gdf_input(private_gdf, "private_gdf")

    # Set default column name
    private_id_col = private_id_col or "tess_id"

    # Handle empty or insufficient data
    if private_gdf.empty or len(private_gdf) < 2:
        group_cols = [group_col or "group"]
        return _create_empty_edges_gdf(
            private_gdf.crs, "from_private_id", "to_private_id", group_cols,
        )

    # Validate contiguity parameter
    if contiguity not in {"queen", "rook"}:
        msg = "contiguity must be 'queen' or 'rook'"
        raise ValueError(msg)

    # Ensure ID column exists and validate group column
    private_gdf, private_id_col = _ensure_id_column(private_gdf, private_id_col, "tess_id")
    if group_col and group_col not in private_gdf.columns:
        msg = f"group_col '{group_col}' not found in private_gdf columns"
        raise ValueError(msg)

    # Reset index for consistent spatial weights computation
    gdf_indexed = private_gdf.reset_index(drop=True)

    # Create spatial weights matrix
    spatial_weights = _create_spatial_weights(gdf_indexed, contiguity)
    if spatial_weights is None or not spatial_weights.neighbors:
        group_cols = [group_col or "group"]
        return _create_empty_edges_gdf(
            private_gdf.crs, "from_private_id", "to_private_id", group_cols,
        )

    # Extract adjacency relationships
    adjacency_data = _extract_adjacency_relationships(
        spatial_weights, gdf_indexed, private_id_col, group_col,
    )

    if adjacency_data.empty:
        group_cols = [group_col or "group"]
        return _create_empty_edges_gdf(
            private_gdf.crs, "from_private_id", "to_private_id", group_cols,
        )

    # Create edge geometries
    edges_gdf = _create_adjacency_edges(adjacency_data, gdf_indexed, group_col or "group")

    return gpd.GeoDataFrame(
        edges_gdf,
        geometry="geometry",
        crs=private_gdf.crs,
    )


# ============================================================================
# PRIVATE TO PUBLIC GRAPH FUNCTIONS
# ============================================================================


def private_to_public_graph(
    private_gdf: gpd.GeoDataFrame,
    public_gdf: gpd.GeoDataFrame,
    private_id_col: str | None = None,
    public_id_col: str | None = None,
    public_geom_col: str | None = None,
    tolerance: float = 1.0,
) -> gpd.GeoDataFrame:
    """
    Create edges between private polygons and nearby public geometries.

    This function identifies spatial relationships between private spaces (tessellations)
    and public spaces (street segments) by finding intersections between buffered public
    geometries and private polygons.

    Parameters
    ----------
    private_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing private space polygons.
    public_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing public space geometries (typically LineStrings).
    private_id_col : str, optional
        Column name for private space identifiers. If None, uses "tess_id".
    public_id_col : str, optional
        Column name for public space identifiers. If None, uses "id".
    public_geom_col : str, optional
        Column name for alternative public geometry. If specified and exists,
        this geometry will be used instead of the main geometry column.
    tolerance : float, default=1.0
        Buffer distance for public geometries to detect proximity to private spaces.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing edge geometries between private and public spaces.
        Columns include private_id, public_id, and geometry.

    Raises
    ------
    TypeError
        If private_gdf or public_gdf are not GeoDataFrames.

    Notes
    -----
    Edge geometries are created as LineStrings connecting the centroids of
    private polygons and public geometries. The function uses spatial joins
    to identify overlapping areas within the specified tolerance.
    """
    # Input validation
    _validate_gdf_input(private_gdf, "private_gdf")
    _validate_gdf_input(public_gdf, "public_gdf")

    # Handle empty data
    if private_gdf.empty or public_gdf.empty:
        return _create_empty_edges_gdf(private_gdf.crs, "private_id", "public_id")

    # Set default column names and ensure they exist
    private_id_col = private_id_col or "tess_id"
    public_id_col = public_id_col or "id"
    private_gdf, private_id_col = _ensure_id_column(private_gdf, private_id_col, "tess_id")
    public_gdf, public_id_col = _ensure_id_column(public_gdf, public_id_col, "id")

    # Ensure CRS consistency
    public_gdf = _ensure_crs_consistency(private_gdf, public_gdf)

    # Determine which geometry to use for spatial join
    join_geom = (public_gdf[public_geom_col]
                if public_geom_col and public_geom_col in public_gdf.columns
                else public_gdf.geometry)

    # Create buffered geometries for spatial join
    buffered_public = gpd.GeoDataFrame(
        {public_id_col: public_gdf[public_id_col]},
        geometry=join_geom.buffer(tolerance),
        crs=public_gdf.crs,
    )

    # Perform spatial join to find intersections
    joined = gpd.sjoin(
        private_gdf[[private_id_col, "geometry"]],
        buffered_public,
        how="inner",
        predicate="intersects",
    )[[private_id_col, public_id_col]]

    # Return empty result if no intersections found
    if joined.empty:
        return _create_empty_edges_gdf(private_gdf.crs, "private_id", "public_id")

    # Remove duplicate connections
    joined = joined.drop_duplicates()

    # Calculate centroids for edge geometry creation
    private_centroids = (private_gdf.drop_duplicates(subset=private_id_col)
                        .set_index(private_id_col).geometry.centroid)
    public_centroids = (public_gdf.drop_duplicates(subset=public_id_col)
                       .set_index(public_id_col).geometry.centroid)

    # Create edge geometries as LineStrings between centroids
    p1 = private_centroids.loc[joined[private_id_col]].reset_index(drop=True)
    p2 = public_centroids.loc[joined[public_id_col]].reset_index(drop=True)
    joined["geometry"] = [
        LineString([(a.x, a.y), (b.x, b.y)])
        for a, b in zip(p1, p2, strict=True)
    ]

    # Rename columns to standard names
    joined = joined.rename(columns={
        private_id_col: "private_id",
        public_id_col: "public_id",
    })

    return gpd.GeoDataFrame(joined, geometry="geometry", crs=private_gdf.crs)


# ============================================================================
# PUBLIC TO PUBLIC GRAPH FUNCTIONS
# ============================================================================


def public_to_public_graph(
    public_gdf: gpd.GeoDataFrame,
    public_id_col: str | None = None,
    tolerance: float = 1e-8,
) -> gpd.GeoDataFrame:
    """
    Create edges between connected public segments based on topological connectivity.

    This function identifies topological connections between public space geometries
    (typically street segments) using the dual graph approach to find segments
    that share endpoints or connection points.

    Parameters
    ----------
    public_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing public space geometries (typically LineStrings).
    public_id_col : str, optional
        Column name for public space identifiers. If None, uses "id".
    tolerance : float, default=1e-8
        Distance tolerance for detecting endpoint connections between segments.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing edge geometries between connected public segments.
        Columns include from_public_id, to_public_id, and geometry.

    Raises
    ------
    TypeError
        If public_gdf is not a GeoDataFrame.

    Notes
    -----
    The function uses the dual graph approach where each LineString becomes a node,
    and edges represent topological connections between segments. Edge geometries
    are created as LineStrings connecting the centroids of connected segments.
    """
    # Input validation
    _validate_gdf_input(public_gdf, "public_gdf")

    # Handle empty or insufficient data
    if public_gdf.empty or len(public_gdf) < 2:
        return _create_empty_edges_gdf(public_gdf.crs, "from_public_id", "to_public_id")

    # Set default column name and ensure it exists
    public_id_col = public_id_col or "id"
    public_gdf, public_id_col = _ensure_id_column(public_gdf, public_id_col, "id")

    # Create dual graph to find connections
    nodes, connections = dual_graph(public_gdf, id_col=public_id_col, tolerance=tolerance)

    # Return empty result if no connections found
    if nodes.empty or not connections:
        return _create_empty_edges_gdf(public_gdf.crs, "from_public_id", "to_public_id")

    # Extract unique pairs from connections dictionary
    unique_pairs = {
        (min(k, v), max(k, v))
        for k, vs in connections.items()
        for v in vs
    }

    # Create DataFrame from connection pairs
    edges_df = pd.DataFrame(
        list(unique_pairs),
        columns=["from_public_id", "to_public_id"],
    )

    # Create edge geometries as LineStrings between node centroids
    p1 = nodes.loc[edges_df["from_public_id"], "geometry"].reset_index(drop=True)
    p2 = nodes.loc[edges_df["to_public_id"], "geometry"].reset_index(drop=True)
    edges_df["geometry"] = [
        LineString([(a.x, a.y), (b.x, b.y)])
        for a, b in zip(p1, p2, strict=True)
    ]

    return gpd.GeoDataFrame(edges_df, geometry="geometry", crs=public_gdf.crs)


# ============================================================================
# HELPER FUNCTIONS FOR VALIDATION AND DATA PROCESSING
# ============================================================================


def _validate_input_gdfs(buildings_gdf: gpd.GeoDataFrame, segments_gdf: gpd.GeoDataFrame) -> None:
    """
    Validate input GeoDataFrames for the morphological graph function.

    Parameters
    ----------
    buildings_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing building polygons
    segments_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing street segments

    Raises
    ------
    TypeError
        If inputs are not GeoDataFrames
    """
    if not isinstance(buildings_gdf, gpd.GeoDataFrame):
        msg = "buildings_gdf must be a GeoDataFrame"
        raise TypeError(msg)
    if not isinstance(segments_gdf, gpd.GeoDataFrame):
        msg = "segments_gdf must be a GeoDataFrame"
        raise TypeError(msg)


def _validate_gdf_input(gdf: gpd.GeoDataFrame, name: str) -> None:
    """
    Validate that input is a GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input to validate
    name : str
        Name of the parameter for error messages

    Raises
    ------
    TypeError
        If input is not a GeoDataFrame
    """
    if not isinstance(gdf, gpd.GeoDataFrame):
        msg = f"{name} must be a GeoDataFrame"
        raise TypeError(msg)


def _ensure_crs_consistency(target_gdf: gpd.GeoDataFrame, source_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Ensure CRS consistency between two GeoDataFrames.

    Parameters
    ----------
    target_gdf : geopandas.GeoDataFrame
        Target GeoDataFrame with the desired CRS
    source_gdf : geopandas.GeoDataFrame
        Source GeoDataFrame to potentially reproject

    Returns
    -------
    geopandas.GeoDataFrame
        Source GeoDataFrame reprojected to target CRS if necessary

    Warns
    -----
    RuntimeWarning
        If CRS mismatch is detected and reprojection is performed
    """
    if source_gdf.crs != target_gdf.crs:
        warnings.warn("CRS mismatch detected, reprojecting", RuntimeWarning, stacklevel=3)
        return source_gdf.to_crs(target_gdf.crs)
    return source_gdf


def _ensure_id_column(
    gdf: gpd.GeoDataFrame,
    column: str | None,
    default: str,
) -> tuple[gpd.GeoDataFrame, str]:
    """
    Ensure that an ID column exists in the GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame
    column : str, optional
        Preferred column name
    default : str
        Default column name to use

    Returns
    -------
    tuple[geopandas.GeoDataFrame, str]
        Tuple of (modified GeoDataFrame, actual column name used)
    """
    col = column or default
    if col in gdf.columns:
        return gdf, col

    gdf = gdf.copy()
    gdf[col] = range(len(gdf))
    return gdf, col


def _prepare_barriers(
    segments: gpd.GeoDataFrame,
    geom_col: str | None,
) -> gpd.GeoDataFrame:
    """
    Prepare barrier geometries for tessellation.

    Parameters
    ----------
    segments : geopandas.GeoDataFrame
        Street segments GeoDataFrame
    geom_col : str, optional
        Alternative geometry column name

    Returns
    -------
    geopandas.GeoDataFrame
        Prepared barriers GeoDataFrame
    """
    if geom_col and geom_col in segments.columns and geom_col != "geometry":
        return gpd.GeoDataFrame(
            segments.drop(columns=["geometry"]),
            geometry=segments[geom_col],
            crs=segments.crs,
        )
    return segments.copy()


def _filter_adjacent_tessellation(
    tess: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    max_distance: float = math.inf,
) -> gpd.GeoDataFrame:
    """
    Filter tessellation to only include cells adjacent to segments.

    Parameters
    ----------
    tess : geopandas.GeoDataFrame
        Tessellation GeoDataFrame
    segments : geopandas.GeoDataFrame
        Street segments GeoDataFrame
    max_distance : float, optional
        Maximum Euclidean distance between tessellation centroids and the
        nearest segment. If ``tess`` contains an ``enclosure_index`` column,
        distances are measured using only segments intersecting each enclosure.
        Defaults to ``math.inf`` which retains all cells.

    Returns
    -------
    geopandas.GeoDataFrame
        Filtered tessellation
    """
    if tess.empty or segments.empty:
        return tess.copy()

    if math.isinf(max_distance):
        return tess.copy()

    if max_distance is None:
        joined = gpd.sjoin(tess, segments, how="inner", predicate="intersects")
        return tess.loc[joined.index.unique()]

    encl_col = "enclosure_index" if "enclosure_index" in tess.columns else None

    if encl_col is None:
        segment_union = segments.unary_union
        centroids = tess.geometry.centroid
        distances = centroids.distance(segment_union)
        return tess.loc[distances <= max_distance].copy()

    filtered_parts: list[gpd.GeoDataFrame] = []
    for encl_id, group in tess.groupby(encl_col):
        enclosure_geom = group.unary_union
        segs = segments[segments.intersects(enclosure_geom)]
        if segs.empty:
            continue
        segment_union = segs.unary_union
        centroids = group.geometry.centroid
        distances = centroids.distance(segment_union)
        filtered = group.loc[distances <= max_distance]
        if not filtered.empty:
            filtered_parts.append(filtered)

    if not filtered_parts:
        return gpd.GeoDataFrame(columns=tess.columns, geometry="geometry", crs=tess.crs)

    return gpd.GeoDataFrame(pd.concat(filtered_parts), crs=tess.crs)


def _filter_tessellation_by_network_distance(
    tess: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | Point,
    max_distance: float,
) -> gpd.GeoDataFrame:
    """Filter tessellation by network distance from a center point."""
    if tess.empty or segments.empty:
        return tess.copy()

    graph = gdf_to_nx(edges=segments)
    pos_dict = _extract_node_positions(graph)
    if not pos_dict:
        return tess.copy()

    nodes = _create_nodes_gdf(pos_dict, "node_id", tess.crs)

    if isinstance(center_point, gpd.GeoDataFrame):
        center_geom = center_point.geometry.iloc[0]
    elif isinstance(center_point, gpd.GeoSeries):
        center_geom = center_point.iloc[0]
    else:
        center_geom = center_point

    center_node = _get_nearest_node(center_geom, nodes, node_id="node_id")

    try:
        distances = nx.single_source_dijkstra_path_length(
            graph, center_node, weight="length"
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return tess.iloc[0:0].copy()

    centroids = tess.geometry.centroid
    centroid_gdf = gpd.GeoDataFrame({"geometry": centroids}, crs=tess.crs)
    nearest = gpd.sjoin_nearest(
        centroid_gdf, nodes, how="left", distance_col="_dist"
    )

    total_dist = nearest["node_id"].map(distances).fillna(math.inf) + nearest["_dist"].fillna(math.inf)

    keep = total_dist <= max_distance
    return tess.loc[keep].copy()


def _add_building_info(
    tess: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Add building information to tessellation.

    Parameters
    ----------
    tess : geopandas.GeoDataFrame
        Tessellation GeoDataFrame
    buildings : geopandas.GeoDataFrame
        Buildings GeoDataFrame

    Returns
    -------
    geopandas.GeoDataFrame
        Tessellation with building information
    """
    if buildings.empty:
        return tess.copy()

    joined = gpd.sjoin(tess, buildings, how="left", predicate="intersects")
    if "index_right" in joined.columns:
        mapping = buildings.geometry.to_dict()
        joined["building_geometry"] = joined["index_right"].map(mapping)
        joined = joined.drop(columns=["index_right"])
    return joined


def _create_empty_edges_gdf(
    crs: str | int | None,
    from_col: str,
    to_col: str,
    extra_cols: list[str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Create an empty edges GeoDataFrame with specified column structure.

    Parameters
    ----------
    crs : str, int, or None
        Coordinate reference system
    from_col : str
        Name of the 'from' ID column
    to_col : str
        Name of the 'to' ID column
    extra_cols : list[str], optional
        Additional columns to include

    Returns
    -------
    geopandas.GeoDataFrame
        Empty GeoDataFrame with specified columns
    """
    columns = [from_col, to_col]
    if extra_cols:
        columns.extend(extra_cols)
    columns.append("geometry")

    return gpd.GeoDataFrame(columns=columns, geometry="geometry", crs=crs)


def _set_index_if_exists(gdf: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
    """
    Set index if column exists in GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame
    col : str
        Column name to use as index

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with index set if column exists
    """
    return gdf.set_index(col) if col in gdf.columns else gdf


def _set_edge_index(
    gdf: gpd.GeoDataFrame,
    from_col: str,
    to_col: str,
) -> gpd.GeoDataFrame:
    """
    Set multi-index for edge GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Edge GeoDataFrame
    from_col : str
        'From' column name
    to_col : str
        'To' column name

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with multi-index set
    """
    if not gdf.empty and from_col in gdf.columns and to_col in gdf.columns:
        return gdf.set_index([from_col, to_col])
    return gdf


# ============================================================================
# HELPER FUNCTIONS FOR SPATIAL WEIGHTS AND ADJACENCY
# ============================================================================


def _create_spatial_weights(gdf: gpd.GeoDataFrame, contiguity: str) -> libpysal.weights.W | None:
    """
    Create spatial weights matrix for adjacency analysis.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame with polygon geometries
    contiguity : str
        Type of contiguity ("queen" or "rook")

    Returns
    -------
    libpysal.weights.W or None
        Spatial weights matrix, or None if creation fails
    """
    try:
        if contiguity == "queen":
            return libpysal.weights.Queen.from_dataframe(gdf)
        return libpysal.weights.Rook.from_dataframe(gdf)
    except (ValueError, IndexError):
        logger.warning("Could not create spatial weights matrix", stacklevel=2)
        return None


def _extract_adjacency_relationships(
    spatial_weights: libpysal.weights.W,
    gdf: gpd.GeoDataFrame,
    id_col: str,
    group_col: str | None,
) -> pd.DataFrame:
    """
    Extract adjacency relationships from spatial weights matrix.

    Parameters
    ----------
    spatial_weights : libpysal.weights.W
        Spatial weights matrix
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame
    id_col : str
        ID column name
    group_col : str, optional
        Group column name for filtering

    Returns
    -------
    pandas.DataFrame
        DataFrame with adjacency relationships
    """
    # Convert to sparse matrix and extract adjacency pairs
    coo = spatial_weights.sparse.tocoo()
    mask = coo.row < coo.col  # Avoid duplicate pairs
    rows = coo.row[mask]
    cols = coo.col[mask]

    # Extract IDs for connected polygons
    from_ids = gdf.iloc[rows][id_col].to_numpy()
    to_ids = gdf.iloc[cols][id_col].to_numpy()

    # Filter by group if specified
    if group_col:
        grp_i = gdf.iloc[rows][group_col].to_numpy()
        grp_j = gdf.iloc[cols][group_col].to_numpy()
        valid = grp_i == grp_j

        rows = rows[valid]
        cols = cols[valid]
        from_ids = from_ids[valid]
        to_ids = to_ids[valid]
        groups = grp_i[valid]
    else:
        groups = ["all"] * len(from_ids)
        group_col = "group"

    # Filter out self-connections
    valid = from_ids != to_ids
    rows = rows[valid]
    cols = cols[valid]
    from_ids = from_ids[valid]
    to_ids = to_ids[valid]
    groups = [groups[i] for i in range(len(groups)) if valid[i]]

    return pd.DataFrame({
        "row": rows,
        "col": cols,
        "from_private_id": from_ids,
        "to_private_id": to_ids,
        group_col: groups,
    })


def _create_adjacency_edges(
    adjacency_data: pd.DataFrame,
    gdf: gpd.GeoDataFrame,
    group_col: str,
) -> pd.DataFrame:
    """
    Create edge geometries from adjacency relationships.

    Parameters
    ----------
    adjacency_data : pandas.DataFrame
        DataFrame with adjacency relationships
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame with geometries
    group_col : str
        Group column name

    Returns
    -------
    pandas.DataFrame
        DataFrame with edge geometries
    """
    # Ensure consistent ordering of edges to avoid duplicates
    adjacency_data[["from_private_id", "to_private_id"]] = pd.DataFrame(
        sorted(pair) for pair in
        adjacency_data[["from_private_id", "to_private_id"]].to_numpy()
    )
    adjacency_data = adjacency_data.drop_duplicates(subset=["from_private_id", "to_private_id"])

    # Calculate centroids for edge geometry creation
    centroids = gdf.geometry.centroid
    p1 = centroids.iloc[adjacency_data["row"]].reset_index(drop=True)
    p2 = centroids.iloc[adjacency_data["col"]].reset_index(drop=True)

    # Create LineString geometries
    adjacency_data["geometry"] = [
        LineString([(a.x, a.y), (b.x, b.y)])
        for a, b in zip(p1, p2, strict=True)
    ]

    # Clean up and return final columns
    return adjacency_data.drop(columns=["row", "col"])[
        ["from_private_id", "to_private_id", group_col, "geometry"]
    ]
