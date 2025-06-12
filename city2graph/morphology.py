"""Module for creating morphological graphs from urban data."""

import logging
import math
import warnings

import geopandas as gpd
import libpysal
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from shapely.creation import linestrings as sh_linestrings
from shapely.geometry import Point

from .utils import create_tessellation
from .utils import dual_graph
from .utils import filter_graph_by_distance
from .utils import gdf_to_nx

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
    clipping_buffer: float = math.inf,
    public_geom_col: str | None = "barrier_geometry",
    contiguity: str = "queen",
    keep_buildings: bool = False,
    private_id_col: str | None = None,  # Made optional
    public_id_col: str | None = None,    # Made optional
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
    clipping_buffer : float, default=math.inf
        Additional buffer distance (non-negative) to be added to `distance` when
        `distance` and `center_point` are specified.
        This sum (`distance + clipping_buffer`) is used as the radius for filtering
        `segs_buffer` (segments used for tessellation context).
        If `clipping_buffer` is `math.inf` (and `distance` is set), `segs_buffer` is
        filtered by `distance` alone.
        The `max_distance` for `_filter_adjacent_tessellation` becomes
        `distance + clipping_buffer` (this evaluates to `math.inf` if `clipping_buffer`
        is `math.inf` or if `distance` is not set).
        If `distance` is not provided, `clipping_buffer` is effectively ignored for
        `segs_buffer` filtering, and `max_distance` for `_filter_adjacent_tessellation`
        defaults to `math.inf`.
        Must be non-negative. Defaults to `math.inf`.
    public_geom_col : str, optional
        Column name containing alternative geometry for public spaces. If specified and exists,
        this geometry will be used instead of the main geometry column for tessellation barriers.
        Default is "barrier_geometry".
    contiguity : str, default="queen"
        Type of spatial contiguity for private-to-private connections.
        Must be either "queen" or "rook".
    keep_buildings : bool, default=False
        If True, preserves building information in the tessellation output.
    private_id_col : str, optional
        Column name for private space (tessellation) identifiers.
        If None, defaults to "tess_id". If the column doesn't exist, it will be created.
    public_id_col : str, optional
        Column name for public space (segment) identifiers.
        If None, defaults to "id". If the column doesn't exist, it will be created.

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
        If clipping_buffer is negative.

    Notes
    -----
    The function first filters the street network by `distance` (resulting in `segs`).
    A `segs_buffer` GeoDataFrame is also created for tessellation context, potentially
    filtered by `distance + clipping_buffer` or `distance` if `center_point` and
    `distance` are provided. This `segs_buffer` is used to create enclosures and
    tessellations.
    It then establishes three types of relationships:
    1. Private-to-private: Adjacency between tessellation cells (handled by private_to_private_graph)
    2. Public-to-public: Topological connectivity between street segments
    3. Private-to-public: Spatial interfaces between tessellations and streets

    The output follows a heterogeneous graph structure suitable for network analysis
    of urban morphology.
    """
    # Validate input GeoDataFrames
    _validate_input_gdfs(buildings_gdf, segments_gdf)

    # Validate clipping_buffer
    if clipping_buffer < 0:
        msg = "clipping_buffer cannot be negative."
        raise ValueError(msg)

    # Ensure CRS consistency
    segments_gdf = _ensure_crs_consistency(buildings_gdf, segments_gdf)

    # Ensure ID column on original segments before filtering
    # _ensure_id_column will use 'public_id' as default if public_id_col is None
    segments_gdf, public_id_col_actual = _ensure_id_column(segments_gdf, public_id_col, "public_id")

    # Filter segments by network distance for the final graph
    if center_point is not None and distance is not None and not segments_gdf.empty:
        segs = filter_graph_by_distance(segments_gdf, center_point, distance)
    else:
        segs = segments_gdf

    # Create a buffered version of the graph for tessellation creation (segs_buffer)
    # This segs_buffer is used for _prepare_barriers -> create_tessellation
    # and as the segments context for _filter_adjacent_tessellation.
    if center_point is not None and distance is not None and not segments_gdf.empty:
        if not math.isinf(clipping_buffer):
            # Finite clipping_buffer: use distance + clipping_buffer for segs_buffer radius
            segs_buffer_radius = distance + clipping_buffer
            segs_buffer = filter_graph_by_distance(segments_gdf, center_point, segs_buffer_radius)
        else:  # clipping_buffer is math.inf
            # Fallback to 'distance' as radius for segs_buffer
            segs_buffer = filter_graph_by_distance(segments_gdf, center_point, distance)
    else:
        # No center_point or no distance, so segs_buffer is not filtered by distance
        segs_buffer = segments_gdf

    # Prepare barriers from the buffered segments and create tessellation
    barriers = _prepare_barriers(segs_buffer, public_geom_col)
    tessellation = create_tessellation(
        buildings_gdf,
        primary_barriers=None if barriers.empty else barriers,
    )
    # _ensure_id_column will use 'private_id' as default if private_id_col is None
    tessellation, private_id_col_actual = _ensure_id_column(tessellation, private_id_col, "private_id")

    # Determine max_distance for _filter_adjacent_tessellation
    max_distance_for_adj_filter = distance + clipping_buffer if distance is not None else math.inf

    # Filter tessellation to only include areas adjacent to the buffered segments
    tessellation = _filter_adjacent_tessellation(
        tessellation,
        segs,
        max_distance=max_distance_for_adj_filter,
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
        private_id_col=private_id_col_actual,
        group_col="enclosure_index",
        contiguity=contiguity,
    )
    pub_pub = public_to_public_graph(segs, public_id_col=public_id_col_actual)
    priv_pub = private_to_public_graph(
        tessellation,
        segs,
        private_id_col=private_id_col_actual,
        public_id_col=public_id_col_actual,
        public_geom_col=public_geom_col,
    )

    # Log warning if no private-public connections found
    if priv_pub.empty:
        logger.warning("No private to public connections found")

    # Organize output as heterogeneous graph structure
    nodes = {
        "private": _set_index_if_exists(tessellation, private_id_col_actual),
        "public": _set_index_if_exists(segs, public_id_col_actual),
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
        If contiguity not in {"queen", "rook"}, or if group_col doesn't exist.

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
    join_geom_series = (public_gdf[public_geom_col]
                        if public_geom_col and public_geom_col in public_gdf.columns
                        else public_gdf.geometry)

    # Create buffered geometries for spatial join
    # Ensure public_id_col is part of buffered_public for the join
    buffered_public_data = {public_id_col: public_gdf[public_id_col]}
    buffered_public = gpd.GeoDataFrame(
        buffered_public_data,
        geometry=join_geom_series.buffer(tolerance),
        crs=public_gdf.crs,
    )

    # Perform spatial join to find intersections
    # Select only necessary columns for the join to avoid large intermediate frames
    joined = gpd.sjoin(
        private_gdf[[private_id_col, "geometry"]],
        buffered_public, # Contains public_id_col and its geometry
        how="inner",
        predicate="intersects",
    )

    # After sjoin, 'joined' will have private_id_col, geometry (from private_gdf),
    # public_id_col (from buffered_public), and potentially 'index_right'.
    # We only need the ID columns for now.
    if not joined.empty:
        id_cols_to_keep = [private_id_col, public_id_col]
        joined = joined[id_cols_to_keep]

    # Return empty result if no intersections found
    if joined.empty:
        return _create_empty_edges_gdf(private_gdf.crs, "private_id", "public_id")

    # Remove duplicate connections
    joined = joined.drop_duplicates()
    if joined.empty: # Check again after drop_duplicates
        return _create_empty_edges_gdf(private_gdf.crs, "private_id", "public_id")

    # Calculate centroids for edge geometry creation
    # Using drop_duplicates before set_index to ensure unique index for centroid map
    private_centroids_map = (private_gdf.drop_duplicates(subset=[private_id_col])
                             .set_index(private_id_col).geometry.centroid)
    public_centroids_map = (public_gdf.drop_duplicates(subset=[public_id_col])
                            .set_index(public_id_col).geometry.centroid)

    # Use .copy() for geometry assignment to avoid SettingWithCopyWarning
    joined_with_geom = joined.copy()

    # Retrieve the centroid geometries based on the IDs in 'joined_with_geom'
    # .loc will raise KeyError if an ID is not found, but sjoin should ensure valid IDs.
    p1_geoms = private_centroids_map.loc[joined_with_geom[private_id_col]].reset_index(drop=True)
    p2_geoms = public_centroids_map.loc[joined_with_geom[public_id_col]].reset_index(drop=True)

    if p1_geoms.empty or p2_geoms.empty or len(p1_geoms) != len(p2_geoms):
        # Fallback: assign an empty GeoSeries or None if geometries can't be formed
        joined_with_geom["geometry"] = gpd.GeoSeries([None] * len(joined_with_geom), crs=private_gdf.crs)
    else:
        # Vectorized coordinate extraction
        coords_p1 = np.array(list(zip(p1_geoms.x, p1_geoms.y, strict=False)))
        coords_p2 = np.array(list(zip(p2_geoms.x, p2_geoms.y, strict=False)))

        # Create an array of coordinate pairs for LineStrings
        line_coords = np.stack((coords_p1, coords_p2), axis=1)

        # Create LineString geometries
        joined_with_geom["geometry"] = list(sh_linestrings(line_coords))

    # Rename columns to standard names
    joined_with_geom = joined_with_geom.rename(columns={
        private_id_col: "private_id",
        public_id_col: "public_id",
    })

    return gpd.GeoDataFrame(joined_with_geom, geometry="geometry", crs=private_gdf.crs)


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
    # nodes is a GeoDataFrame (Points) indexed by public_id_col, connections is a dict
    nodes, connections = dual_graph(public_gdf, id_col=public_id_col, tolerance=tolerance)

    # Return empty result if no connections found
    if nodes.empty or not connections: # nodes is GDF, connections is dict
        return _create_empty_edges_gdf(public_gdf.crs, "from_public_id", "to_public_id")

    # Extract unique pairs from connections dictionary
    # (min(k,v), max(k,v)) ensures undirected edges (A,B) is same as (B,A) and avoids duplicates
    unique_pairs = {
        (min(k, v_node), max(k, v_node))
        for k, v_list in connections.items()
        for v_node in v_list
        if k != v_node # Explicitly filter self-loops if dual_graph could produce them
    }
    # If dual_graph guarantees k != v_node, the condition can be removed.
    # For safety, it's good to keep if the behavior of dual_graph isn't strictly no-self-loops.
    # If k can equal v_node and self-loops are desired, remove `if k != v_node`.
    # Assuming typical graph edges are between distinct nodes.

    if not unique_pairs: # No valid (non-self-loop) pairs formed
        return _create_empty_edges_gdf(public_gdf.crs, "from_public_id", "to_public_id")

    # Create DataFrame from connection pairs
    edges_df = pd.DataFrame(
        list(unique_pairs),
        columns=["from_public_id", "to_public_id"],
    )

    # Create edge geometries as LineStrings between node centroids
    # 'nodes' GeoDataFrame has Point geometries, indexed by public_id_col.
    # .loc accesses rows by index (ID), .geometry accesses the geometry Series.
    p1_geoms = nodes.loc[edges_df["from_public_id"]].geometry.reset_index(drop=True)
    p2_geoms = nodes.loc[edges_df["to_public_id"]].geometry.reset_index(drop=True)

    if p1_geoms.empty or p2_geoms.empty or len(p1_geoms) != len(p2_geoms):
        # Fallback if geometries can't be formed (e.g., if edges_df non-empty but loc fails)
        edges_df["geometry"] = gpd.GeoSeries([None] * len(edges_df), crs=public_gdf.crs)
    else:
        # Vectorized coordinate extraction
        coords_p1 = np.array(list(zip(p1_geoms.x, p1_geoms.y, strict=False)))
        coords_p2 = np.array(list(zip(p2_geoms.x, p2_geoms.y, strict=False)))

        # Create an array of coordinate pairs for LineStrings
        line_coords = np.stack((coords_p1, coords_p2), axis=1)

        # Create LineString geometries
        edges_df["geometry"] = list(sh_linestrings(line_coords))

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
        Street segments GeoDataFrame to measure distance against.
    max_distance : float, optional
        Maximum Euclidean distance between tessellation centroids and the
        nearest segment (from the `segments` GeoDataFrame).
        In `morphological_graph`, this is typically derived from `distance + clipping_buffer`
        if `distance` is specified, or `math.inf` otherwise.
        If ``tess`` contains an ``enclosure_index`` column,
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
    for _, group in tess.groupby(encl_col):
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


def _build_spatial_graph(
    segments: gpd.GeoDataFrame,
    tess_centroids: gpd.GeoSeries,
) -> tuple[nx.Graph, dict[int, str], dict[str, tuple[float, float]]]:
    """Build a spatial graph from segments and tessellation centroids."""
    graph = gdf_to_nx(edges=segments)
    seg_nodes_pos_dict = nx.get_node_attributes(graph, "pos")

    centroid_iloc_to_node_id: dict[int, str] = {}
    new_nodes_for_graph = []
    for i, pt in enumerate(tess_centroids):
        node_id = f"tess_centroid_{i}"
        centroid_iloc_to_node_id[i] = node_id
        new_nodes_for_graph.append((node_id, {"pos": (pt.x, pt.y), "type": "centroid_node"}))
    graph.add_nodes_from(new_nodes_for_graph)
    return graph, centroid_iloc_to_node_id, seg_nodes_pos_dict


def _connect_centroids_to_segment_graph(
    graph: nx.Graph,
    centroids: gpd.GeoSeries,
    centroid_iloc_to_node_id: dict[int, str],
    seg_nodes_pos_dict: dict[str, tuple[float, float]],
) -> None:
    """Connect centroid nodes to the nearest segment graph nodes using KDTree."""
    if not seg_nodes_pos_dict or centroids.empty:
        return

    seg_node_ids_list = list(seg_nodes_pos_dict.keys())
    seg_node_coords_list = [list(coord) for coord in seg_nodes_pos_dict.values()]

    if not seg_node_coords_list:
        return

    seg_node_coords_array = np.array(seg_node_coords_list)
    if seg_node_coords_array.ndim != 2 or seg_node_coords_array.shape[1] != 2:
        return # Invalid coordinates array

    tree = KDTree(seg_node_coords_array)

    centroid_coords_list = [(pt.x, pt.y) for pt in centroids.tolist()]
    if not centroid_coords_list:
        return

    centroid_coords_array = np.array(centroid_coords_list)
    if centroid_coords_array.ndim != 2 or centroid_coords_array.shape[1] != 2:
        return # Invalid coordinates array

    distances_to_seg, indices_in_seg_nodes = tree.query(centroid_coords_array)

    for i, centroid_node_id in enumerate(centroid_iloc_to_node_id.values()):
        if 0 <= indices_in_seg_nodes[i] < len(seg_node_ids_list):
            nearest_seg_node_id_in_graph = seg_node_ids_list[indices_in_seg_nodes[i]]
            graph.add_edge(
                centroid_node_id,
                nearest_seg_node_id_in_graph,
                length=distances_to_seg[i],
            )
        # else:
            # logger.warning(f"KDTree returned invalid index {indices_in_seg_nodes[i]} for centroid {i}")


def _connect_centroids_to_centroids(
    graph: nx.Graph,
    centroids: gpd.GeoSeries,
    centroid_iloc_to_node_id: dict[int, str],
) -> None:
    """Connect centroid nodes to each other based on Euclidean distance."""
    if len(centroids) < 2:
        return

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            node_id_i = centroid_iloc_to_node_id[i]
            node_id_j = centroid_iloc_to_node_id[j]
            point_i = centroids.iloc[i]
            point_j = centroids.iloc[j]
            distance = point_i.distance(point_j)
            graph.add_edge(node_id_i, node_id_j, length=distance)


def _find_closest_node_to_center(
    graph: nx.Graph,
    center_point_geom: Point,
) -> str | None:
    """Find the graph node closest to the geographic center point."""
    all_node_pos_dict = nx.get_node_attributes(graph, "pos")
    if not all_node_pos_dict:
        return None

    # Using a generator expression for min function for slight memory efficiency
    closest_node_id, _ = min(
        all_node_pos_dict.items(),
        key=lambda item: Point(item[1]).distance(center_point_geom),
    )
    return closest_node_id


def _filter_nodes_by_path_length(
    graph: nx.Graph,
    source_node_id: str,
    max_distance: float,
    centroid_iloc_to_node_id: dict[int, str],
) -> list[int]:
    """Filter nodes by shortest path length from a source node."""
    try:
        lengths = nx.single_source_dijkstra_path_length(graph, source_node_id, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    return [
        iloc
        for iloc, node_id in centroid_iloc_to_node_id.items()
        if lengths.get(node_id, math.inf) <= max_distance
    ]


def _filter_tessellation_by_network_distance(
    tess: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | Point,
    max_distance: float,
) -> gpd.GeoDataFrame:
    """Filter tessellation by network distance from a center point."""
    if tess.empty or segments.empty:
        return tess.copy()

    centroids = tess.geometry.centroid
    graph, centroid_iloc_to_node_id, seg_nodes_pos_dict = _build_spatial_graph(segments, centroids)

    _connect_centroids_to_segment_graph(
        graph, centroids, centroid_iloc_to_node_id, seg_nodes_pos_dict,
    )
    _connect_centroids_to_centroids(graph, centroids, centroid_iloc_to_node_id)

    if isinstance(center_point, gpd.GeoDataFrame):
        center_geom = center_point.geometry.iloc[0]
    elif isinstance(center_point, gpd.GeoSeries):
        center_geom = center_point.iloc[0]
    else:
        center_geom = center_point  # Assuming center_point is a Shapely Point

    center_node_id_in_graph = _find_closest_node_to_center(graph, center_geom)
    if center_node_id_in_graph is None:
        return tess.iloc[0:0].copy() # No nodes with position or graph is empty

    keep_ilocs = _filter_nodes_by_path_length(
        graph, center_node_id_in_graph, max_distance, centroid_iloc_to_node_id,
    )

    return tess.iloc[sorted(keep_ilocs)].copy()


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
        groups = np.full(len(from_ids), "all", dtype=object) # Use numpy for potential efficiency
        group_col = "group"

    # The condition from_ids != to_ids should ideally always be true if id_col contains unique IDs
    # and spatial_weights.sparse.tocoo() with row < col mask is used.
    # Kept for safety.
    valid_ids_filter = from_ids != to_ids
    from_ids = from_ids[valid_ids_filter]
    to_ids = to_ids[valid_ids_filter]
    groups = groups[valid_ids_filter]
    # Need to filter rows and cols as well if they are used later by _create_adjacency_edges
    rows = rows[valid_ids_filter]
    cols = cols[valid_ids_filter]


    return pd.DataFrame({
        "row": rows, # row index from original gdf for centroid lookup
        "col": cols, # col index from original gdf for centroid lookup
        "from_private_id": from_ids,
        "to_private_id": to_ids,
        group_col: groups,
    })


def _create_adjacency_edges(
    adjacency_data: pd.DataFrame,
    gdf: gpd.GeoDataFrame, # This gdf is the original private_gdf, indexed 0..N-1
    group_col: str,
) -> pd.DataFrame: # Returns a DataFrame, to be converted to GeoDataFrame by caller
    """
    Create edge geometries from adjacency relationships.

    Parameters
    ----------
    adjacency_data : pandas.DataFrame
        DataFrame with adjacency relationships (must include 'row', 'col',
        'from_private_id', 'to_private_id', and group_col columns)
    gdf : geopandas.GeoDataFrame
        Input GeoDataFrame with geometries (assumed to have a simple 0..N-1 index
        corresponding to 'row'/'col' in adjacency_data)
    group_col : str
        Group column name present in adjacency_data

    Returns
    -------
    pandas.DataFrame
        DataFrame with edge geometries
    """
    if adjacency_data.empty:
        return pd.DataFrame(columns=["from_private_id", "to_private_id", group_col, "geometry"])

    adj_data_processed = adjacency_data.copy()

    id_pairs = adj_data_processed[["from_private_id", "to_private_id"]].to_numpy()
    sorted_id_pairs = np.sort(id_pairs, axis=1)
    adj_data_processed["from_private_id"] = sorted_id_pairs[:, 0]
    adj_data_processed["to_private_id"] = sorted_id_pairs[:, 1]

    adj_data_processed = adj_data_processed.drop_duplicates(
        subset=["from_private_id", "to_private_id", group_col],
    )

    if adj_data_processed.empty:
        return pd.DataFrame(columns=["from_private_id", "to_private_id", group_col, "geometry"])

    centroids = gdf.geometry.centroid

    rows_idx = adj_data_processed["row"].to_numpy()
    cols_idx = adj_data_processed["col"].to_numpy()

    points_p1 = centroids.iloc[rows_idx]
    points_p2 = centroids.iloc[cols_idx]

    if points_p1.empty or points_p2.empty or len(points_p1) != len(points_p2):
        adj_data_processed["geometry"] = None
    else:
        coords_p1 = np.array(list(zip(points_p1.x, points_p1.y, strict=True)))
        coords_p2 = np.array(list(zip(points_p2.x, points_p2.y, strict=True)))

        line_coords = np.stack((coords_p1, coords_p2), axis=1)
        adj_data_processed["geometry"] = list(sh_linestrings(line_coords))

    columns_to_drop = [col for col in ["row", "col"] if col in adj_data_processed.columns]
    final_columns = ["from_private_id", "to_private_id", group_col, "geometry"]
    return adj_data_processed.drop(columns=columns_to_drop)[final_columns]
