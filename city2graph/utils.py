"""Module for loading and processing geospatial data from Overture Maps."""

import logging
from typing import Any

import geopandas as gpd
import momepy
import networkx as nx
import pandas as pd
import shapely
from shapely.geometry import LineString
from shapely.geometry import Point

# Define the public API for this module
__all__ = [
    "create_isochrone",
    "create_tessellation",
    "dual_graph",
    "filter_graph_by_distance",
    "gdf_to_nx",
    "nx_to_gdf",
]

logger = logging.getLogger(__name__)


# ============================================================================
# TESSELLATION FUNCTIONS
# ============================================================================


def create_tessellation(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    primary_barriers: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    shrink: float = 0.4,
    segment: float = 0.5,
    threshold: float = 0.05,
    n_jobs: int = -1,
    **kwargs: Any) -> gpd.GeoDataFrame:  # noqa: ANN401
    """
    Create tessellations from the given geometries.

    If primary_barriers are provided, enclosed tessellations are created.
    If not, morphological tessellations are created.
    For more details, see momepy.enclosed_tessellation and momepy.morphological_tessellation.

    Parameters
    ----------
    geometry : Union[geopandas.GeoDataFrame, geopandas.GeoSeries]
        Input geometries to create a tessellation for. Should contain the geometries to tessellate.
    primary_barriers : Optional[Union[gpd.GeoDataFrame, gpd.GeoSeries]], default=None
        Optional GeoDataFrame or GeoSeries containing barriers to use for enclosed tessellation.
        If provided, the function will create enclosed tessellation using these barriers.
        If None, morphological tessellation will be created using the input geometries.
    shrink : float, default=0.4
        Distance for negative buffer of tessellations. By default, 0.4.
        Used for both momepy.enclosed_tessellation and momepy.morphological_tessellation.
    segment : float, default=0.5
        Maximum distance between points for tessellations. By default, 0.5.
        Used for both momepy.enclosed_tessellation and momepy.morphological_tessellation.
    threshold : float, default=0.05
        Minimum threshold for a building to be considered within an enclosure. By default, 0.05.
        Used for both momepy.enclosed_tessellation.
    n_jobs : int, default=-1
        Number of parallel jobs to run. By default, -1 (use all available cores).
        Used for both momepy.enclosed_tessellation.
    **kwargs : Any
        Additional keyword arguments to pass to momepy.enclosed_tessellation.
        These can include parameters specific to the tessellation method used.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the tessellation polygons.
    """
    # Validate inputs and ensure consistency
    geometry, primary_barriers = _validate_gdf(geometry, primary_barriers, check_crs=True)

    # Create tessellation using momepy based on whether primary_barriers are provided
    if primary_barriers is not None:

        # Create enclosures for enclosed tessellation
        enclosures = momepy.enclosures(
            primary_barriers=primary_barriers,
            limit=None,
            additional_barriers=None,
            enclosure_id="eID",
            clip=False,
        )

        try:
            tessellation = momepy.enclosed_tessellation(
                geometry=geometry,
                enclosures=enclosures,
                shrink=shrink,
                segment=segment,
                threshold=threshold,
                n_jobs=n_jobs,
                **kwargs,
            )
        except (ValueError, TypeError) as e:
            if "No objects to concatenate" in str(e) or "incorrect geometry type" in str(e):
                # Return empty tessellation when momepy can't create valid tessellations
                return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=geometry.crs)
            raise

        # Apply ID handling for enclosed tessellation
        tessellation["tess_id"] = [
            f"{i}_{j}"
            for i, j in zip(tessellation["enclosure_index"], tessellation.index, strict=False)
        ]
        tessellation = tessellation.reset_index(drop=True)
    else:
        # Disallow geographic CRS
        if hasattr(geometry, "crs") and geometry.crs == "EPSG:4326":
            msg = "Geometry is in a geographic CRS"
            raise ValueError(msg)
        # Create morphological tessellation
        try:
            tessellation = momepy.morphological_tessellation(
                geometry=geometry, clip="bounding_box", shrink=shrink, segment=segment,
            )
        except (ValueError, TypeError) as e:
            if "No objects to concatenate" in str(e) or "incorrect geometry type" in str(e):
                # Return empty tessellation when momepy can't create valid tessellations
                return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=geometry.crs)
            raise
        tessellation["tess_id"] = tessellation.index

    return tessellation


# ============================================================================
# DUAL GRAPH FUNCTIONS
# ============================================================================

def dual_graph(gdf: gpd.GeoDataFrame, keep_original_geom: bool = False) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Convert a GeoDataFrame of LineStrings to a graph representation with nodes and edges.

    Based on momepy's dual graph approach. Nodes represent original LineStrings (with Point
    geometry at centroids), and edges represent connectivity from momepy.

    The process is:
    1. Keep the original index of `gdf` in a dedicated column.
    2. Create a dual graph (NetworkX graph) using momepy.gdf_to_nx.
    3. Convert the NetworkX graph back to GeoDataFrames (nodes_gdf, edges_gdf)
       using city2graph.nx_to_gdf.
    4. Establish a mapping from temporary momepy-generated node IDs to the original IDs.
    5. Set the index of the final nodes_gdf to be the original IDs.
    6. Convert the MultiIndex of the final edges_gdf to use original IDs.
    7. If `keep_original_geom` is True, add the original geometry of `gdf` to `nodes_gdf`
       as a new column named "original_geometry".

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing LineString geometries. Must have a CRS.
    keep_original_geom : bool, default=False
        If True, the original geometry of `gdf` is preserved in the output `nodes_gdf`
        as a GeoSeries column named "original_geometry".

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        - nodes_gdf: Nodes with original identifiers as index and Point geometry.
                     Other attributes from the input `gdf` are preserved.
        - edges_gdf: Edges with MultiIndex of original identifiers and LineString geometry
                     representing connections from momepy's dual graph.
    """
    # Validate input GeoDataFrame
    if not isinstance(gdf, gpd.GeoDataFrame):
        msg = "Input `gdf` must be a GeoDataFrame."
        raise TypeError(msg)

    # Optionally preserve original geometry
    if keep_original_geom:
        gdf["original_geometry"] = gdf.geometry.copy()

    # Determine a unique column name for original IDs
    original_id_col_name = _get_original_id_col_name(gdf)
    # Define schema for node and edge GeoDataFrames
    node_schema_cols = [col for col in gdf.columns if col != gdf.geometry.name] + ["geometry"]
    edge_schema_cols = ["geometry"]

    # Create empty GeoDataFrames for nodes and edges
    empty_nodes_gdf, empty_edges_gdf = _create_empty_dual_graph_gdfs(
        gdf.crs, original_id_col_name, node_schema_cols, edge_schema_cols,
    )

    # Return empty GDFs if input is empty
    if gdf.empty:
        return empty_nodes_gdf, empty_edges_gdf

    # Ensure input GDF has a CRS
    if gdf.crs is None:
        msg = "Input `gdf` must have a CRS."
        raise ValueError(msg)

    # Ensure all geometries are LineStrings
    if not all(isinstance(geom, shapely.geometry.LineString) for geom in gdf.geometry if geom is not None):
        msg = "All valid geometries in input `gdf` must be LineString for dual graph."
        raise ValueError(msg)

    # Prepare GDF for momepy: store original index and reset current index
    processed_gdf = gdf.copy()
    processed_gdf[original_id_col_name] = gdf.index
    gdf_for_momepy = processed_gdf.reset_index(drop=True)

    # Perform graph conversion using momepy and nx_to_gdf
    _, nodes_intermediate_gdf, edges_intermediate_gdf = _perform_graph_conversion(
        gdf_for_momepy, empty_nodes_gdf, empty_edges_gdf,
    )
    # If conversion failed, return empty GDFs
    if nodes_intermediate_gdf is empty_nodes_gdf or edges_intermediate_gdf is empty_edges_gdf :
         return empty_nodes_gdf, empty_edges_gdf # Return the specific empty ones passed

    # Check if original ID column is present in intermediate nodes GDF
    if original_id_col_name not in nodes_intermediate_gdf.columns:
        logger.error(
            "Column '%s' not found in intermediate nodes GDF after nx_to_gdf.", original_id_col_name,
        )
        return empty_nodes_gdf, empty_edges_gdf

    # Create a mapping from momepy's node IDs to original IDs
    id_map_series = nodes_intermediate_gdf[original_id_col_name]
    # Set the index of the final nodes GDF to original IDs
    final_nodes_gdf = nodes_intermediate_gdf.set_index(original_id_col_name, drop=True)
    final_nodes_gdf.index.name = original_id_col_name

    # Finalize edges GDF by mapping indices and ensuring schema
    final_edges_gdf = _finalize_edges(
        edges_intermediate_gdf, id_map_series, original_id_col_name, gdf.crs, edge_schema_cols,
    )

    # Ensure consistent CRS for final GDFs
    authoritative_crs = final_nodes_gdf.crs if final_nodes_gdf.crs is not None else gdf.crs

    if final_nodes_gdf.crs is None and authoritative_crs is not None:
        final_nodes_gdf.set_crs(authoritative_crs, inplace=True, allow_override=True)
    if final_edges_gdf.crs is None and authoritative_crs is not None:
        final_edges_gdf.set_crs(authoritative_crs, inplace=True, allow_override=True)

    # Ensure geometry column is named "geometry"
    if "geometry" not in final_nodes_gdf.columns and final_nodes_gdf.geometry.name is not None:
        final_nodes_gdf = final_nodes_gdf.rename_geometry("geometry")
    if "geometry" not in final_edges_gdf.columns and final_edges_gdf.geometry.name is not None:
        final_edges_gdf = final_edges_gdf.rename_geometry("geometry")

    return final_nodes_gdf, final_edges_gdf


def _get_original_id_col_name(gdf: gpd.GeoDataFrame) -> str:
    """Determine a unique column name for storing original GDF index."""
    # Use existing index name if available, otherwise default to "original_id"
    original_id_col_name = gdf.index.name
    if original_id_col_name is None:
        original_id_col_name = "original_id"

    # Ensure uniqueness if the chosen name collides with an existing column
    i = 0
    base_name = original_id_col_name
    while original_id_col_name in gdf.columns:
        original_id_col_name = f"{base_name}_{i}"
        i += 1
    return original_id_col_name


def _create_empty_dual_graph_gdfs(
    crs: str | int | None,
    original_id_col_name: str,  # Name of the ID column from the original GDF
    node_schema_cols: list[str],  # Full schema for node GDF columns
    edge_schema_cols: list[str],  # Schema for edge GDF columns (typically just ["geometry"])
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Helper to create empty nodes and edges GDFs for dual_graph, compatible with its calling signature."""
    # Prepare columns for empty_nodes_gdf
    # Ensure "geometry" is present for the geometry column.
    final_node_cols = list(node_schema_cols)  # Start with a copy
    if "geometry" not in final_node_cols:
        final_node_cols.append("geometry")
    # Ensure uniqueness, keeping "geometry" (usually added last if not present)
    # If node_schema_cols was empty, final_node_cols is now ["geometry"]
    final_node_cols = list(dict.fromkeys(final_node_cols))

    empty_nodes_gdf = gpd.GeoDataFrame(
        columns=final_node_cols,
        geometry="geometry",  # The column named "geometry" will be the geometry column
        crs=crs,
    )

    # Prepare columns for empty_edges_gdf
    from_col = f"from_{original_id_col_name}"
    to_col = f"to_{original_id_col_name}"

    # Start with from/to columns, then add others from edge_schema_cols
    final_edge_cols = [from_col, to_col]
    for col in edge_schema_cols:
        if col not in final_edge_cols:  # Avoid duplicating from/to if they were in edge_schema_cols
            final_edge_cols.append(col)

    # Ensure "geometry" is present for the geometry column.
    if "geometry" not in final_edge_cols:
        final_edge_cols.append("geometry")
    # Ensure uniqueness
    # If edge_schema_cols was ["geometry"], final_edge_cols is [from, to, "geometry"]
    final_edge_cols = list(dict.fromkeys(final_edge_cols))

    empty_edges_gdf = gpd.GeoDataFrame(
        columns=final_edge_cols,
        geometry="geometry",  # The column named "geometry" will be the geometry column
        crs=crs,
    )
    return empty_nodes_gdf, empty_edges_gdf


def _perform_graph_conversion(
    gdf_for_momepy: gpd.GeoDataFrame,
    empty_nodes_gdf: gpd.GeoDataFrame, # For returning on error
    empty_edges_gdf: gpd.GeoDataFrame, # For returning on error
) -> tuple[nx.Graph | None, gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    """Call momepy.gdf_to_nx and city2graph.nx_to_gdf, handling primary errors."""
    try:
        # Convert GeoDataFrame to NetworkX graph using momepy's dual approach
        graph_nx = momepy.gdf_to_nx(gdf_for_momepy, approach="dual", preserve_index=True)
    except Exception: # Broad exception due to external library
        logger.exception("Error during momepy.gdf_to_nx")
        # Return None for graph and the provided empty GDFs on error
        return None, empty_nodes_gdf, empty_edges_gdf

    # Check if momepy produced an empty graph from non-empty input
    if graph_nx.number_of_nodes() == 0 and not gdf_for_momepy.empty:
        logger.warning("momepy.gdf_to_nx resulted in an empty graph from non-empty input.")
        # Return None for graph and the provided empty GDFs
        return None, empty_nodes_gdf, empty_edges_gdf

    # Convert NetworkX graph back to GeoDataFrames for nodes and edges
    nodes_intermediate_gdf, edges_intermediate_gdf = nx_to_gdf(graph_nx, nodes=True, edges=True)

    # Check if nx_to_gdf resulted in empty nodes GDF from a non-empty graph
    if nodes_intermediate_gdf.empty and graph_nx.number_of_nodes() > 0:
        logger.warning("nx_to_gdf resulted in empty nodes_gdf from a non-empty graph.")
        # Return the graph (for potential partial use) and the provided empty GDFs
        return graph_nx, empty_nodes_gdf, empty_edges_gdf

    return graph_nx, nodes_intermediate_gdf, edges_intermediate_gdf


def _finalize_edges(
    edges_intermediate_gdf: gpd.GeoDataFrame,
    id_map_series: pd.Series,
    original_id_col_name: str,
    fallback_crs: Any,
    edge_schema_cols: list[str],
) -> gpd.GeoDataFrame:
    """Map edge indices to original IDs and finalize the edges GeoDataFrame."""
    # Process if intermediate edges GDF is not empty
    if not edges_intermediate_gdf.empty:
        # Map 'u' and 'v' components of the edge index using the id_map_series
        u_original = edges_intermediate_gdf.index.get_level_values(0).map(id_map_series)
        v_original = edges_intermediate_gdf.index.get_level_values(1).map(id_map_series)

        # Create a mask for valid edges (where both u and v were successfully mapped)
        valid_edges_mask = ~u_original.isna() & ~v_original.isna()
        # If some edges are invalid, log a warning and filter them out
        if not valid_edges_mask.all():
            num_invalid = len(valid_edges_mask) - valid_edges_mask.sum()
            logger.warning(
                "Removing %s edges due to ID mapping issues. "
                "This can happen if graph nodes were removed or altered unexpectedly.",
                num_invalid,
            )
            u_original = u_original[valid_edges_mask]
            v_original = v_original[valid_edges_mask]
            edges_intermediate_gdf = edges_intermediate_gdf[valid_edges_mask]

        # Create a copy of the (potentially filtered) intermediate edges GDF
        final_edges_gdf = edges_intermediate_gdf.copy()
        # If there are any valid edges left
        if not final_edges_gdf.empty:
            # Set the MultiIndex of the final edges GDF using the mapped original IDs
            final_edges_gdf.index = pd.MultiIndex.from_arrays(
                [u_original, v_original],
                names=(f"{original_id_col_name}_u", f"{original_id_col_name}_v"),
            )
            return final_edges_gdf
        # If all edges became invalid, return an empty edges GDF with correct schema
        return _create_empty_dual_graph_gdfs(
            fallback_crs, original_id_col_name, [], edge_schema_cols,
        )[1]
    # If intermediate edges GDF was already empty, return an empty edges GDF
    return _create_empty_dual_graph_gdfs(
        fallback_crs, original_id_col_name, [], edge_schema_cols,
    )[1]


# ============================================================================
# GRAPH FILTERING FUNCTIONS
# ============================================================================


def filter_graph_by_distance(graph: gpd.GeoDataFrame | nx.Graph,
                             center_point: Point | gpd.GeoSeries | gpd.GeoDataFrame,
                             distance: float,
                             edge_attr: str = "length",
                             node_id_col: str | None = None) -> gpd.GeoDataFrame | nx.Graph:
    """
    Extract a filtered graph containing only elements within a given shortest-path distance.

    Filters graph elements based on distance from specified center point(s).

    Parameters
    ----------
    graph : Union[gpd.GeoDataFrame, nx.Graph]
        Input graph data as either a GeoDataFrame of edges or a NetworkX graph.
    center_point : Union[Point, gpd.GeoSeries, gpd.GeoDataFrame]
        Center point(s) for distance calculation.
        Can be a single Point, GeoSeries of points, or GeoDataFrame with point geometries.
    distance : float
        Maximum shortest-path distance from any center node.
    edge_attr : str, default="length"
        Edge attribute to use as weight for distance calculation.
    node_id_col : Optional[str], default=None
        Column name in nodes GeoDataFrame to use as node identifier.
        If None, will use auto-generated node IDs.

    Returns
    -------
    Union[gpd.GeoDataFrame, nx.Graph]
        Filtered graph containing only elements within distance of any center point.
        Returns the same type as the input (either GeoDataFrame or NetworkX graph).
    """
    is_graph_input = isinstance(graph, nx.Graph)

    # Convert input to graph and preserve CRS - validation is done internally
    if is_graph_input:
        graph = _validate_nx(graph)
        original_crs = graph.graph.get("crs")
    else:
        # gdf_to_nx will validate the GeoDataFrame internally
        graph = gdf_to_nx(edges=graph)
        original_crs = graph.graph["crs"]

    # Extract node positions or return empty result if not available
    pos_dict = _extract_node_positions(graph)
    if not pos_dict:
        return _create_empty_result(is_graph_input, original_crs)

    # Create nodes GeoDataFrame
    node_id_name = node_id_col or "node_id"
    nodes_gdf = _create_nodes_gdf(pos_dict, node_id_name, original_crs)

    # Normalize center points
    center_points = _normalize_center_points(center_point)

    # Compute nodes within distance
    nodes_within_distance = _compute_nodes_within_distance(
        graph, center_points, nodes_gdf, distance, edge_attr, node_id_name,
    )

    # Return subgraph or empty result
    if not nodes_within_distance:
        return _create_empty_result(is_graph_input, original_crs)

    subgraph = graph.subgraph(nodes_within_distance)

    if is_graph_input:
        return subgraph

    # Convert back to GeoDataFrame
    filtered_gdf = nx_to_gdf(subgraph, nodes=False, edges=True)
    if not isinstance(filtered_gdf.geometry, gpd.GeoSeries):
        filtered_gdf = gpd.GeoDataFrame(
            filtered_gdf, geometry="geometry", crs=original_crs,
        )

    return filtered_gdf


def create_isochrone(
    graph: gpd.GeoDataFrame | nx.Graph,
    center_point: Point | gpd.GeoSeries | gpd.GeoDataFrame,
    distance: float,
    edge_attr: str = "length",
) -> gpd.GeoDataFrame:
    """
    Generate isochrone polygon(s) as convex hull of reachable areas within distance.

    Parameters
    ----------
    graph : Union[gpd.GeoDataFrame, nx.Graph]
        Input graph edges or NetworkX graph.
    center_point : Union[Point, GeoSeries, GeoDataFrame]
        Center point(s) for distance calculation.
    distance : float
        Maximum shortest-path distance from center.
    edge_attr : str, default="length"
        Edge weight attribute for path length.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing a single isochrone Polygon or MultiPolygon.
    """
    # Filter graph by distance
    reachable = filter_graph_by_distance(graph, center_point, distance, edge_attr)
    # Convert to GeoDataFrame if NetworkX graph returned
    if isinstance(reachable, nx.Graph):
        reachable = nx_to_gdf(reachable, nodes=False, edges=True)
    # Empty result
    if reachable.empty:
        return gpd.GeoDataFrame(geometry=[], crs=getattr(reachable, "crs", None))
    # Create convex hull of reachable area
    union_geom = reachable.unary_union
    hull = union_geom.convex_hull
    return gpd.GeoDataFrame(geometry=[hull], crs=reachable.crs)


def _get_nearest_node(point: Point | gpd.GeoSeries,
                      nodes_gdf: gpd.GeoDataFrame,
                      node_id: str = "node_id") -> str | int:
    """Find the nearest node in a GeoDataFrame."""
    if isinstance(point, gpd.GeoSeries):
        point = point.iloc[0]
    nearest_idx = nodes_gdf.distance(point).idxmin()
    return nodes_gdf.loc[nearest_idx, node_id]


def _extract_node_positions(graph: nx.Graph) -> dict[str | int, tuple[float, float]]:
    """Extract node positions from a NetworkX graph."""
    pos_dict = nx.get_node_attributes(graph, "pos")

    if pos_dict:
        return pos_dict

    # Handle case where graph has no pos attributes
    nodes_list = list(graph.nodes())
    if not nodes_list:
        return {}

    # Check if nodes are coordinate tuples (common when using momepy.gdf_to_nx)
    first_node = nodes_list[0]
    if isinstance(first_node, tuple) and len(first_node) == 2:
        return {node_id: node_id for node_id in graph.nodes()}

    # Check if nodes have x,y attributes
    node_attrs = dict(graph.nodes(data=True))
    if node_attrs and all("x" in attrs and "y" in attrs for attrs in node_attrs.values()):
        return {node_id: (attrs["x"], attrs["y"]) for node_id, attrs in node_attrs.items()}

    return {}


def _create_nodes_gdf(pos_dict: dict[str | int, tuple[float, float]],
                      node_id_col: str,
                      crs: str | int | None) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame from node positions."""
    if not pos_dict:
        return gpd.GeoDataFrame({node_id_col: [], "geometry": []}, crs=crs)

    node_ids, coordinates = zip(*pos_dict.items(), strict=False)
    geometries = [Point(coord) for coord in coordinates]

    return gpd.GeoDataFrame(
        {node_id_col: node_ids, "geometry": geometries},
        crs=crs,
    )


def _compute_nodes_within_distance(graph: nx.Graph,
                                   center_points: list[Point] | gpd.GeoSeries,
                                   nodes_gdf: gpd.GeoDataFrame,
                                   distance: float,
                                   edge_attr: str,
                                   node_id_name: str) -> set[str | int]:
    """Compute all nodes within distance from any center point."""
    # Convert center points to list for consistent processing
    center_points_list = (center_points.tolist()
                         if hasattr(center_points, "tolist")
                         else list(center_points))

    # Vectorized computation using list comprehensions
    all_valid_nodes = []
    for point in center_points_list:
        try:
            nearest_node = _get_nearest_node(point, nodes_gdf, node_id=node_id_name)
            distance_dict = nx.shortest_path_length(graph, nearest_node, weight=edge_attr)

            # Extract nodes within distance using list comprehension
            valid_nodes = [node_id for node_id, dist in distance_dict.items() if dist < distance]
            all_valid_nodes.extend(valid_nodes)
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            logger.warning("Could not compute paths from a center point: %s", e, stacklevel=2)

    return set(all_valid_nodes)


def _normalize_center_points(center_point: Point | gpd.GeoSeries | gpd.GeoDataFrame,
                             ) -> list[Point] | gpd.GeoSeries:
    """Normalize center point input to a consistent format."""
    if isinstance(center_point, gpd.GeoDataFrame):
        return center_point.geometry
    if isinstance(center_point, gpd.GeoSeries):
        return center_point
    return [center_point]


def _create_empty_result(is_graph_input: bool, original_crs: str | int | None) -> gpd.GeoDataFrame | nx.Graph:
    """Create an empty result in the appropriate format."""
    if is_graph_input:
        return nx.Graph()
    return gpd.GeoDataFrame(geometry=[], crs=original_crs)


# ============================================================================
# GEODATAFRAME TO NETWORKX CONVERSION FUNCTIONS
# ============================================================================


def gdf_to_nx(nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None = None,
              edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
              node_id_col: str | dict[str, str] | None = None,
              edge_id_col: str | dict[tuple[str, str, str], str] | None = None,
              keep_geom: bool = True) -> nx.Graph:
    """
    Convert GeoDataFrames of nodes and edges to a NetworkX graph.

    Parameters
    ----------
    nodes : GeoDataFrame or dict[str, GeoDataFrame], optional
        Point or Polygon geometries for graph nodes; node attributes preserved.
        For heterogeneous graphs, provide a dict mapping node types to GeoDataFrames.
    edges : GeoDataFrame or dict[tuple[str, str, str], GeoDataFrame], optional
        LineString geometries for graph edges; edge attributes preserved.
        For heterogeneous graphs, provide a dict mapping edge types to GeoDataFrames.
    node_id_col : str or dict[str, str], optional
        Column name to use for node identifiers; if None, uses index.
        For heterogeneous graphs, provide a dict mapping node types to column names.
    edge_id_col : str or dict[tuple[str, str, str], str], optional
        Column name to use for edge identifiers; if None, uses index.
        For heterogeneous graphs, provide a dict mapping edge types to column names.
    keep_geom : bool, default=True
        If True, include original geometry columns on edges.

    Returns
    -------
    networkx.Graph
        Graph with 'crs', 'node_geom_cols', 'edge_geom_cols', node 'pos', and attributes.
        For heterogeneous graphs, includes type information and metadata.
    """
    # Handle heterogeneous graphs (dictionaries of GeoDataFrames)
    if isinstance(nodes, dict) or isinstance(edges, dict):
        return _gdf_to_nx_heterogeneous(nodes, edges, node_id_col, edge_id_col, keep_geom)

    # Handle homogeneous graphs (single GeoDataFrames) - validation done internally
    return _gdf_to_nx_homogeneous(nodes, edges, node_id_col, edge_id_col, keep_geom)


def nx_to_gdf(
    G: nx.Graph,
    nodes: bool = True,
    edges: bool = True,
) -> (gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] |
      tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]):
    """
    Convert a NetworkX graph to a GeoDataFrame for nodes or edges.

    Parameters
    ----------
    G : networkx.Graph
        Graph with 'crs', 'node_geom_cols', and 'edge_geom_cols' in G.graph.
    nodes : bool, default=True
        If True, return node GeoDataFrame.
    edges : bool, default=True
        If True, return edge GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame or tuple
        If both nodes and edges requested: tuple of (nodes_gdf, edges_gdf)
        If only nodes requested: nodes GeoDataFrame
        If only edges requested: edges GeoDataFrame
        For heterogeneous graphs: tuple of (nodes_dict, edges_dict)

    Raises
    ------
    ValueError
        If required 'pos' or geometry attributes are missing, or if neither flag is set.
    """
    # Check if this is a heterogeneous graph
    is_hetero = G.graph.get("is_hetero", False)

    if is_hetero:
        # Handle heterogeneous graph reconstruction
        return _reconstruct_heterogeneous_gdfs(G, nodes=nodes, edges=edges)

    # Both requested: return node and edge frames via separate calls
    if nodes and edges:
        nodes_gdf = nx_to_gdf(G, nodes=True, edges=False)
        edges_gdf = nx_to_gdf(G, nodes=False, edges=True)
        return nodes_gdf, edges_gdf

    # Require at least one of nodes or edges
    if not nodes and not edges:
        msg = "At least one of 'nodes' or 'edges' must be True"
        raise ValueError(msg)

    # Fallback for nodes-only graphs: treat node IDs as coords
    if nodes and not nx.get_node_attributes(G, "pos"):
        node_list = list(G.nodes())
        # Check if node IDs are coordinate tuples
        if node_list and isinstance(node_list[0], tuple) and len(node_list[0]) == 2:
            try:
                # Convert coordinate tuple node IDs to Point geometries
                coords = {node: node for node in node_list}
                nx.set_node_attributes(G, coords, "pos")
            except (ValueError, TypeError):
                msg = "Graph nodes don't have 'pos' attributes and can't be interpreted as coordinates"
                raise ValueError(msg) from None

    # Validate the graph structure
    G = _validate_nx(G, nodes=nodes)

    # Create and return appropriate GeoDataFrame
    if nodes:
        return _create_nodes_gdf_from_graph(G)
    return _create_edges_gdf_from_graph(G)


# ============================================================================
# GRAPH VALIDATION HELPER FUNCTIONS
# ============================================================================


def _validate_gdf(nodes: gpd.GeoDataFrame | None,
                  edges: gpd.GeoDataFrame | None,
                  strict: bool = True,
                  allow_empty: bool = True,
                  check_crs: bool = True) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]:
    """
    Comprehensive validation and cleaning of node and edge GeoDataFrames.

    This function centralizes all data quality checks including:
    - Geometry type validation
    - CRS consistency checks
    - Empty/null geometry handling
    - Geographic CRS restrictions
    - Data completeness validation

    Parameters
    ----------
    nodes : gpd.GeoDataFrame, optional
        Node GeoDataFrame to validate
    edges : gpd.GeoDataFrame, optional
        Edge GeoDataFrame to validate
    strict : bool, default=True
        If True, raises errors for invalid data. If False, logs warnings and cleans data.
    allow_empty : bool, default=True
        If True, allows empty GeoDataFrames. If False, raises error for empty data.
    check_crs : bool, default=True
        If True, validates CRS consistency between nodes and edges.

    Returns
    -------
    tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]
        Validated and cleaned nodes and edges GeoDataFrames

    Raises
    ------
    TypeError
        If inputs are not GeoDataFrames when provided
    ValueError
        If data fails validation checks and strict=True
    """
    # If both are None, nothing to validate
    if nodes is None and edges is None:
        return nodes, edges

    # Type validation
    if nodes is not None and not isinstance(nodes, gpd.GeoDataFrame):
        msg = "Nodes input must be a GeoDataFrame"
        raise TypeError(msg)
    if edges is not None and not isinstance(edges, gpd.GeoDataFrame):
        msg = "Edges input must be a GeoDataFrame"
        raise TypeError(msg)

    # Validate and clean nodes GeoDataFrame
    if nodes is not None:
        nodes = _validate_nodes_gdf(nodes, strict=strict, allow_empty=allow_empty)

    # Validate and clean edges GeoDataFrame
    if edges is not None:
        edges = _validate_edges_gdf(edges, strict=strict, allow_empty=allow_empty)

    # CRS validation
    if check_crs and nodes is not None and edges is not None and not nodes.empty and not edges.empty:
        if nodes.crs != edges.crs:
            msg = f"CRS mismatch: nodes CRS ({nodes.crs}) != edges CRS ({edges.crs})"
            if strict:
                raise ValueError(msg)
            logger.warning("Validation: %s", msg)

        # Check for missing CRS only if both have data
        if nodes.crs is None or edges.crs is None:
            msg = "Both nodes and edges must have a defined CRS"
            if strict:
                raise ValueError(msg)
            logger.warning("Validation: %s", msg)

    return nodes, edges


def _validate_nodes_gdf(nodes: gpd.GeoDataFrame,
                        strict: bool = True,
                        allow_empty: bool = True) -> gpd.GeoDataFrame:
    """Validate and clean nodes GeoDataFrame for geometry quality."""
    if nodes.empty:
        if not allow_empty:
            msg = "Nodes GeoDataFrame is empty"
            if strict:
                raise ValueError(msg)
        logger.warning("Validation: Nodes GeoDataFrame is empty")
        return nodes

    original_count = len(nodes)

    # Check for geographic CRS (disallowed for most operations)
    if hasattr(nodes, "crs") and nodes.crs and str(nodes.crs) == "EPSG:4326":
        msg = "Nodes are in geographic CRS (EPSG:4326). Consider reprojecting to a projected CRS."
        if strict:
            raise ValueError(msg)
        logger.warning("Validation: %s", msg)

    # Create composite mask for all invalid conditions
    null_mask = nodes.geometry.isna()
    invalid_mask = ~nodes.geometry.is_valid
    empty_mask = nodes.geometry.is_empty

    # Check for centroid computation issues vectorized
    try:
        centroids = nodes.geometry.centroid
        centroid_mask = centroids.isna()
    except (AttributeError, ValueError, TypeError):
        # Fallback: assume all centroids are valid initially
        centroid_mask = pd.Series(data=False, index=nodes.index)

    # Check for valid geometry types for nodes (Point, Polygon, MultiPolygon)
    valid_node_types = nodes.geometry.apply(
        lambda g: isinstance(g, (Point, shapely.geometry.Polygon, shapely.geometry.MultiPolygon))
        if g is not None else False,
    )

    # Combine all masks - keep only valid geometries
    valid_mask = ~(null_mask | invalid_mask | empty_mask | centroid_mask) & valid_node_types
    nodes_clean = nodes[valid_mask]

    # Log removal statistics
    if null_mask.any():
        logger.warning("Validation: Removing %d nodes with null geometries", null_mask.sum())
    if invalid_mask.any():
        logger.warning("Validation: Removing %d nodes with invalid geometries", invalid_mask.sum())
    if empty_mask.any():
        logger.warning("Validation: Removing %d nodes with empty geometries", empty_mask.sum())
    if centroid_mask.any():
        logger.warning("Validation: Removing %d nodes with invalid centroids", centroid_mask.sum())
    if not valid_node_types.all():
        logger.warning(
            "Validation: Removing %d nodes with invalid geometry types "
            "(not Point, Polygon, or MultiPolygon)", (~valid_node_types).sum(),
        )

    removed_count = original_count - len(nodes_clean)
    if removed_count > 0:
        logger.warning("Validation: Removed %d invalid nodes out of %d total", removed_count, original_count)

    return nodes_clean


def _validate_edges_gdf(edges: gpd.GeoDataFrame,
                        strict: bool = True,
                        allow_empty: bool = True) -> gpd.GeoDataFrame:
    """Validate and clean edges GeoDataFrame for geometry quality."""
    if edges.empty:
        if not allow_empty:
            msg = "Edges GeoDataFrame is empty"
            if strict:
                raise ValueError(msg)
        logger.warning("Validation: Edges GeoDataFrame is empty")
        return edges

    original_count = len(edges)

    # Check for geographic CRS (disallowed for most operations)
    if hasattr(edges, "crs") and edges.crs and str(edges.crs) == "EPSG:4326":
        msg = "Edges are in geographic CRS (EPSG:4326). Consider reprojecting to a projected CRS."
        if strict:
            raise ValueError(msg)
        logger.warning("Validation: %s", msg)

    # Create composite mask for all invalid conditions
    null_mask = edges.geometry.isna()
    invalid_mask = ~edges.geometry.is_valid
    empty_mask = edges.geometry.is_empty

    # Check for valid geometry types vectorized
    valid_geom_types = edges.geometry.apply(
        lambda g: isinstance(g, (LineString, shapely.geometry.MultiLineString))
        if g is not None else False,
    )

    # Combine all masks - keep only valid geometries
    valid_mask = ~(null_mask | invalid_mask | empty_mask) & valid_geom_types
    edges_clean = edges[valid_mask]

    # Log removal statistics
    if null_mask.any():
        logger.warning("Validation: Removing %d edges with null geometries", null_mask.sum())
    if invalid_mask.any():
        logger.warning("Validation: Removing %d edges with invalid geometries", invalid_mask.sum())
    if empty_mask.any():
        logger.warning("Validation: Removing %d edges with empty geometries", empty_mask.sum())
    if not valid_geom_types.all():
        logger.warning(
            "Validation: Removing %d edges with invalid geometry types "
            "(not LineString or MultiLineString)", (~valid_geom_types).sum(),
        )

    removed_count = original_count - len(edges_clean)
    if removed_count > 0:
        logger.warning("Validation: Removed %d invalid edges out of %d total", removed_count, original_count)

    return edges_clean


def _validate_nx(graph: nx.Graph,
                 nodes: bool = False,
                 strict: bool = True,
                 require_crs: bool = True,
                 require_positions: bool = False) -> nx.Graph:
    """
    Comprehensive validation and cleaning of NetworkX graphs.

    Centralizes all NetworkX graph quality checks including:
    - Graph structure validation
    - Node position validation
    - Edge connectivity validation
    - Metadata completeness checks
    - CRS validation

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph to validate
    nodes : bool, default=False
        If True, validates for node-focused operations
    strict : bool, default=True
        If True, raises errors for invalid data. If False, logs warnings and cleans data.
    require_crs : bool, default=True
        If True, requires CRS metadata in graph attributes
    require_positions : bool, default=False
        If True, requires node position attributes

    Returns
    -------
    nx.Graph
        Validated and cleaned NetworkX graph

    Raises
    ------
    ValueError
        If graph fails validation checks and strict=True
    """
    # Make a copy to avoid modifying the original graph
    clean_graph = graph.copy()

    # Check if graph is empty first
    if clean_graph.number_of_nodes() == 0:
        msg = "Graph has no nodes"
        if strict and not nodes:
            raise ValueError(msg)
        logger.warning("Validation: %s", msg)
        return clean_graph

    # Validate graph metadata
    if require_crs:
        crs = clean_graph.graph.get("crs")
        if crs is None:
            msg = "Missing CRS in graph attributes. Set 'crs' in graph.graph."
            if strict:
                raise ValueError(msg)
            logger.warning("Validation: %s", msg)

    # Validate and clean node positions
    clean_graph = _validate_node_positions(clean_graph, nodes, strict, require_positions)

    # Validate edges if needed
    if not nodes:
        clean_graph = _validate_graph_edges(clean_graph, strict)

    return clean_graph


def _validate_node_positions(graph: nx.Graph,
                             nodes: bool,
                             strict: bool = True,
                             require_positions: bool = False) -> nx.Graph:
    """Validate and clean node positions in the graph."""
    pos = nx.get_node_attributes(graph, "pos")

    if not pos:
        if nodes or require_positions:
            msg = "Missing 'pos' attribute for nodes"
            if strict:
                raise ValueError(msg)
            logger.warning("Validation: %s", msg)
        return graph

    # Vectorized position validation
    invalid_nodes = [node_id for node_id, position in pos.items()
                    if not _is_valid_position(position)]

    # Remove invalid nodes in batch
    if invalid_nodes:
        graph.remove_nodes_from(invalid_nodes)
        logger.warning("Validation: Removed %d nodes with invalid positions", len(invalid_nodes))

    return graph


def _is_valid_position(position: tuple | list) -> bool:
    """Check if a position is valid."""
    if not isinstance(position, (tuple, list)) or len(position) < 2:
        return False
    return all(isinstance(coord, (int, float)) for coord in position[:2])


def _validate_graph_edges(graph: nx.Graph, strict: bool = True) -> nx.Graph:
    """Validate edges in the graph."""
    if graph.number_of_edges() == 0:
        logger.warning("Validation: Graph has no edges")
        return graph

    # Check for geometry or position availability
    has_geometry = any(attrs.get("geometry") is not None
                      for _, _, attrs in graph.edges(data=True))
    pos = nx.get_node_attributes(graph, "pos")

    if not has_geometry and not pos:
        msg = "Missing edge geometry and node positions"
        if strict:
            raise ValueError(msg)
        logger.warning("Validation: %s", msg)

    # Find edges that reference non-existent nodes
    invalid_edges = [(u, v) for u, v in graph.edges()
                     if not graph.has_node(u) or not graph.has_node(v)]

    # Remove invalid edges in batch
    if invalid_edges:
        graph.remove_edges_from(invalid_edges)
        logger.warning("Validation: Removed %d edges referencing non-existent nodes", len(invalid_edges))

    return graph


# ============================================================================
# GRAPH CONVERSION HELPER FUNCTIONS
# ============================================================================


def _gdf_to_nx_homogeneous(
    nodes: gpd.GeoDataFrame | None,
    edges: gpd.GeoDataFrame | None,
    node_id_col: str | None,
    edge_id_col: str | None,
    keep_geom: bool,
) -> nx.Graph:
    """Convert homogeneous GeoDataFrames to NetworkX graph (original implementation)."""
    # Validate GeoDataFrames first
    nodes, edges = _validate_gdf(nodes, edges)

    if edges is None:
        msg = "Must provide edges GeoDataFrame"
        raise ValueError(msg)

    # Initialize graph with metadata
    graph = nx.Graph()

    # Set metadata for homogeneous graph
    _set_graph_metadata(graph,
                        nodes,
                        edges,
                        is_hetero=False,
                        node_id_col=node_id_col,
                        edge_id_col=edge_id_col)

    # Add nodes with original index information
    if nodes is not None:
        # Vectorized node processing - avoid iterrows()
        centroids = nodes.geometry.centroid
        node_data = nodes.drop(columns="geometry")

        # Create node attributes dictionary efficiently
        node_attrs_dict = {}
        for idx, orig_idx in enumerate(nodes.index):
            attrs = node_data.iloc[idx].to_dict()
            attrs["_original_index"] = orig_idx
            attrs["pos"] = (centroids.iloc[idx].x, centroids.iloc[idx].y)
            node_attrs_dict[idx] = attrs

        # Add all nodes with attributes at once
        graph.add_nodes_from(node_attrs_dict.items())

    # Process edges efficiently
    _process_homogeneous_edges(graph, edges, nodes, keep_geom)

    return graph


def _gdf_to_nx_heterogeneous(
    nodes_dict: dict[str, gpd.GeoDataFrame] | None,
    edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    edge_id_cols: dict[tuple[str, str, str], str] | str | None,
    keep_geom: bool,
) -> nx.Graph:
    """Convert heterogeneous GeoDataFrames to NetworkX graph."""
    # Initialize graph with heterogeneous metadata
    graph = nx.Graph()

    # Set metadata for heterogeneous graph
    _set_graph_metadata(graph, nodes_dict, edges_dict, is_hetero=True)

    # Process nodes and edges
    _process_hetero_nodes(graph, nodes_dict)
    _process_hetero_edges(graph, edges_dict, edge_id_cols, keep_geom, nodes_dict)

    return graph


def _create_nodes_gdf_from_graph(graph: nx.Graph) -> gpd.GeoDataFrame:
    """Create nodes GeoDataFrame from NetworkX graph."""
    pos = nx.get_node_attributes(graph, "pos")
    crs = graph.graph.get("crs")
    node_geom_cols = graph.graph.get("node_geom_cols", [])

    if not graph.nodes():
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    # Extract nodes with original index information - vectorized approach
    node_data = dict(graph.nodes(data=True))

    # Extract original indices and records in vectorized manner
    original_indices = [attrs.get("_original_index", nid) for nid, attrs in node_data.items()]

    # Build records without loops
    records = []
    for nid, attrs in node_data.items():
        record = {k: v for k, v in attrs.items() if k not in ["pos", "_original_index"]}
        pos_val = pos.get(nid, (0, 0))
        record["geometry"] = Point(*pos_val) if isinstance(pos_val, (tuple, list)) else Point(pos_val, 0)
        records.append(record)

    gdf = gpd.GeoDataFrame(records, index=original_indices, crs=crs)

    # Convert geometry columns to GeoSeries
    for col in node_geom_cols:
        if col in gdf.columns:
            gdf[col] = gpd.GeoSeries(gdf[col], crs=crs)

    # Restore original index names
    node_index_names = graph.graph.get("node_index_names")
    if node_index_names and isinstance(node_index_names, list) and len(node_index_names) == 1:
        gdf.index.name = node_index_names[0]
    elif node_index_names and isinstance(node_index_names, list) and isinstance(gdf.index, pd.MultiIndex):
        gdf.index.names = node_index_names[:gdf.index.nlevels]

    return gdf


def _create_edges_gdf_from_graph(graph: nx.Graph) -> gpd.GeoDataFrame:
    """Create edges GeoDataFrame from NetworkX graph."""
    pos = nx.get_node_attributes(graph, "pos")
    crs = graph.graph.get("crs")
    edge_geom_cols = graph.graph.get("edge_geom_cols", [])

    if graph.number_of_edges() == 0:
        return gpd.GeoDataFrame(columns=["geometry"], crs=crs)

    # Extract all edge data at once
    edge_data = list(graph.edges(data=True))

    # Vectorized extraction of indices and records
    original_edge_indices = [
        attrs.get("_original_edge_index", (u, v))
        for u, v, attrs in edge_data
    ]

    # Vectorized record creation
    records = [
        {
            **{k: v for k, v in attrs.items() if k not in ["_original_edge_index"]},
            "geometry": (
                attrs.get("geometry")
                if attrs.get("geometry") is not None
                else LineString([pos[u], pos[v]])
            ),
        }
        for u, v, attrs in edge_data
    ]

    # Create GeoDataFrame with original indices
    if original_edge_indices:
        # Check if we have tuple indices (MultiIndex case)
        if original_edge_indices and isinstance(original_edge_indices[0], tuple):
            edge_index = pd.MultiIndex.from_tuples(original_edge_indices)
        else:
            edge_index = original_edge_indices

        gdf = gpd.GeoDataFrame(records, index=edge_index, crs=crs)
    else:
        gdf = gpd.GeoDataFrame(records, crs=crs)

    # Convert geometry columns to GeoSeries
    for col in edge_geom_cols:
        if col in gdf.columns:
            gdf[col] = gpd.GeoSeries(gdf[col], crs=crs)

    # Restore original edge index names
    edge_index_names = graph.graph.get("edge_index_names")
    if edge_index_names and hasattr(gdf.index, "names"):
        if isinstance(gdf.index, pd.MultiIndex):
            # For MultiIndex, set all available names
            gdf.index.names = edge_index_names[:gdf.index.nlevels]
        elif len(edge_index_names) >= 1:
            # For regular Index, set single name
            gdf.index.name = edge_index_names[0]

    return gdf





def _set_graph_metadata(
    graph: nx.Graph,
    nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None,
    edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    is_hetero: bool = False,
    node_id_col: str | None = None,
    edge_id_col: str | None = None,
) -> None:
    """Set metadata for both homogeneous and heterogeneous graphs."""
    graph.graph["is_hetero"] = is_hetero

    if is_hetero:
        # Heterogeneous graph metadata
        if isinstance(nodes, dict):
            graph.graph["node_types"] = list(nodes.keys())
        if isinstance(edges, dict):
            graph.graph["edge_types"] = list(edges.keys())

        # Get CRS from first available GeoDataFrame
        crs = None
        if isinstance(nodes, dict) and nodes:
            crs = next(iter(nodes.values())).crs
        elif isinstance(edges, dict) and edges:
            crs = next(iter(edges.values())).crs
        graph.graph["crs"] = crs
    else:
        # Homogeneous graph metadata
        crs = nodes.crs if nodes is not None else edges.crs if edges is not None else None
        graph.graph.update({
            "crs": crs,
            "node_index_col": node_id_col,
            "edge_index_col": edge_id_col,
        })

        # Store original index information for reconstruction
        if nodes is not None:
            graph.graph["node_geom_cols"] = list(nodes.select_dtypes(include=["geometry"]).columns)
            graph.graph["node_index_names"] = nodes.index.names if hasattr(nodes.index, "names") else None

        if edges is not None:
            graph.graph["edge_geom_cols"] = list(edges.select_dtypes(include=["geometry"]).columns)
            graph.graph["edge_index_names"] = edges.index.names if hasattr(edges.index, "names") else None


def _process_homogeneous_edges(
    graph: nx.Graph,
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame | None,
    keep_geom: bool,
) -> None:
    """Process edges for homogeneous graph efficiently."""
    if nodes is not None:
        # Create coordinate to node ID mapping for fast lookup
        coord_to_node = {node_data["pos"]: node_id
                        for node_id, node_data in graph.nodes(data=True)}

        # Vectorized edge coordinate extraction
        start_coords = edges.geometry.apply(lambda g: g.coords[0])
        end_coords = edges.geometry.apply(lambda g: g.coords[-1])

        # Map coordinates to node IDs vectorized
        u_nodes = start_coords.map(coord_to_node)
        v_nodes = end_coords.map(coord_to_node)

        # Filter out edges where nodes weren't found
        valid_mask = u_nodes.notna() & v_nodes.notna()
        valid_edges = edges[valid_mask]
        valid_u = u_nodes[valid_mask]
        valid_v = v_nodes[valid_mask]

        # Prepare edge attributes
        if keep_geom:
            edge_attrs_data = valid_edges.to_dict("records")
        else:
            edge_attrs_data = valid_edges.drop(columns=["geometry"]).to_dict("records")

        # Add original edge indices vectorized
        for i, orig_idx in enumerate(valid_edges.index):
            edge_attrs_data[i]["_original_edge_index"] = orig_idx

        # Create edges to add
        edges_to_add = [(valid_u.iloc[i], valid_v.iloc[i], edge_attrs_data[i])
                       for i in range(len(valid_edges))]

        # Add all edges at once
        graph.add_edges_from(edges_to_add)
    else:
        # Use coordinate tuples as node IDs - vectorized approach
        start_coords = edges.geometry.apply(lambda g: g.coords[0])
        end_coords = edges.geometry.apply(lambda g: g.coords[-1])

        # Get unique nodes
        all_coords = pd.concat([start_coords, end_coords]).unique()
        nodes_to_add = {coord: {"pos": coord} for coord in all_coords}

        # Prepare edge attributes vectorized
        if keep_geom:
            edge_attrs_data = edges.to_dict("records") # This is a list of dicts
        else:
            edge_attrs_data = edges.drop(columns=["geometry"]).to_dict("records")

        # Add original edge indices vectorized
        for i, orig_idx in enumerate(edges.index):
            edge_attrs_data[i]["_original_edge_index"] = orig_idx
            # === Add length calculation here ===
            if "length" not in edge_attrs_data[i]:
                geom = edges.geometry.iloc[i]
                if geom: # Check if geom is not None
                    edge_attrs_data[i]["length"] = geom.length
            # === End of addition ===

        # Create edges to add
        # Edges are ( (x1,y1), (x2,y2), attrs )
        edges_to_add = [(start_coords.iloc[i], end_coords.iloc[i], edge_attrs_data[i])
                       for i in range(len(edges))]

        # Add all nodes and edges at once
        graph.add_nodes_from(nodes_to_add.items())
        graph.add_edges_from(edges_to_add)



# ============================================================================
# HETEROGENEOUS GRAPH HELPER FUNCTIONS
# ============================================================================


def _process_hetero_nodes(
    graph: nx.Graph,
    nodes_dict: dict[str, gpd.GeoDataFrame] | None,
) -> dict[str, int]:
    """Process nodes for heterogeneous graph and return offset mapping."""
    node_offset = {}
    current_offset = 0

    if not nodes_dict:
        return node_offset

    for node_type, node_gdf_orig in nodes_dict.items():
        # Validate and clean this node type
        node_gdf_validated, _ = _validate_gdf(node_gdf_orig, None)
        if node_gdf_validated is None or node_gdf_validated.empty:
            continue

        # Store offset for this node type
        node_offset[node_type] = current_offset

        # Add nodes with type information
        _add_hetero_nodes_to_graph(graph, node_gdf_validated, node_type, current_offset)
        current_offset += len(node_gdf_validated)

    return node_offset


def _add_hetero_nodes_to_graph(
    graph: nx.Graph, node_gdf: gpd.GeoDataFrame, node_type: str, offset: int,
) -> None:
    """Add heterogeneous nodes to graph."""
    # Vectorized approach to avoid iterrows()
    centroids = node_gdf.geometry.centroid
    node_data = node_gdf.drop(columns="geometry")

    # Batch process all nodes
    nodes_to_add = []
    for idx, orig_idx in enumerate(node_gdf.index):
        node_id = offset + idx

        # Store all attributes including type and original index
        node_attrs = node_data.iloc[idx].to_dict()
        node_attrs["node_type"] = node_type
        node_attrs["_original_index"] = orig_idx
        node_attrs["pos"] = (centroids.iloc[idx].x, centroids.iloc[idx].y)

        nodes_to_add.append((node_id, node_attrs))

    # Add all nodes at once
    graph.add_nodes_from(nodes_to_add)


def _process_hetero_edges(
    graph: nx.Graph,
    edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    keep_geom: bool,
    nodes_dict: dict[str, gpd.GeoDataFrame] | None,
) -> None:
    """Process edges for heterogeneous graph."""
    if not edges_dict:
        return

    for edge_type, edge_gdf_orig in edges_dict.items():
        # Validate and clean this edge type
        _, edge_gdf_validated = _validate_gdf(None, edge_gdf_orig)
        if edge_gdf_validated is None or edge_gdf_validated.empty:
            continue

        # Add edges with type information
        _add_hetero_edges_to_graph(
            graph, edge_gdf_validated, edge_type, keep_geom, nodes_dict,
        )


def _add_hetero_edges_to_graph(
    graph: nx.Graph,
    edge_gdf: gpd.GeoDataFrame,
    edge_type: tuple[str, str, str],
    keep_geom: bool,
) -> None:
    """Add heterogeneous edges to graph."""
    src_type, rel_type, dst_type = edge_type

    # Create reverse lookup for nodes by type and original index
    node_lookup = {}
    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get("node_type")
        orig_idx = node_data.get("_original_index")
        if node_type and orig_idx is not None:
            if node_type not in node_lookup:
                node_lookup[node_type] = {}
            node_lookup[node_type][orig_idx] = node_id

    # Process all edges at once
    edges_to_add = []
    for orig_idx, row in edge_gdf.iterrows():
        # Find source and target nodes by matching edge index to node indices
        if isinstance(orig_idx, tuple) and len(orig_idx) == 2:
            # MultiIndex case: (source_node_idx, target_node_idx)
            src_orig_idx, dst_orig_idx = orig_idx

            # Find corresponding nodes using lookup
            u = node_lookup.get(src_type, {}).get(src_orig_idx)
            v = node_lookup.get(dst_type, {}).get(dst_orig_idx)

            if u is not None and v is not None:
                # Store edge attributes
                edge_attrs = row.drop("geometry").to_dict() if not keep_geom else row.to_dict()
                edge_attrs["edge_type"] = rel_type
                edge_attrs["_original_edge_index"] = orig_idx

                edges_to_add.append((u, v, edge_attrs))

    # Add all edges at once
    graph.add_edges_from(edges_to_add)


def _reconstruct_heterogeneous_gdfs(
    G: nx.Graph, nodes: bool = True, edges: bool = True,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]:
    """Reconstruct heterogeneous GeoDataFrames from NetworkX graph."""
    node_types = G.graph.get("node_types", [])
    edge_types = G.graph.get("edge_types", [])
    crs = G.graph.get("crs")

    nodes_dict = {}
    edges_dict = {}

    if nodes:
        nodes_dict = _reconstruct_hetero_nodes(G, node_types, crs)

    if edges:
        edges_dict = _reconstruct_hetero_edges(G, edge_types, crs)

    return nodes_dict, edges_dict


def _reconstruct_hetero_nodes(
    G: nx.Graph, node_types: list[str], crs: str | None,
) -> dict[str, gpd.GeoDataFrame]:
    """Reconstruct heterogeneous nodes."""
    nodes_dict = {}

    for node_type in node_types:
        # Get all nodes of this type
        type_nodes = [(n, d) for n, d in G.nodes(data=True)
                      if d.get("node_type") == node_type]

        if not type_nodes:
            nodes_dict[node_type] = gpd.GeoDataFrame(columns=["geometry"], crs=crs)
            continue

        # Vectorized processing
        original_mappings = G.graph.get("_node_mappings", {})
        type_mapping = original_mappings.get(node_type, {})
        original_ids = (type_mapping.get("original_ids")
                       if isinstance(type_mapping, dict) else None)

        # Extract data vectorized
        node_ids, attrs_list = zip(*type_nodes, strict=False)

        # Create indices array
        indices = [
            original_ids[i] if original_ids and i < len(original_ids) else node_id
            for i, node_id in enumerate(node_ids)
        ]

        # Create records array
        records = [
            {
                **{k: v for k, v in attrs.items() if k not in ["pos", "node_type"]},
                "geometry": Point(attrs.get("pos")) if attrs.get("pos") else None,
            }
            for attrs in attrs_list
        ]

        nodes_dict[node_type] = gpd.GeoDataFrame(
            records, geometry="geometry", index=indices, crs=crs,
        )

        # Restore index names
        node_index_names = G.graph.get("_node_index_names", {})
        if node_type in node_index_names:
            index_names = node_index_names[node_type]
            if (isinstance(index_names, list) and len(index_names) == 1
                and index_names[0] is not None):
                nodes_dict[node_type].index.name = index_names[0]

    return nodes_dict


def _reconstruct_hetero_edges(
    G: nx.Graph, edge_types: list[tuple[str, str, str]], crs: str | None,
) -> dict[tuple[str, str, str], gpd.GeoDataFrame]:
    """Reconstruct heterogeneous edges."""
    edges_dict = {}

    for edge_type in edge_types:
        src_type, rel_type, dst_type = edge_type

        # Get all edges of this type
        type_edges = [(u, v, d) for u, v, d in G.edges(data=True)
                      if d.get("edge_type") == rel_type]

        if not type_edges:
            edges_dict[edge_type] = gpd.GeoDataFrame(columns=["geometry"], crs=crs)
            continue

        # Get original edge index values
        edge_index_values = G.graph.get("_edge_index_values", {})
        type_edge_values = edge_index_values.get(edge_type, [])

        # Vectorized processing
        indices = [
            type_edge_values[i] if type_edge_values and i < len(type_edge_values) else (u, v)
            for i, (u, v, _) in enumerate(type_edges)
        ]

        records = []
        for u, v, attrs in type_edges:
            # Create geometry from node positions or existing geometry
            geometry = attrs.get("geometry")
            if geometry is None:
                u_pos = G.nodes[u].get("pos")
                v_pos = G.nodes[v].get("pos")
                if u_pos and v_pos:
                    geometry = LineString([u_pos, v_pos])

            record = {k: v for k, v in attrs.items() if k not in ["edge_type"]}
            record["geometry"] = geometry
            records.append(record)

        edges_dict[edge_type] = gpd.GeoDataFrame(
            records, geometry="geometry", index=indices, crs=crs,
        )

        # Restore index names
        edge_index_names = G.graph.get("_edge_index_names", {})
        if edge_type in edge_index_names:
            index_names = edge_index_names[edge_type]
            if (isinstance(index_names, list) and len(index_names) == 1
                and index_names[0] is not None):
                edges_dict[edge_type].index.name = index_names[0]

    return edges_dict



