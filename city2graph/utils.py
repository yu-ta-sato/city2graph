"""Module for loading and processing geospatial data from Overture Maps."""

import logging
from typing import Any

import geopandas as gpd
import momepy
import networkx as nx
from shapely.geometry import Point

# Define the public API for this module
__all__ = [
    "create_tessellation",
    "filter_graph_by_distance",
]

logger = logging.getLogger(__name__)


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
    # Create tessellation using momepy based on whether primary_barriers are provided
    if primary_barriers is not None:
        # Convert primary_barriers to GeoDataFrame if it's a GeoSeries
        if isinstance(primary_barriers, gpd.GeoSeries):
            primary_barriers = gpd.GeoDataFrame(
                geometry=primary_barriers, crs=primary_barriers.crs,
            )

        # Ensure the barriers are in the same CRS as the input geometry
        if geometry.crs != primary_barriers.crs:
            msg = "CRS mismatch: geometry and barriers must have the same CRS."
            raise ValueError(
                msg,
            )

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


def _get_nearest_node(point: Point | gpd.GeoSeries,
                      nodes_gdf: gpd.GeoDataFrame,
                      node_id: str = "node_id") -> str | int:
    """Find the nearest node in a GeoDataFrame."""
    if isinstance(point, gpd.GeoSeries):
        point = point.iloc[0]
    nearest_idx = nodes_gdf.distance(point).idxmin()
    return nodes_gdf.loc[nearest_idx, node_id]


def filter_graph_by_distance(
    network: gpd.GeoDataFrame | nx.Graph,
    center_point: Point | gpd.GeoSeries | gpd.GeoDataFrame,
    distance: float,
    edge_attr: str = "length",
    node_id_col: str | None = None,
) -> gpd.GeoDataFrame | nx.Graph:
    """
    Extract a filtered network containing only elements within a given shortest-path distance.

    Filters network elements based on distance from specified center point(s).

    Parameters
    ----------
    network : Union[gpd.GeoDataFrame, nx.Graph]
        Input network data as either a GeoDataFrame of edges or a NetworkX graph.
    center_point : Union[Point, gpd.GeoSeries, gpd.GeoDataFrame]
        Center point(s) for distance calculation.
        Can be a single Point, GeoSeries of points, or GeoDataFrame with point geometries.
    distance : float, default=1000
        Maximum shortest-path distance from any center node.
    edge_attr : str, default="length"
        Edge attribute to use as weight for distance calculation.
    node_id_col : Optional[str], default=None
        Column name in nodes GeoDataFrame to use as node identifier.
        If None, will use auto-generated node IDs.

    Returns
    -------
    Union[gpd.GeoDataFrame, nx.Graph]
        Filtered network containing only elements within distance of any center point.
        Returns the same type as the input (either GeoDataFrame or NetworkX graph).
    """
    # Determine input type
    is_graph_input = isinstance(network, nx.Graph)

    # Convert to NetworkX graph if input is GeoDataFrame
    if is_graph_input:
        G = network
        original_crs = None
    else:
        G = momepy.gdf_to_nx(network)
        original_crs = network.crs

    # Build a GeoDataFrame for the nodes using their 'x' and 'y' attributes
    node_ids, node_geometries = zip(
        *[
            (nid, Point([attrs.get("x"), attrs.get("y")]))
            for nid, attrs in G.nodes(data=True)
        ], strict=False,
    )

    # Use provided node_id_col or default
    node_id_name = node_id_col or "node_id"

    nodes_gdf = gpd.GeoDataFrame(
        {node_id_name: node_ids, "geometry": node_geometries}, crs=original_crs,
    )

    # Initialize a set to collect nodes within distance from any center point
    nodes_within_distance = set()

    # Handle different types of center_point input
    center_points = center_point
    if isinstance(center_point, (gpd.GeoSeries, gpd.GeoDataFrame)):
        # If it's a GeoDataFrame, convert to GeoSeries
        if isinstance(center_point, gpd.GeoDataFrame):
            center_points = center_point.geometry
    else:
        # Convert single point to a list
        center_points = [center_point]

    # Process each center point
    for point in center_points:
        # Find the nearest node to this center
        nearest_node = _get_nearest_node(point, nodes_gdf, node_id=node_id_name)

        # Compute shortest path lengths from this center
        try:
            distance_dict = nx.shortest_path_length(G, nearest_node, weight=edge_attr)
            # Add nodes within distance from this center
            nodes_within_distance.update(
                k for k, v in distance_dict.items() if v < distance
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            logger.warning("Could not compute paths from a center point: %s", e, stacklevel=2)

    # Extract subgraph for nodes within distance from any center
    if nodes_within_distance:
        # Create a subgraph from the original graph
        subgraph = G.subgraph(nodes_within_distance)

        # Return the result in the same format as the input
        if is_graph_input:
            return subgraph
        filtered_gdf = momepy.nx_to_gdf(subgraph, points=False)

        # Ensure that the geometry column is properly set as GeoSeries
        if not isinstance(filtered_gdf.geometry, gpd.GeoSeries):
            filtered_gdf = gpd.GeoDataFrame(
                filtered_gdf, geometry="geometry", crs=original_crs,
            )

        return filtered_gdf
    # Return empty result in the same format as the input
    if is_graph_input:
        return nx.Graph()
    return gpd.GeoDataFrame(geometry=[], crs=original_crs)
