"""Module for loading and processing geospatial data from Overture Maps."""

import logging
from typing import Any

import geopandas as gpd
import momepy
import networkx as nx
from shapely.geometry import LineString
from shapely.geometry import Point

# Define the public API for this module
__all__ = [
    "create_tessellation",
    "filter_graph_by_distance",
    "gdf_to_nx",
    "nx_to_gdf",
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
    nodes_within_distance = set()

    for point in center_points:
        try:
            nearest_node = _get_nearest_node(point, nodes_gdf, node_id=node_id_name)
            distance_dict = nx.shortest_path_length(graph, nearest_node, weight=edge_attr)

            # Add nodes within distance from this center
            nodes_within_distance.update(
                node_id for node_id, dist in distance_dict.items() if dist < distance
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            logger.warning("Could not compute paths from a center point: %s", e, stacklevel=2)

    return nodes_within_distance


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


def filter_graph_by_distance(network: gpd.GeoDataFrame | nx.Graph,
                             center_point: Point | gpd.GeoSeries | gpd.GeoDataFrame,
                             distance: float,
                             edge_attr: str = "length",
                             node_id_col: str | None = None) -> gpd.GeoDataFrame | nx.Graph:
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
        Filtered network containing only elements within distance of any center point.
        Returns the same type as the input (either GeoDataFrame or NetworkX graph).
    """
    is_graph_input = isinstance(network, nx.Graph)

    # Convert input to graph and preserve CRS
    if is_graph_input:
        graph = network
        original_crs = None
    else:
        graph = momepy.gdf_to_nx(network)
        original_crs = network.crs

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
    filtered_gdf = momepy.nx_to_gdf(subgraph, points=False)
    if not isinstance(filtered_gdf.geometry, gpd.GeoSeries):
        filtered_gdf = gpd.GeoDataFrame(
            filtered_gdf, geometry="geometry", crs=original_crs,
        )

    return filtered_gdf


def _validate_gdf(nodes: gpd.GeoDataFrame | None,
                  edges: gpd.GeoDataFrame | None) -> None:
    """Validate node and edge GeoDataFrames for correct geometry types and matching CRS."""
    if edges is not None and not edges.geometry.apply(lambda g: isinstance(g, LineString)).all():
        msg = "Edges GeoDataFrame must have LineString geometries"
        raise ValueError(msg)
    if nodes is not None and not nodes.geometry.apply(lambda g: isinstance(g, Point)).all():
        msg = "Nodes GeoDataFrame must have Point geometries"
        raise ValueError(msg)
    if nodes is not None and edges is not None and nodes.crs != edges.crs:
        msg = "CRS of nodes and edges must match"
        raise ValueError(msg)


def _prepare_dataframes(nodes: gpd.GeoDataFrame | None,
                        edges: gpd.GeoDataFrame | None,
                        node_id_col: str | None,
                        edge_id_col: str | None) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame]:
    """Prepare and standardize node and edge dataframes."""
    if edges is None:
        msg = "Must provide edges GeoDataFrame"
        raise ValueError(msg)

    _validate_gdf(nodes, edges)

    # Standardize node IDs
    if nodes is not None:
        if node_id_col and node_id_col in nodes.columns:
            nodes = nodes.set_index(node_id_col, drop=False)
        else:
            nodes = nodes.reset_index(drop=True)

    # Standardize edge IDs
    if edge_id_col and edge_id_col in edges.columns:
        edges = edges.set_index(edge_id_col, drop=False)
    else:
        edges = edges.reset_index(drop=True)

    return nodes, edges


def _add_nodes_to_graph(graph: nx.Graph, nodes: gpd.GeoDataFrame) -> None:
    """Add nodes and their attributes to the graph."""
    graph.add_nodes_from(nodes.index)
    node_attrs = nodes.drop(columns="geometry").to_dict("index")
    nx.set_node_attributes(graph, node_attrs)
    pos = {idx: (geom.x, geom.y) for idx, geom in nodes.geometry.items()}
    nx.set_node_attributes(graph, pos, "pos")


def _prepare_edge_dataframe(edges: gpd.GeoDataFrame,
                           nodes: gpd.GeoDataFrame | None,
                           keep_geom: bool) -> tuple[gpd.GeoDataFrame, list[str]]:
    """Prepare edge dataframe with node mappings and select appropriate columns."""
    edges_df = edges.copy()

    # Determine edge node IDs based on provided nodes or coordinate tuples
    if nodes is not None:
        coord_map = {(geom.x, geom.y): idx for idx, geom in nodes.geometry.items()}
        edges_df["u"] = edges_df.geometry.map(lambda g: coord_map[g.coords[0]])
        edges_df["v"] = edges_df.geometry.map(lambda g: coord_map[g.coords[-1]])
    else:
        edges_df["u"] = edges_df.geometry.map(lambda g: g.coords[0])
        edges_df["v"] = edges_df.geometry.map(lambda g: g.coords[-1])

    # Select edge attributes based on keep_geom parameter
    if keep_geom:
        edge_cols = [col for col in edges_df.columns if col not in ["u", "v"]]
    else:
        geom_cols = [col for col in edges_df.columns
                    if edges_df[col].apply(lambda x: hasattr(x, "geom_type")).all()]
        edge_cols = [col for col in edges_df.columns
                    if col not in ["u", "v"] and col not in geom_cols]

    return edges_df, edge_cols


def _add_edges_to_graph(graph: nx.Graph,
                       edges_df: gpd.GeoDataFrame,
                       edge_cols: list[str]) -> None:
    """Add edges to the graph using vectorized operations."""
    graph_with_edges = nx.from_pandas_edgelist(
        edges_df,
        source="u",
        target="v",
        edge_attr=edge_cols,
    )
    graph.add_edges_from(graph_with_edges.edges(data=True))


def gdf_to_nx(nodes: gpd.GeoDataFrame | None = None,
              edges: gpd.GeoDataFrame | None = None,
              node_id_col: str | None = None,
              edge_id_col: str | None = None,
              keep_geom: bool = True) -> nx.Graph:
    """
    Convert GeoDataFrames of nodes and edges to a NetworkX graph.

    Parameters
    ----------
    nodes : GeoDataFrame, optional
        Point geometries for graph nodes; node attributes preserved.
    edges : GeoDataFrame
        LineString geometries for graph edges; edge attributes preserved.
    node_id_col : str, optional
        Column name to use for node identifiers; if None, sequential IDs are assigned.
    edge_id_col : str, optional
        Column name to use for edge identifiers; if None, sequential IDs are assigned.
    keep_geom : bool, default=True
        If True, include original geometry columns on edges.

    Returns
    -------
    networkx.Graph
        Graph with 'crs', 'node_geom_cols', 'edge_geom_cols', node 'pos', and attributes.
    """
    # Prepare and validate dataframes
    nodes, edges = _prepare_dataframes(nodes, edges, node_id_col, edge_id_col)

    # Initialize graph with metadata
    graph = nx.Graph()
    crs = nodes.crs if nodes is not None else edges.crs
    graph.graph.update({
        "crs": crs,
        "node_index_col": node_id_col,
        "edge_index_col": edge_id_col,
    })

    # Store geometry column names
    if nodes is not None:
        graph.graph["node_geom_cols"] = list(nodes.select_dtypes(include=["geometry"]).columns)
    if edges is not None:
        graph.graph["edge_geom_cols"] = list(edges.select_dtypes(include=["geometry"]).columns)

    # Add nodes if provided
    if nodes is not None:
        _add_nodes_to_graph(graph, nodes)

    # Add edges
    edges_df, edge_cols = _prepare_edge_dataframe(edges, nodes, keep_geom)
    _add_edges_to_graph(graph, edges_df, edge_cols)

    # Ensure node positions exist for fallback
    if nodes is None:
        pos = {node: node for node in graph.nodes()}
        nx.set_node_attributes(graph, pos, "pos")

    return graph



def _validate_nx(graph: nx.Graph, nodes: bool = False) -> None:
    """Validate NetworkX graph before converting to GeoDataFrame."""
    crs = graph.graph.get("crs")
    if crs is None:
        msg = "Missing CRS in graph attributes"
        raise ValueError(msg)

    pos = nx.get_node_attributes(graph, "pos")
    if nodes and not pos:
        msg = "Missing 'pos' attribute for nodes"
        raise ValueError(msg)
    if not nodes:
        # For edges, check if we have either geometry or pos for creating LineStrings
        has_geometry = any(attrs.get("geometry") is not None
                          for _, _, attrs in graph.edges(data=True))
        if not has_geometry and not pos:
            msg = "Missing edge geometry and node positions"
            raise ValueError(msg)


def _create_nodes_gdf_from_graph(graph: nx.Graph) -> gpd.GeoDataFrame:
    """Create nodes GeoDataFrame from NetworkX graph."""
    pos = nx.get_node_attributes(graph, "pos")
    crs = graph.graph.get("crs")
    node_geom_cols = graph.graph.get("node_geom_cols", [])

    records = [
        {
            "node_id": nid,
            **{k: v for k, v in attrs.items() if k != "pos"},
            "geometry": Point(*pos[nid]) if isinstance(pos[nid], (tuple, list)) else Point(pos[nid], 0),
        }
        for nid, attrs in graph.nodes(data=True)
    ]

    gdf = gpd.GeoDataFrame(records, crs=crs)

    # Convert geometry columns to GeoSeries
    for col in node_geom_cols:
        if col in gdf.columns:
            gdf[col] = gpd.GeoSeries(gdf[col], crs=crs)

    # Restore original node index
    node_index_col = graph.graph.get("node_index_col")
    if node_index_col:
        return gdf.set_index("node_id", drop=True).rename_axis(node_index_col)
    return gdf.set_index("node_id", drop=True).rename_axis(None)


def _create_edges_gdf_from_graph(graph: nx.Graph) -> gpd.GeoDataFrame:
    """Create edges GeoDataFrame from NetworkX graph."""
    pos = nx.get_node_attributes(graph, "pos")
    crs = graph.graph.get("crs")
    edge_geom_cols = graph.graph.get("edge_geom_cols", [])

    edge_data = list(graph.edges(data=True))
    records = [
        {
            **{k: v for k, v in attrs.items() if k != "geometry"},
            "u": u,
            "v": v,
            "geometry": (
                attrs.get("geometry")
                if attrs.get("geometry") is not None
                else LineString([pos[u], pos[v]])
            ),
        }
        for u, v, attrs in edge_data
    ]

    gdf = gpd.GeoDataFrame(records, crs=crs)

    # Convert geometry columns to GeoSeries
    for col in edge_geom_cols:
        if col in gdf.columns:
            gdf[col] = gpd.GeoSeries(gdf[col], crs=crs)

    # Restore original edge index if available
    edge_index_col = graph.graph.get("edge_index_col")
    if edge_index_col and edge_index_col in gdf.columns:
        gdf = gdf.set_index(edge_index_col, drop=True)

    return gdf


def nx_to_gdf(
    G: nx.Graph,
    nodes: bool = False,
    edges: bool = False,
) -> gpd.GeoDataFrame:
    """
    Convert a NetworkX graph to a GeoDataFrame for nodes or edges.

    Parameters
    ----------
    G : networkx.Graph
        Graph with 'crs', 'node_geom_cols', and 'edge_geom_cols' in G.graph.
    nodes : bool, default=False
        If True, return node GeoDataFrame.
    edges : bool, default=False
        If True, return edge GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with correct CRS and all geometry columns as GeoSeries.

    Raises
    ------
    ValueError
        If required 'pos' or geometry attributes are missing, or if both/neither flags are set.
    """
    # Both requested: return node and edge frames via separate calls
    if nodes and edges:
        nodes_gdf = nx_to_gdf(G, nodes=True, edges=False)
        edges_gdf = nx_to_gdf(G, nodes=False, edges=True)
        return nodes_gdf, edges_gdf

    # Require exactly one of nodes or edges
    if nodes == edges:
        msg = "Specify exactly one of 'nodes' or 'edges'"
        raise ValueError(msg)

    # Fallback for nodes-only graphs: treat node IDs as coords
    if nodes and not nx.get_node_attributes(G, "pos"):
        fallback = {n: n for n in G.nodes()}
        nx.set_node_attributes(G, fallback, "pos")

    # Validate the graph structure
    _validate_nx(G, nodes=nodes)

    # Create and return appropriate GeoDataFrame
    if nodes:
        return _create_nodes_gdf_from_graph(G)
    return _create_edges_gdf_from_graph(G)
