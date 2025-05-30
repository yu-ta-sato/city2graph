"""Network analysis functions for creating and manipulating network graphs."""

import warnings

import geopandas as gpd
import libpysal
import momepy
import networkx as nx
import numpy as np
import pandas as pd
import shapely

from city2graph.utils import create_tessellation
from city2graph.utils import filter_network_by_distance

# Define the public API for this module
__all__ = [
    "convert_gdf_to_dual",
    "morphological_graph",
    "private_to_private_network",
    "private_to_public_network",
    "public_to_public_network",
]


def _validate_inputs(privates: gpd.GeoDataFrame,
                     publics: gpd.GeoDataFrame,
                     buffer: float) -> None:
    """Validate input parameters for _get_adjacent_publics."""
    if not isinstance(privates, gpd.GeoDataFrame):
        msg = "privates must be a GeoDataFrame"
        raise TypeError(msg)
    if not isinstance(publics, gpd.GeoDataFrame):
        msg = "publics must be a GeoDataFrame"
        raise TypeError(msg)
    if not isinstance(buffer, (int, float)):
        msg = "buffer must be a number"
        raise TypeError(msg)


def _check_empty_dataframes(privates: gpd.GeoDataFrame,
                            publics: gpd.GeoDataFrame) -> bool:
    """Check if DataFrames are empty and warn accordingly."""
    if privates.empty:
        warnings.warn("privates GeoDataFrame is empty", RuntimeWarning, stacklevel=3)
        return True
    if publics.empty:
        warnings.warn("publics GeoDataFrame is empty", RuntimeWarning, stacklevel=3)
        return True
    return False


def _validate_columns(
    publics: gpd.GeoDataFrame,
    privates: gpd.GeoDataFrame,
    public_id_col: str,
    private_id_col: str | None,
    public_geom_col: str | None) -> None:
    """Validate that required columns exist in DataFrames."""
    if public_id_col not in publics.columns:
        msg = f"public_id_col '{public_id_col}' not found in publics"
        raise ValueError(msg)
    if private_id_col and private_id_col not in privates.columns:
        msg = f"private_id_col '{private_id_col}' not found in privates"
        raise ValueError(msg)
    if public_geom_col and public_geom_col not in publics.columns:
        msg = f"public_geom_col '{public_geom_col}' not found in publics"
        raise ValueError(msg)


def _validate_geometries(privates: gpd.GeoDataFrame,
                         publics: gpd.GeoDataFrame) -> None:
    """Validate geometry types in DataFrames."""
    if not all(
        isinstance(geom, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon))
        for geom in privates.geometry
    ):
        warnings.warn(
            "Some geometries in privates are not Polygons or MultiPolygons",
            RuntimeWarning, stacklevel=3,
        )
    if not all(
        isinstance(
            geom, (shapely.geometry.LineString, shapely.geometry.MultiLineString),
        )
        for geom in publics.geometry
    ):
        warnings.warn(
            "Some geometries in publics are not LineStrings or MultiLineStrings",
            RuntimeWarning, stacklevel=3,
        )


def _get_adjacent_publics(
    privates: gpd.GeoDataFrame,
    publics: gpd.GeoDataFrame,
    public_id_col: str = "id",
    private_id_col: str | None = None,
    public_geom_col: str | None = None,
    buffer: float = 1) -> dict[str, list[str]]:
    """
    Identify the streets that are contained in or adjacent to each polygon.

    This function returns a dictionary mapping each polygon (by its index)
    to a list of street identifiers that intersect the polygon geometry.

    Parameters
    ----------
    privates : geopandas.GeoDataFrame
        GeoDataFrame containing private space polygons
    publics : geopandas.GeoDataFrame
        GeoDataFrame containing public space linestrings
    public_id_col : str, default "id"
        Column name in publics that uniquely identifies each public space
    private_id_col : str, default None
        Column name in privates that uniquely identifies each private space
        If None, will use DataFrame index
    public_geom_col : str, default None
        Column name in publics containing alternative geometry to use
        If None, will use the geometry column
    buffer : float, default 1
        Buffer distance to apply to public geometries for intersection

    Returns
    -------
    dict
        Dictionary mapping private space IDs to lists of adjacent public space IDs

    Raises
    ------
    ValueError
        If input GeoDataFrames are empty or required columns are missing
    TypeError
        If inputs are not GeoDataFrames or if buffer is not a number
    """
    # Input validation
    _validate_inputs(privates, publics, buffer)

    # Check for empty DataFrames
    if _check_empty_dataframes(privates, publics):
        return {}

    # Check if required columns exist
    _validate_columns(publics, privates, public_id_col, private_id_col, public_geom_col)

    # Check geometry types
    _validate_geometries(privates, publics)

    # Ensure both GeoDataFrames use the same CRS
    if privates.crs != publics.crs:
        publics = publics.to_crs(privates.crs)

    # Prepare ID columns
    if private_id_col is None:
        privates = privates.reset_index(drop=True)
        private_id_col = "index"

    if public_id_col is None:
        publics = publics.reset_index()
        public_id_col = "index"

    # Determine the street geometry column and create buffered version
    publics_copy = publics.copy()
    if public_geom_col is not None:
        publics_copy.geometry = gpd.GeoSeries(
            publics_copy[public_geom_col], crs=publics_copy.crs,
        )

    # Create buffered geometries for intersection test
    publics_copy.geometry = publics_copy.geometry.buffer(buffer)

    # Perform spatial join to get publics intersecting each polygon
    joined = gpd.sjoin(
        publics_copy,
        privates,
        how="inner",
        predicate="intersects",
    )

    # Group by private_id and collect public ids - vectorized approach
    index_col = "index_right" if private_id_col == "index" else private_id_col
    return (
        joined.groupby(index_col)[public_id_col]
        .apply(lambda x: x.unique().tolist())
        .to_dict()
    )



def _extract_dual_graph_nodes(dual_graph: nx.Graph,
                              id_col: str,
                              gdf_crs: str | dict | None) -> gpd.GeoDataFrame:
    """
    Extract nodes from dual graph into a GeoDataFrame.

    Parameters
    ----------
    dual_graph : networkx.Graph
        Dual graph representation of the network
    id_col : str
        Column name for the node identifiers
    gdf_crs : pyproj.CRS
        Coordinate reference system for the output GeoDataFrame

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing nodes from the dual graph
    """
    # Extract nodes data efficiently
    nodes_data = pd.DataFrame.from_dict(
        dict(dual_graph.nodes(data=True)), orient="index",
    )

    if nodes_data.empty:
        return gpd.GeoDataFrame(
            columns=[id_col, "geometry"], geometry="geometry", crs=gdf_crs,
        )

    # Check if id_col exists before filtering
    if id_col in nodes_data.columns:
        nodes_data = nodes_data[nodes_data[id_col].notna()]

        # Create geometries from coordinates
        nodes_data["geometry"] = [shapely.Point(coord) for coord in nodes_data.index]

        # Create a GeoDataFrame
        dual_node_gdf = gpd.GeoDataFrame(nodes_data, geometry="geometry", crs=gdf_crs)

        # Set the index of the dual_gdf
        return dual_node_gdf.set_index(dual_node_gdf[id_col]).drop(
            columns=[id_col],
        )

    return gpd.GeoDataFrame(
        columns=[id_col, "geometry"], geometry="geometry", crs=gdf_crs,
    )


def _extract_node_connections(dual_graph: nx.Graph,
                              id_col: str) -> dict:
    """
    Extract connections between nodes from dual graph.

    Parameters
    ----------
    dual_graph : networkx.Graph
        Dual graph representation of the network
    id_col : str
        Column name for the node identifiers

    Returns
    -------
    dict
        Dictionary mapping node IDs to lists of connected node IDs
    """
    # Extract connections using dictionary comprehension
    connections = {}
    for node in dual_graph.nodes():
        node_id = dual_graph.nodes[node].get(id_col, node)
        connections[node_id] = [
            dual_graph.nodes[n].get(id_col, n) for n in dual_graph.neighbors(node)
        ]

    return connections


def _find_additional_connections(line_gdf: gpd.GeoDataFrame,
                                 id_col: str,
                                 tolerance: float,
                                 connections: dict | None = None) -> dict:
    """
    Find additional connections between lines based on endpoint proximity.

    Parameters
    ----------
    line_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing LineString geometries
    id_col : str
        Column name for the line identifiers
    tolerance : float
        Distance tolerance for endpoint connections
    connections : dict, optional
        Existing connections dictionary to update

    Returns
    -------
    dict
        Updated dictionary of connections
    """
    if connections is None:
        connections = {}

    if line_gdf.empty:
        return connections

    # Extract endpoints vectorized
    endpoints_data = []

    # This part builds a list of endpoints for each linestring
    for idx, row in line_gdf.iterrows():
        if not isinstance(row.geometry, shapely.geometry.LineString):
            continue

        coords = list(row.geometry.coords)
        endpoints_data.extend(
            [
                {
                    "line_idx": idx,
                    id_col: row[id_col],
                    "endpoint": shapely.Point(coords[0]),
                },
                {
                    "line_idx": idx,
                    id_col: row[id_col],
                    "endpoint": shapely.Point(coords[-1]),
                },
            ],
        )

    if not endpoints_data:
        return connections

    # Create GeoDataFrame of endpoints
    endpoints_gdf = gpd.GeoDataFrame(
        endpoints_data, geometry="endpoint", crs=line_gdf.crs,
    )

    # Create buffers around endpoints for spatial join
    endpoints_gdf["buffer_geom"] = endpoints_gdf.endpoint.buffer(tolerance)

    # Prepare lines for joining
    lines_for_join = (
        line_gdf[[id_col, "geometry"]]
        .reset_index()
        .rename(columns={"index": "line_index"})
    )

    # Spatial join buffered endpoints with lines
    joined = gpd.sjoin(
        endpoints_gdf.set_geometry("buffer_geom"),
        lines_for_join,
        predicate="intersects",
        how="left",
    )

    # Filter out self-matches
    joined = joined[joined.line_idx != joined.line_index]

    # Update connections dictionary based on endpoint intersections
    if not joined.empty:
        # Determine column names
        orig_id_col = f"{id_col}_left" if f"{id_col}_left" in joined.columns else id_col
        other_id_col = (
            f"{id_col}_right" if f"{id_col}_right" in joined.columns else id_col
        )

        # Process each valid connection
        for _, row in joined.iterrows():
            orig_id = row[orig_id_col]
            other_id = row[other_id_col]

            if orig_id not in connections:
                connections[orig_id] = []
            if other_id not in connections[orig_id]:
                connections[orig_id].append(other_id)

            if other_id not in connections:
                connections[other_id] = []
            if orig_id not in connections[other_id]:
                connections[other_id].append(orig_id)

    return connections


def convert_gdf_to_dual(gdf: gpd.GeoDataFrame,
                        id_col: str | None = None,
                        tolerance: float = 1e-8) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Convert a GeoDataFrame to a NetworkX graph and then back to a GeoDataFrame.

    Also detects connections where endpoints of one linestring are on another linestring.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing LineString geometries to convert
    id_col : str, default None
        Column name that uniquely identifies each feature
        If None, will use DataFrame index
    tolerance : float, default 1e-8
        Distance tolerance for detecting endpoint connections

    Returns
    -------
    tuple
        (GeoDataFrame of nodes, dict of connections)

    Raises
    ------
    TypeError
        If input is not a GeoDataFrame or if tolerance is not a number
    ValueError
        If GeoDataFrame is empty or contains invalid geometries
    """
    # Input validation
    if not isinstance(gdf, gpd.GeoDataFrame):
        msg = "Input must be a GeoDataFrame"
        raise TypeError(msg)
    if not isinstance(tolerance, (int, float)):
        msg = "Tolerance must be a number"
        raise TypeError(msg)
    if tolerance < 0:
        msg = "Tolerance must be non-negative"
        raise ValueError(msg)

    # Check for empty DataFrame
    if gdf.empty:
        warnings.warn("Input GeoDataFrame is empty", RuntimeWarning, stacklevel=2)
        return (gpd.GeoDataFrame(columns=[id_col] if id_col else [], crs=gdf.crs), {})

    # Validate geometry types
    invalid_geoms = gdf.geometry.apply(
        lambda g: not isinstance(
            g, (shapely.geometry.LineString, shapely.geometry.MultiLineString),
        ),
    )
    if invalid_geoms.any():
        warnings.warn(
            f"Found {invalid_geoms.sum()} geometries that are not LineString or MultiLineString",
            RuntimeWarning, stacklevel=2,
        )

    # Check for null geometries
    if gdf.geometry.isna().any():
        warnings.warn("Found null geometries in input GeoDataFrame", RuntimeWarning, stacklevel=2)
        gdf = gdf.dropna(subset=["geometry"])

    # Handle ID column
    if id_col is None:
        gdf = gdf.reset_index(drop=True)
        gdf["temp_id"] = gdf.index
        id_col = "temp_id"

    # Convert to NetworkX graph
    dual_graph = momepy.gdf_to_nx(gdf, approach="dual", preserve_index=True)

    # Extract nodes as GeoDataFrame
    dual_node_gdf = _extract_dual_graph_nodes(dual_graph, id_col, gdf.crs)

    # Extract node connections from graph
    connections = _extract_node_connections(dual_graph, id_col)

    # Find additional connections based on endpoints
    line_gdf = gdf[
        gdf.geometry.apply(lambda g: isinstance(g, shapely.geometry.LineString))
    ]
    connections = _find_additional_connections(line_gdf, id_col, tolerance, connections)

    return dual_node_gdf, connections


def _create_connecting_lines(
    enclosed_tess: gpd.GeoDataFrame,
    public_dual_gdf: gpd.GeoDataFrame,
    adjacent_streets: dict,
    private_id_col: str | None = None,
    public_id_col: str | None = None) -> gpd.GeoDataFrame:
    """
    Create a LineStrings connecting the centroid of each tessellation to the nodes of adjacent streets.

    Parameters
    ----------
    enclosed_tess : geopandas.GeoDataFrame
        GeoDataFrame containing private space polygons
    public_dual_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing public space nodes
    adjacent_streets : dict
        Dictionary mapping private space IDs to lists of adjacent public space IDs
    private_id_col : str, default None
        Column name in enclosed_tess that uniquely identifies each private space
        If None, will use DataFrame index
    public_id_col : str, default None
        Column name in public_dual_gdf that uniquely identifies each public space
        If None, will use DataFrame index

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing LineString connections between private and public spaces

    Raises
    ------
    TypeError
        If inputs are not of correct type
    ValueError
        If required data is missing or invalid
    """
    # Input validation
    if not isinstance(enclosed_tess, gpd.GeoDataFrame):
        msg = "enclosed_tess must be a GeoDataFrame"
        raise TypeError(msg)
    if not isinstance(public_dual_gdf, gpd.GeoDataFrame):
        msg = "public_dual_gdf must be a GeoDataFrame"
        raise TypeError(msg)
    if not isinstance(adjacent_streets, dict):
        msg = "adjacent_streets must be a dictionary"
        raise TypeError(msg)

    # Check for empty inputs
    if enclosed_tess.empty:
        warnings.warn("enclosed_tess GeoDataFrame is empty", RuntimeWarning, stacklevel=2)
        return gpd.GeoDataFrame(
            {"private_id": [], "public_id": [], "geometry": []}, crs=enclosed_tess.crs,
        )
    if public_dual_gdf.empty:
        warnings.warn("public_dual_gdf GeoDataFrame is empty", RuntimeWarning, stacklevel=2)
        return gpd.GeoDataFrame(
            {"private_id": [], "public_id": [], "geometry": []}, crs=enclosed_tess.crs,
        )
    if not adjacent_streets:
        warnings.warn("adjacent_streets dictionary is empty", RuntimeWarning, stacklevel=2)
        return gpd.GeoDataFrame(
            {"private_id": [], "public_id": [], "geometry": []}, crs=enclosed_tess.crs,
        )

    # Check CRS compatibility
    if enclosed_tess.crs != public_dual_gdf.crs:
        warnings.warn(
            "CRS mismatch between enclosed_tess and public_dual_gdf. "
            "Converting public_dual_gdf to match enclosed_tess CRS.",
            RuntimeWarning, stacklevel=2,
        )
        public_dual_gdf = public_dual_gdf.to_crs(enclosed_tess.crs)

    # Handle private ID column
    if private_id_col is None:
        enclosed_tess = enclosed_tess.reset_index(drop=True)
        private_id_col = "index"
        enclosed_tess[private_id_col] = enclosed_tess.index

    # Handle public ID column
    if public_id_col is None:
        public_dual_gdf = public_dual_gdf.reset_index(drop=True)
        public_id_col = "index"
        public_dual_gdf[public_id_col] = public_dual_gdf.index

    # Convert the adjacency dictionary to a DataFrame for vectorized operations
    connections = []
    for private_id, public_ids in adjacent_streets.items():
        connections.extend(
            [{"private_id": private_id, "public_id": pid} for pid in public_ids],
        )

    if not connections:
        return gpd.GeoDataFrame(
            {"private_id": [], "public_id": [], "geometry": []}, crs=enclosed_tess.crs,
        )

    # Create DataFrame of connections
    connections_df = pd.DataFrame(connections)

    # Compute centroids once
    centroids = enclosed_tess.set_index(private_id_col)["geometry"].centroid

    # Join connections with centroids
    connections_df = connections_df.join(centroids.rename("centroid"), on="private_id")

    # Join with public geometries
    connections_df = connections_df.join(
        public_dual_gdf["geometry"].rename("public_geom"), on="public_id",
    )

    # Filter valid connections and create LineStrings
    valid_df = connections_df.dropna(subset=["centroid", "public_geom"])

    # Create LineStrings vectorized
    valid_df["geometry"] = valid_df.apply(
        lambda row: shapely.LineString(
            [(row.centroid.x, row.centroid.y), (row.public_geom.x, row.public_geom.y)],
        ),
        axis=1,
    )

    # Create final GeoDataFrame
    return gpd.GeoDataFrame(
        valid_df[["private_id", "public_id", "geometry"]],
        geometry="geometry",
        crs=enclosed_tess.crs,
    )


def _prep_contiguity_graph(
    tess_group: gpd.GeoDataFrame,
    private_id_col: str | None,
    contiguity: str) -> tuple[nx.Graph | None, dict | None]:
    """
    Prepare a contiguity-based graph for a group of tessellation polygons.

    Parameters
    ----------
    tess_group : geopandas.GeoDataFrame
        GeoDataFrame containing tessellation polygons for one group
    private_id_col : str
        Column name that uniquely identifies each private space
    contiguity : str
        Type of contiguity ('queen' or 'rook')

    Returns
    -------
    tuple
        (NetworkX graph, mapping of positions to IDs)
    """
    # Skip if only one polygon
    if tess_group.shape[0] < 2:
        return None, None

    # Create mapping from position to ID
    pos_to_id_mapping = dict(
        enumerate(tess_group[private_id_col] if private_id_col else tess_group.index),
    )

    # Create contiguity weights
    if contiguity.lower() == "queen":
        w = libpysal.weights.Queen.from_dataframe(
            tess_group, geom_col="geometry", use_index=False,
        )
    elif contiguity.lower() == "rook":
        w = libpysal.weights.Rook.from_dataframe(
            tess_group, geom_col="geometry", use_index=False,
        )
    else:
        msg = "contiguity must be 'queen' or 'rook'"
        raise ValueError(msg)

    w_nx = w.to_networkx()

    # Skip if no edges
    if w_nx.number_of_edges() == 0:
        return None, None

    # Compute centroids vectorized
    centroids = np.column_stack(
        [tess_group.geometry.centroid.x, tess_group.geometry.centroid.y],
    )

    # Create positions dictionary
    positions = {node: (x, y) for node, (x, y) in zip(w_nx.nodes(), centroids, strict=False)}

    # Set node attributes
    nx.set_node_attributes(w_nx, {n: pos[0] for n, pos in positions.items()}, "x")
    nx.set_node_attributes(w_nx, {n: pos[1] for n, pos in positions.items()}, "y")

    # Create edge geometries
    node_data = dict(w_nx.nodes(data=True))
    edge_geometry = {
        (u, v): shapely.geometry.LineString(
            [
                (node_data[u]["x"], node_data[u]["y"]),
                (node_data[v]["x"], node_data[v]["y"]),
            ],
        )
        for u, v in w_nx.edges()
    }

    nx.set_edge_attributes(w_nx, edge_geometry, "geometry")
    w_nx.graph["approach"] = "primal"

    return w_nx, pos_to_id_mapping


def private_to_private_network(
    private_gdf: gpd.GeoDataFrame,
    private_id_col: str | None = None,
    group_col: str | None = None,
    contiguity: str = "queen") -> gpd.GeoDataFrame:
    """
    Create contiguity matrices for private spaces.

    If group_col is provided, the contiguity is calculated for each group separately.

    Parameters
    ----------
    private_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing tessellation polygons
    private_id_col : str, default None
        Column name that uniquely identifies each private space
        If None, will use DataFrame index
    group_col : str, default None
        Column name used to group tessellation polygons
        If None, all polygons are treated as one group
    contiguity : str, default "queen"
        Type of contiguity, either 'queen' or 'rook'

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing LineString connections between adjacent private spaces

    Raises
    ------
    TypeError
        If input is not a GeoDataFrame
    ValueError
        If contiguity is not 'queen' or 'rook' or if required columns are missing
    """
    # Input validation
    if not isinstance(private_gdf, gpd.GeoDataFrame):
        msg = "private_gdf must be a GeoDataFrame"
        raise TypeError(msg)

    # Check for empty DataFrame
    if private_gdf.empty:
        warnings.warn("private_gdf GeoDataFrame is empty", RuntimeWarning, stacklevel=2)
        return gpd.GeoDataFrame(
            columns=[
                "from_private_id",
                "to_private_id",
                group_col or "group",
                "geometry",
            ],
            geometry="geometry",
            crs=private_gdf.crs,
        )

    # Validate contiguity parameter
    if contiguity.lower() not in ["queen", "rook"]:
        msg = "contiguity must be 'queen' or 'rook'"
        raise ValueError(msg)

    # Check specified columns exist
    if private_id_col is not None and private_id_col not in private_gdf.columns:
        msg = f"private_id_col '{private_id_col}' not found in private_gdf"
        raise ValueError(msg)

    if group_col is not None and group_col not in private_gdf.columns:
        msg = f"group_col '{group_col}' not found in private_gdf"
        raise ValueError(msg)

    # Validate geometry types
    invalid_geoms = private_gdf.geometry.apply(
        lambda g: not isinstance(
            g, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon),
        ),
    )
    if invalid_geoms.any():
        warnings.warn(
            f"Found {invalid_geoms.sum()} geometries that are not Polygon or MultiPolygon",
            RuntimeWarning, stacklevel=2,
        )

    # Check for null geometries
    if private_gdf.geometry.isna().any():
        warnings.warn("Found null geometries in private_gdf", RuntimeWarning, stacklevel=2)
        private_gdf = private_gdf.dropna(subset=["geometry"])

    # Handle private ID column
    if private_id_col is None:
        private_gdf = private_gdf.reset_index(drop=True)

    # Handle grouping
    if group_col is None:
        tess_groups = {"all": private_gdf.copy()}
    else:
        tess_groups = {name: group for name, group in private_gdf.groupby(group_col)}  # noqa: C416

    # List to collect edge DataFrames
    queen_edges_list = []

    for enc, tess_group in tess_groups.items():
        # Prepare contiguity graph for this group
        w_nx, pos_to_id_mapping = _prep_contiguity_graph(
            tess_group, private_id_col, contiguity,
        )

        if w_nx is None:
            continue

        # Convert to GeoDataFrame
        _, edges = momepy.nx_to_gdf(w_nx)
        edges = edges.set_crs(tess_group.crs)

        # Map node indices to IDs vectorized
        edges["from_private_id"] = edges["node_start"].map(pos_to_id_mapping)
        edges["to_private_id"] = edges["node_end"].map(pos_to_id_mapping)

        # Drop unnecessary columns
        edges = edges.drop(columns=["node_start", "node_end"])

        # Set group column
        group_key = group_col if group_col is not None else "group"
        edges[group_key] = enc

        # Select columns
        edges = edges[["from_private_id", "to_private_id", group_key, "geometry"]]
        queen_edges_list.append(edges)

    if queen_edges_list:
        return pd.concat(queen_edges_list, ignore_index=True)
    cols = [
        "from_private_id",
        "to_private_id",
        group_col if group_col else "group",
        "geometry",
    ]
    return gpd.GeoDataFrame(columns=cols, crs=private_gdf.crs)


def private_to_public_network(
    private_gdf: gpd.GeoDataFrame,
    public_gdf: gpd.GeoDataFrame,
    private_id_col: str | None = None,
    public_id_col: str | None = None,
    public_geom_col: str | None = None,
    tolerance: float = 1.0) -> gpd.GeoDataFrame:
    """
    Create connections between private spaces and public spaces.

    Parameters
    ----------
    private_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing private space polygons
    public_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing public space linestrings
    private_id_col : str, default None
        Column name in private_gdf that uniquely identifies each private space
        If None, will use DataFrame index
    public_id_col : str, default None
        Column name in public_gdf that uniquely identifies each public space
        If None, will use DataFrame index
    public_geom_col : str, default None
        Column name in public_gdf containing alternative geometry to use
        If None, will use the geometry column
    tolerance : float, default 1
        Buffer tolerance for spatial joins

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing LineString connections between private and public spaces
    """
    # Create dual graph from publics
    public_dual_gdf, _ = convert_gdf_to_dual(
        public_gdf, id_col=public_id_col, tolerance=tolerance,
    )

    # Identify adjacent streets
    adjacent_publics = _get_adjacent_publics(
        private_gdf,
        public_gdf,
        private_id_col=private_id_col,
        public_id_col=public_id_col,
        public_geom_col=public_geom_col,
        buffer=tolerance,
    )

    # Create connecting lines
    return _create_connecting_lines(
        private_gdf,
        public_dual_gdf,
        adjacent_publics,
        private_id_col=private_id_col,
        public_id_col=public_id_col,
    )


def public_to_public_network(
    public_gdf: gpd.GeoDataFrame,
    public_id_col: str | None = None,
    tolerance: float = 1e-8) -> gpd.GeoDataFrame:
    """
    Create connections between public spaces represented as a dual graph.

    Parameters
    ----------
    public_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing public space linestrings
    public_id_col : str, default None
        Column name that uniquely identifies each public space
        If None, will use DataFrame index
    tolerance : float, default 1e-8
        Distance tolerance for detecting endpoint connections

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing LineString connections between public spaces
    """
    # Create dual representation
    public_dual_gdf, public_to_public_dict = convert_gdf_to_dual(
        public_gdf, id_col=public_id_col, tolerance=tolerance,
    )

    # Handle empty result
    if public_dual_gdf.empty or not public_to_public_dict:
        return gpd.GeoDataFrame(
            {"from_public_id": [], "to_public_id": [], "geometry": []},
            crs=public_gdf.crs,
        )

    # Extract point coordinates efficiently
    point_coords = public_dual_gdf.geometry.apply(lambda p: (p.x, p.y)).to_dict()

    # Track processed connections using a set
    processed_connections = set()

    # Prepare records list - preallocate for better performance
    connections_data = []

    # Generate unique connections
    for from_id, connected_ids in public_to_public_dict.items():
        if from_id not in point_coords:
            continue

        from_coords = point_coords[from_id]

        for to_id in connected_ids:
            # Create a unique identifier for this connection
            connection = tuple(sorted([from_id, to_id]))

            # Skip if already processed
            if connection in processed_connections or to_id not in point_coords:
                continue

            processed_connections.add(connection)
            to_coords = point_coords[to_id]

            # Add connection record
            connections_data.append(
                {
                    "from_public_id": from_id,
                    "to_public_id": to_id,
                    "geometry": shapely.LineString([from_coords, to_coords]),
                },
            )

    # Create GeoDataFrame from connection data
    if connections_data:
        return gpd.GeoDataFrame(
            connections_data, geometry="geometry", crs=public_dual_gdf.crs,
        )
    return gpd.GeoDataFrame(
        {"from_public_id": [], "to_public_id": [], "geometry": []},
        crs=public_gdf.crs,
    )


def morphological_graph(
    buildings_gdf: gpd.GeoDataFrame,
    segments_gdf: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame = None,
    distance: float | None = None,
    private_id_col: str = "tess_id",
    public_id_col: str = "id",
    public_geom_col: str = "barrier_geometry",
    contiguity: str = "queen") -> dict:
    """
    Create a morphological graph from buildings and road segments.

    The private spaces are represented as tessellations, and the public spaces are
    represented as a dual graph of road segments.

    This function performs a series of operations:
    1. Creates tessellations based on buildings and road barrier geometries
    2. Optionally filters the network by distance from a specified center point
    3. Identifies enclosed tessellations adjacent to the segments
    4. Creates three types of connections:
       - Private to private (between tessellation cells)
       - Public to public (between road segments)
       - Private to public (between tessellation cells and road segments)

    Parameters
    ----------
    buildings_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing building polygons
    segments_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing road segments as LineStrings.
        Should include a 'barrier_geometry' column or the column specified in public_geom_col.
    center_point : Union[Point, gpd.GeoSeries, gpd.GeoDataFrame], optional
        Optional center point for filtering the network by distance.
        If provided, only segments within the specified distance will be included.
    distance : float, default=1000
        Maximum network distance from center_point to include segments, if center_point is provided.
        If None, no distance filtering will be applied even if center_point is provided.
    private_id_col : str, default='tess_id'
        Column name that uniquely identifies each tessellation cell (private space).
    public_id_col : str, default='id'
        Column name that uniquely identifies each road segment (public space).
    public_geom_col : str, default='barrier_geometry'
        Column name in segments_gdf containing the processed geometry to use.
    tolerance : float, default=1
        Buffer tolerance for spatial joins and endpoint connections (in meters).
    contiguity : str, default='queen'
        Type of contiguity for private-to-private connections, either 'queen' or 'rook'.

    Returns
    -------
    dict
        Dictionary containing the following elements:
        - 'tessellation': GeoDataFrame of tessellation cells (private spaces)
        - 'segments': GeoDataFrame of road segments (public spaces)
        - 'buildings': GeoDataFrame of buildings
        - 'private_to_private': GeoDataFrame of connections between tessellation cells
        - 'public_to_public': GeoDataFrame of connections between road segments
        - 'private_to_public': GeoDataFrame of connections between tessellation cells and road segments

    Notes
    -----
    - If center_point is not provided, all segments will be included.
    - The barrier_geometry column should contain the processed geometries for road segments.
      If it doesn't exist, the function will use the column specified in public_geom_col.
    - The function requires the city2graph package and depends on create_tessellation,
      filter_network_by_distance, create_private_to_private, create_public_to_public,
      and create_private_to_public functions.
    """
    # Check if public_geom_col exists in segments_gdf
    if public_geom_col not in segments_gdf.columns:
        warnings.warn(
            f"Column '{public_geom_col}' not found in segments_gdf. Using 'geometry' column instead.",
            RuntimeWarning, stacklevel=2,
        )
        public_geom_col = "geometry"

    # Create tessellations based on buildings and road barrier geometries
    enclosed_tess = create_tessellation(
        buildings_gdf, primary_barriers=segments_gdf[public_geom_col],
    )

    # Optionally filter the network by distance from a specified center point
    if center_point is not None and distance is not None:
        segments_subset = filter_network_by_distance(
            segments_gdf, center_point, distance=distance,
        )
    else:
        segments_subset = segments_gdf.copy()

    # Get the enclosure indices that are adjacent to the segments using a spatial join
    adjacent_enclosure_indices = gpd.sjoin(
        enclosed_tess, segments_subset, how="inner", predicate="intersects",
    )["enclosure_index"].unique()

    # Get enclosed tessellation for the adjacent enclosures
    tess_subset = enclosed_tess[
        enclosed_tess["enclosure_index"].isin(adjacent_enclosure_indices)
    ]

    # Filter buildings that intersect with the enclosed tessellation
    buildings_subset = buildings_gdf[
        buildings_gdf.geometry.intersects(tess_subset.unary_union)
    ]

    # Create connections between adjacent private spaces (tessellation cells)
    private_to_private = private_to_private_network(
        tess_subset,
        private_id_col=private_id_col,
        group_col="enclosure_index",
        contiguity=contiguity,
    )

    # Create connections between street segments
    public_to_public = public_to_public_network(
        segments_subset,
        public_id_col=public_id_col,
    )

    # Create connections between private spaces and public spaces
    private_to_public = private_to_public_network(
        tess_subset,
        segments_subset,
        private_id_col=private_id_col,
        public_id_col=public_id_col,
        public_geom_col=public_geom_col,
    )

    # Return all created elements
    return {
        "tessellations": tess_subset,
        "segments": segments_subset,
        "buildings": buildings_subset,
        "private_to_private": private_to_private,
        "public_to_public": public_to_public,
        "private_to_public": private_to_public,
    }
