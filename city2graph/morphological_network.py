"""
Network analysis functions for creating and manipulating network graphs.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import shapely
import libpysal
import momepy
import warnings

# Define the public API for this module
__all__ = [
    'extract_nearby_segments',
    'convert_gdf_to_dual',
    'create_private_to_private',
    'create_private_to_public',
    'create_public_to_public'
]

# Helper function to find the nearest node in a GeoDataFrame.
def _get_nearest_node(point, nodes_gdf, node_id='node_id'):
    if isinstance(point, gpd.GeoSeries):
        point = point.iloc[0]
    nearest_idx = nodes_gdf.distance(point).idxmin()
    return nodes_gdf.loc[nearest_idx, node_id]


def extract_nearby_segments(gdf, center_point, threshold=1000):
    """
    Extracts a filtered segments GeoDataFrame containing only segments
    within a given shortest-path distance threshold from specified center point(s).

    Args:
        gdf (geopandas.GeoDataFrame): Input segments.
        center_point (geopandas.GeoSeries or geopandas.GeoDataFrame): Center point(s) for distance calculation.
                                                                      Can be a single point or multiple points.
        threshold (float): Maximum shortest-path distance from any center node.
    
    Returns:
        geopandas.GeoDataFrame: Filtered segments GeoDataFrame containing segments
                               within threshold distance of any center point.
    """
    
    # Convert segments GeoDataFrame to a NetworkX graph using momepy function.
    G = momepy.gdf_to_nx(gdf)
    
    # Build a GeoDataFrame for the nodes using their 'x' and 'y' attributes.
    node_ids, node_geometries = zip(*[
        (nid, shapely.Point([attrs.get('x'), attrs.get('y')]))
        for nid, attrs in G.nodes(data=True)
    ])
    
    nodes_gdf = gpd.GeoDataFrame(
        {'node_id': node_ids, 'geometry': node_geometries},
        crs=gdf.crs
    )
    
    # Initialize a set to collect nodes within threshold from any center point
    nodes_within_threshold = set()
    
    # Handle both single point and multiple points cases
    center_points = center_point
    if isinstance(center_point, gpd.GeoSeries) or isinstance(center_point, gpd.GeoDataFrame):
        # If it's a GeoDataFrame, convert to GeoSeries
        if isinstance(center_point, gpd.GeoDataFrame):
            center_points = center_point.geometry
    else:
        # Convert single point to a list
        center_points = [center_point]
    
    # Process each center point
    for point in center_points:
        # Find the nearest node to this center
        nearest_node = _get_nearest_node(point, nodes_gdf)
        
        # Compute shortest path lengths from this center
        try:
            distance_dict = nx.shortest_path_length(G, nearest_node, weight="length")
            # Add nodes within threshold from this center
            nodes_within_threshold.update(
                k for k, v in distance_dict.items() if v < threshold
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            warnings.warn(f"Could not compute paths from a center point: {e}", RuntimeWarning)
    
    # Extract subgraph for nodes within threshold from any center
    if nodes_within_threshold:
        # Create a subgraph from the original graph
        subgraph = G.subgraph(nodes_within_threshold)
        gdf_filtered = momepy.nx_to_gdf(subgraph, points=False)

        # Ensure that the geometry column is properly set as GeoSeries
        if not isinstance(gdf_filtered.geometry, gpd.GeoSeries):
            gdf_filtered = gpd.GeoDataFrame(gdf_filtered, geometry='geometry', crs=gdf.crs)

        return gdf_filtered
    else:
        # Return empty GeoDataFrame with same structure if no nodes within threshold
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)


def _get_adjacent_publics(
    privates,
    publics,
    public_id_col="id",
    private_id_col=None,
    public_geom_col=None,
    buffer=1,
):
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
    if not isinstance(privates, gpd.GeoDataFrame):
        raise TypeError("privates must be a GeoDataFrame")
    if not isinstance(publics, gpd.GeoDataFrame):
        raise TypeError("publics must be a GeoDataFrame")
    if not isinstance(buffer, (int, float)):
        raise TypeError("buffer must be a number")
        
    # Check for empty DataFrames
    if privates.empty:
        warnings.warn("privates GeoDataFrame is empty", RuntimeWarning)
        return {}
    if publics.empty:
        warnings.warn("publics GeoDataFrame is empty", RuntimeWarning)
        return {}
        
    # Check if required columns exist
    if public_id_col not in publics.columns:
        raise ValueError(f"public_id_col '{public_id_col}' not found in publics")
    if private_id_col and private_id_col not in privates.columns:
        raise ValueError(f"private_id_col '{private_id_col}' not found in privates")
    if public_geom_col and public_geom_col not in publics.columns:
        raise ValueError(f"public_geom_col '{public_geom_col}' not found in publics")
        
    # Check geometry types
    if not all(isinstance(geom, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)) 
              for geom in privates.geometry):
        warnings.warn("Some geometries in privates are not Polygons or MultiPolygons", 
                     RuntimeWarning)
    if not all(isinstance(geom, (shapely.geometry.LineString, shapely.geometry.MultiLineString)) 
              for geom in publics.geometry):
        warnings.warn("Some geometries in publics are not LineStrings or MultiLineStrings", 
                     RuntimeWarning)

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
        publics_copy.geometry = publics_copy[public_geom_col]
    
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
    result = (
        joined.groupby(index_col)[public_id_col]
        .apply(lambda x: x.unique().tolist())
        .to_dict()
    )

    return result


def _extract_dual_graph_nodes(dual_graph, id_col, gdf_crs):
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
        dict(dual_graph.nodes(data=True)), orient="index"
    )
    
    if nodes_data.empty:
        return gpd.GeoDataFrame(
            columns=[id_col, "geometry"], geometry="geometry", crs=gdf_crs
        )

    # Check if id_col exists before filtering
    if id_col in nodes_data.columns:
        nodes_data = nodes_data[nodes_data[id_col].notna()]
        
        # Create geometries from coordinates
        nodes_data["geometry"] = [shapely.Point(coord) for coord in nodes_data.index]
        
        # Create a GeoDataFrame
        dual_node_gdf = gpd.GeoDataFrame(nodes_data, geometry="geometry", crs=gdf_crs)
        
        # Set the index of the dual_gdf
        dual_node_gdf = dual_node_gdf.set_index(dual_node_gdf[id_col]).drop(
            columns=[id_col]
        )
        
        return dual_node_gdf
    else:
        return gpd.GeoDataFrame(
            columns=[id_col, "geometry"], geometry="geometry", crs=gdf_crs
        )


def _extract_node_connections(dual_graph, id_col):
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


def _find_additional_connections(line_gdf, id_col, tolerance, connections=None):
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
            ]
        )

    if not endpoints_data:
        return connections
        
    # Create GeoDataFrame of endpoints
    endpoints_gdf = gpd.GeoDataFrame(
        endpoints_data, geometry="endpoint", crs=line_gdf.crs
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
        orig_id_col = (
            f"{id_col}_left" if f"{id_col}_left" in joined.columns else id_col
        )
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


def convert_gdf_to_dual(gdf, id_col=None, tolerance=1e-8):
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
        raise TypeError("Input must be a GeoDataFrame")
    if not isinstance(tolerance, (int, float)):
        raise TypeError("Tolerance must be a number")
    if tolerance < 0:
        raise ValueError("Tolerance must be non-negative")
        
    # Check for empty DataFrame
    if gdf.empty:
        warnings.warn("Input GeoDataFrame is empty", RuntimeWarning)
        return (
            gpd.GeoDataFrame(columns=[id_col] if id_col else [], crs=gdf.crs),
            {}
        )
        
    # Validate geometry types
    invalid_geoms = gdf.geometry.apply(
        lambda g: not isinstance(g, (shapely.geometry.LineString, shapely.geometry.MultiLineString))
    )
    if invalid_geoms.any():
        warnings.warn(
            f"Found {invalid_geoms.sum()} geometries that are not LineString or MultiLineString",
            RuntimeWarning
        )
        
    # Check for null geometries
    if gdf.geometry.isna().any():
        warnings.warn("Found null geometries in input GeoDataFrame", RuntimeWarning)
        gdf = gdf.dropna(subset=['geometry'])

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
    connections = _find_additional_connections(
        line_gdf, id_col, tolerance, connections
    )
    
    return dual_node_gdf, connections


def _create_connecting_lines(
    enclosed_tess,
    public_dual_gdf,
    adjacent_streets,
    private_id_col=None,
    public_id_col=None,
):
    """
    Create a GeoDataFrame of LineStrings connecting the centroid of each tessellation
    polygon to the nodes of adjacent streets.
    
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
        raise TypeError("enclosed_tess must be a GeoDataFrame")
    if not isinstance(public_dual_gdf, gpd.GeoDataFrame):
        raise TypeError("public_dual_gdf must be a GeoDataFrame")
    if not isinstance(adjacent_streets, dict):
        raise TypeError("adjacent_streets must be a dictionary")
        
    # Check for empty inputs
    if enclosed_tess.empty:
        warnings.warn("enclosed_tess GeoDataFrame is empty", RuntimeWarning)
        return gpd.GeoDataFrame(
            {"private_id": [], "public_id": [], "geometry": []},
            crs=enclosed_tess.crs
        )
    if public_dual_gdf.empty:
        warnings.warn("public_dual_gdf GeoDataFrame is empty", RuntimeWarning)
        return gpd.GeoDataFrame(
            {"private_id": [], "public_id": [], "geometry": []},
            crs=enclosed_tess.crs
        )
    if not adjacent_streets:
        warnings.warn("adjacent_streets dictionary is empty", RuntimeWarning)
        return gpd.GeoDataFrame(
            {"private_id": [], "public_id": [], "geometry": []},
            crs=enclosed_tess.crs
        )

    # Check CRS compatibility
    if enclosed_tess.crs != public_dual_gdf.crs:
        warnings.warn("CRS mismatch between enclosed_tess and public_dual_gdf. Converting public_dual_gdf to match enclosed_tess CRS.", RuntimeWarning)
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
            [{"private_id": private_id, "public_id": pid} for pid in public_ids]
        )

    if not connections:
        return gpd.GeoDataFrame(
            {"private_id": [], "public_id": [], "geometry": []}, crs=enclosed_tess.crs
        )

    # Create DataFrame of connections
    connections_df = pd.DataFrame(connections)

    # Compute centroids once
    centroids = enclosed_tess.set_index(private_id_col)["geometry"].centroid

    # Join connections with centroids
    connections_df = connections_df.join(centroids.rename("centroid"), on="private_id")

    # Join with public geometries
    connections_df = connections_df.join(
        public_dual_gdf["geometry"].rename("public_geom"), on="public_id"
    )

    # Filter valid connections and create LineStrings
    valid_df = connections_df.dropna(subset=["centroid", "public_geom"])

    # Create LineStrings vectorized
    valid_df["geometry"] = valid_df.apply(
        lambda row: shapely.LineString(
            [(row.centroid.x, row.centroid.y), (row.public_geom.x, row.public_geom.y)]
        ),
        axis=1,
    )

    # Create final GeoDataFrame
    return gpd.GeoDataFrame(
        valid_df[["private_id", "public_id", "geometry"]],
        geometry="geometry",
        crs=enclosed_tess.crs,
    )


def _prep_contiguity_graph(tess_group, private_id_col, contiguity):
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
        enumerate(
            tess_group[private_id_col] if private_id_col else tess_group.index
        )
    )

    # Create contiguity weights
    if contiguity.lower() == "queen":
        w = libpysal.weights.Queen.from_dataframe(
            tess_group, geom_col="geometry", use_index=False
        )
    elif contiguity.lower() == "rook":
        w = libpysal.weights.Rook.from_dataframe(
            tess_group, geom_col="geometry", use_index=False
        )
    else:
        raise ValueError("contiguity must be 'queen' or 'rook'")

    w_nx = w.to_networkx()

    # Skip if no edges
    if w_nx.number_of_edges() == 0:
        return None, None

    # Compute centroids vectorized
    centroids = np.column_stack(
        [tess_group.geometry.centroid.x, tess_group.geometry.centroid.y]
    )

    # Create positions dictionary
    positions = {node: (x, y) for node, (x, y) in zip(w_nx.nodes(), centroids)}

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
            ]
        )
        for u, v in w_nx.edges()
    }

    nx.set_edge_attributes(w_nx, edge_geometry, "geometry")
    w_nx.graph["approach"] = "primal"

    return w_nx, pos_to_id_mapping


def create_private_to_private(
    private_gdf, private_id_col=None, group_col=None, contiguity="queen"
):
    """
    Create contiguity matrices for each group defined by group_col in the tessellation GeoDataFrame.
    
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
        raise TypeError("private_gdf must be a GeoDataFrame")
        
    # Check for empty DataFrame
    if private_gdf.empty:
        warnings.warn("private_gdf GeoDataFrame is empty", RuntimeWarning)
        return gpd.GeoDataFrame(
            columns=["from_private_id", "to_private_id", group_col or "group", "geometry"],
            geometry="geometry",
            crs=private_gdf.crs
        )
        
    # Validate contiguity parameter
    if contiguity.lower() not in ["queen", "rook"]:
        raise ValueError("contiguity must be 'queen' or 'rook'")
    
    # Check specified columns exist
    if private_id_col is not None and private_id_col not in private_gdf.columns:
        raise ValueError(f"private_id_col '{private_id_col}' not found in private_gdf")
    
    if group_col is not None and group_col not in private_gdf.columns:
        raise ValueError(f"group_col '{group_col}' not found in private_gdf")
    
    # Validate geometry types
    invalid_geoms = private_gdf.geometry.apply(
        lambda g: not isinstance(g, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon))
    )
    if invalid_geoms.any():
        warnings.warn(
            f"Found {invalid_geoms.sum()} geometries that are not Polygon or MultiPolygon",
            RuntimeWarning
        )
    
    # Check for null geometries
    if private_gdf.geometry.isna().any():
        warnings.warn("Found null geometries in private_gdf", RuntimeWarning)
        private_gdf = private_gdf.dropna(subset=["geometry"])
    
    # Handle private ID column
    if private_id_col is None:
        private_gdf = private_gdf.reset_index(drop=True)
        id_series = private_gdf.index
    else:
        id_series = private_gdf[private_id_col]

    # Handle grouping
    if group_col is None:
        tess_groups = {"all": private_gdf.copy()}
    else:
        # More efficient grouping
        tess_groups = {name: group for name, group in private_gdf.groupby(group_col)}

    # List to collect edge DataFrames
    queen_edges_list = []

    for enc, tess_group in tess_groups.items():
        # Prepare contiguity graph for this group
        w_nx, pos_to_id_mapping = _prep_contiguity_graph(
            tess_group, private_id_col, contiguity
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
    else:
        cols = [
            "from_private_id",
            "to_private_id",
            group_col if group_col else "group",
            "geometry",
        ]
        return gpd.GeoDataFrame(columns=cols, crs=private_gdf.crs)


def create_private_to_public(
    private_gdf,
    public_gdf,
    private_id_col=None,
    public_id_col=None,
    public_geom_col=None,
    tolerance=1,
):
    """
    Creates connections between tessellation polygons (private space) and street segments (public space).
    
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
        public_gdf, id_col=public_id_col, tolerance=tolerance
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


def create_public_to_public(public_gdf, public_id_col=None, tolerance=1e-8):
    """
    Create connections between street segments (public-public connections).
    
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
        public_gdf, id_col=public_id_col, tolerance=tolerance
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
                }
            )

    # Create GeoDataFrame from connection data
    if connections_data:
        return gpd.GeoDataFrame(
            connections_data, geometry="geometry", crs=public_dual_gdf.crs
        )
    else:
        return gpd.GeoDataFrame(
            {"from_public_id": [], "to_public_id": [], "geometry": []},
            crs=public_gdf.crs,
        )
