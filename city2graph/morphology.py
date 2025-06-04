"""Network analysis functions for creating and manipulating network graphs."""

import logging

import geopandas as gpd
import libpysal
import momepy
import networkx as nx
import numpy as np
import pandas as pd
import shapely

from city2graph.utils import create_tessellation
from city2graph.utils import dual_graph
from city2graph.utils import filter_graph_by_distance

# Set up logger for this module
logger = logging.getLogger(__name__)

# Define the public API for this module
__all__ = [
    "morphological_graph",
    "private_to_private_graph",
    "private_to_public_graph",
    "public_to_public_graph",
]


# ===============================================================================
# VALIDATION FUNCTIONS
# ===============================================================================

def _validate_inputs(
    privates: gpd.GeoDataFrame,
    publics: gpd.GeoDataFrame,
    buffer: float,
) -> None:
    """Validate input parameters for spatial operations."""
    if not isinstance(privates, gpd.GeoDataFrame):
        msg = "privates must be a GeoDataFrame"
        raise TypeError(msg)
    if not isinstance(publics, gpd.GeoDataFrame):
        msg = "publics must be a GeoDataFrame"
        raise TypeError(msg)
    if not isinstance(buffer, (int, float)):
        msg = "buffer must be a number"
        raise TypeError(msg)


def _check_empty_dataframes(
    privates: gpd.GeoDataFrame,
    publics: gpd.GeoDataFrame,
) -> bool:
    """Check if DataFrames are empty and warn accordingly."""
    if privates.empty:
        logger.warning("privates GeoDataFrame is empty")
        return True
    if publics.empty:
        logger.warning("publics GeoDataFrame is empty")
        return True
    return False


def _validate_columns(
    publics: gpd.GeoDataFrame,
    privates: gpd.GeoDataFrame,
    public_id_col: str,
    private_id_col: str | None,
    public_geom_col: str | None,
) -> None:
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


def _validate_geometries(
    privates: gpd.GeoDataFrame,
    publics: gpd.GeoDataFrame,
) -> None:
    """Validate geometry types in DataFrames."""
    # Check private space geometries (should be polygons)
    invalid_private_geoms = ~privates.geometry.apply(
        lambda g: isinstance(g, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)),
    )
    if invalid_private_geoms.any():
        logger.warning(
            "Found %d private geometries that are not Polygons or MultiPolygons",
            invalid_private_geoms.sum(),
        )

    # Check public space geometries (should be linestrings)
    invalid_public_geoms = ~publics.geometry.apply(
        lambda g: isinstance(g, (shapely.geometry.LineString, shapely.geometry.MultiLineString)),
    )
    if invalid_public_geoms.any():
        logger.warning(
            "Found %d public geometries that are not LineStrings or MultiLineStrings",
            invalid_public_geoms.sum(),
        )


def _ensure_crs_compatibility(
    target_gdf: gpd.GeoDataFrame,
    source_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Ensure source GeoDataFrame has same CRS as target, transforming if necessary."""
    if source_gdf.crs != target_gdf.crs:
        logger.warning(
            "CRS mismatch detected. Converting from %s to %s",
            source_gdf.crs, target_gdf.crs,
        )
        return source_gdf.to_crs(target_gdf.crs)
    return source_gdf


# ===============================================================================
# SPATIAL ANALYSIS FUNCTIONS
# ===============================================================================

def _prepare_id_columns(
    privates: gpd.GeoDataFrame,
    publics: gpd.GeoDataFrame,
    private_id_col: str | None,
    public_id_col: str | None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, str, str]:
    """Prepare ID columns for spatial operations."""
    privates_copy = privates.copy()
    publics_copy = publics.copy()

    if private_id_col is None:
        privates_copy = privates_copy.reset_index(drop=True)
        private_id_col = "index"
        privates_copy[private_id_col] = privates_copy.index

    if public_id_col is None:
        publics_copy = publics_copy.reset_index(drop=True)
        public_id_col = "index"
        publics_copy[public_id_col] = publics_copy.index

    return privates_copy, publics_copy, private_id_col, public_id_col


def _prepare_buffered_geometry(
    publics: gpd.GeoDataFrame,
    public_geom_col: str | None,
    buffer: float,
) -> gpd.GeoDataFrame:
    """Prepare buffered public geometries for spatial operations."""
    publics_copy = publics.copy()

    # Use alternative geometry column if specified
    if public_geom_col is not None:
        publics_copy.geometry = gpd.GeoSeries(
            publics_copy[public_geom_col], crs=publics_copy.crs,
        )

    # Create buffered geometries for intersection test
    publics_copy.geometry = publics_copy.geometry.buffer(buffer)
    return publics_copy


def _get_adjacent_publics(
    privates: gpd.GeoDataFrame,
    publics: gpd.GeoDataFrame,
    public_id_col: str = "id",
    private_id_col: str | None = None,
    public_geom_col: str | None = None,
    buffer: float = 1,
) -> dict[str, list[str]]:
    """
    Identify streets that are contained in or adjacent to each polygon.

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

    # Ensure CRS compatibility
    publics = _ensure_crs_compatibility(privates, publics)

    # Prepare ID columns
    privates_prep, publics_prep, private_id_final, public_id_final = _prepare_id_columns(
        privates, publics, private_id_col, public_id_col,
    )

    # Prepare buffered geometries
    publics_buffered = _prepare_buffered_geometry(publics_prep, public_geom_col, buffer)

    # Perform spatial join
    joined = gpd.sjoin(
        publics_buffered,
        privates_prep,
        how="inner",
        predicate="intersects",
    )

    # Group by private_id and collect public ids
    index_col = "index_right" if private_id_final == "index" else private_id_final
    return (
        joined.groupby(index_col)[public_id_final]
        .apply(lambda x: x.unique().tolist())
        .to_dict()
    )



def _validate_connecting_lines_inputs(
    enclosed_tess: gpd.GeoDataFrame,
    public_dual_gdf: gpd.GeoDataFrame,
    adjacent_streets: dict,
) -> None:
    """Validate inputs for creating connecting lines."""
    if not isinstance(enclosed_tess, gpd.GeoDataFrame):
        msg = "enclosed_tess must be a GeoDataFrame"
        raise TypeError(msg)
    if not isinstance(public_dual_gdf, gpd.GeoDataFrame):
        msg = "public_dual_gdf must be a GeoDataFrame"
        raise TypeError(msg)
    if not isinstance(adjacent_streets, dict):
        msg = "adjacent_streets must be a dictionary"
        raise TypeError(msg)


def _check_empty_connecting_inputs(
    enclosed_tess: gpd.GeoDataFrame,
    public_dual_gdf: gpd.GeoDataFrame,
    adjacent_streets: dict,
) -> gpd.GeoDataFrame | None:
    """Check for empty inputs and return empty GeoDataFrame if found."""
    empty_gdf = gpd.GeoDataFrame(
        {"private_id": [], "public_id": [], "geometry": []},
        crs=enclosed_tess.crs,
    )

    if enclosed_tess.empty:
        logger.warning("enclosed_tess GeoDataFrame is empty")
        return empty_gdf
    if public_dual_gdf.empty:
        logger.warning("public_dual_gdf GeoDataFrame is empty")
        return empty_gdf
    if not adjacent_streets:
        logger.warning("adjacent_streets dictionary is empty")
        return empty_gdf

    return None


def _prepare_id_columns_for_connections(
    enclosed_tess: gpd.GeoDataFrame,
    public_dual_gdf: gpd.GeoDataFrame,
    private_id_col: str | None,
    public_id_col: str | None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, str, str]:
    """Prepare ID columns for connection creation."""
    enclosed_tess_copy = enclosed_tess.copy()
    public_dual_copy = public_dual_gdf.copy()

    # Handle private ID column
    if private_id_col is None:
        enclosed_tess_copy = enclosed_tess_copy.reset_index(drop=True)
        private_id_col = "index"
        enclosed_tess_copy[private_id_col] = enclosed_tess_copy.index

    # Handle public ID column
    if public_id_col is None:
        public_dual_copy = public_dual_copy.reset_index(drop=True)
        public_id_col = "index"
        public_dual_copy[public_id_col] = public_dual_copy.index

    return enclosed_tess_copy, public_dual_copy, private_id_col, public_id_col


def _create_connecting_lines(
    enclosed_tess: gpd.GeoDataFrame,
    public_dual_gdf: gpd.GeoDataFrame,
    adjacent_streets: dict,
    private_id_col: str | None = None,
    public_id_col: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Create LineStrings connecting tessellation centroids to adjacent street nodes.

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
    _validate_connecting_lines_inputs(enclosed_tess, public_dual_gdf, adjacent_streets)

    # Check for empty inputs
    empty_result = _check_empty_connecting_inputs(enclosed_tess, public_dual_gdf, adjacent_streets)
    if empty_result is not None:
        return empty_result

    # Check CRS compatibility
    if enclosed_tess.crs != public_dual_gdf.crs:
        logger.warning(
            "CRS mismatch between enclosed_tess and public_dual_gdf. "
            "Converting public_dual_gdf to match enclosed_tess CRS.",
        )
        public_dual_gdf = public_dual_gdf.to_crs(enclosed_tess.crs)

    # Prepare ID columns for tessellation
    enclosed_tess_prep = enclosed_tess.copy()
    if private_id_col is None:
        enclosed_tess_prep = enclosed_tess_prep.reset_index(drop=True)
        private_id_col = "index"
        enclosed_tess_prep[private_id_col] = enclosed_tess_prep.index

    # Convert the adjacency dictionary to a DataFrame for vectorized operations
    connections = []
    for private_id, public_ids in adjacent_streets.items():
        connections.extend(
            [{"private_id": private_id, "public_id": pid} for pid in public_ids],
        )

    if not connections:
        return gpd.GeoDataFrame(
            {"private_id": [], "public_id": [], "geometry": []},
            crs=enclosed_tess.crs,
        )

    # Create DataFrame of connections
    connections_df = pd.DataFrame(connections)

    # Compute centroids once
    centroids = enclosed_tess_prep.set_index(private_id_col)["geometry"].centroid

    # Join connections with centroids
    connections_df = connections_df.join(centroids.rename("centroid"), on="private_id")

    # For dual graphs, the index typically corresponds to the original segment IDs
    # Create a mapping from public_id to dual graph geometry
    if public_id_col and public_id_col in public_dual_gdf.columns:
        # If the dual graph has the ID column, use it
        public_geom_series = public_dual_gdf.set_index(public_id_col)["geometry"]
    else:
        # Otherwise, assume the index corresponds to the public IDs
        public_geom_series = public_dual_gdf["geometry"]
        # Reset index to make it accessible for joining
        public_geom_series.index = public_dual_gdf.index

    # Join with public geometries
    connections_df = connections_df.join(
        public_geom_series.rename("public_geom"),
        on="public_id",
    )

    # Filter valid connections and create LineStrings
    valid_df = connections_df.dropna(subset=["centroid", "public_geom"])

    if valid_df.empty:
        return gpd.GeoDataFrame(
            {"private_id": [], "public_id": [], "geometry": []},
            crs=enclosed_tess.crs,
        )

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


def private_to_private_graph(
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
        logger.warning("private_gdf GeoDataFrame is empty")
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
        logger.warning(
            "Found %d geometries that are not Polygon or MultiPolygon",
            invalid_geoms.sum(),
        )

    # Check for null geometries
    if private_gdf.geometry.isna().any():
        logger.warning("Found null geometries in private_gdf")
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


def private_to_public_graph(
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
    # Validate and filter public_gdf geometry types before creating dual graph
    if not public_gdf.geometry.apply(
        lambda g: isinstance(g, (shapely.geometry.LineString, shapely.geometry.MultiLineString)),
    ).all():
        logger.warning(
            "Some geometries in public_gdf are not LineString or MultiLineString. "
            "Filtering to valid geometries only.",
        )

    # Check if we have any valid geometries left
    if public_gdf.empty:
        return gpd.GeoDataFrame(
            {"private_id": [], "public_id": [], "geometry": []},
            crs=private_gdf.crs if hasattr(private_gdf, "crs") else None,
        )

    # Create dual graph from publics
    public_dual_gdf, public_to_public_dict = dual_graph(
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


def public_to_public_graph(
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
    # Validate and filter public_gdf geometry types before creating dual graph
    if not public_gdf.geometry.apply(
        lambda g: isinstance(g, (shapely.geometry.LineString, shapely.geometry.MultiLineString)),
    ).all():
        logger.warning(
            "Some geometries in public_gdf are not LineString or MultiLineString. "
            "Filtering to valid geometries only.",
        )

    # Check if we have any valid geometries left
    if public_gdf.empty:
        return gpd.GeoDataFrame(
            {"from_public_id": [], "to_public_id": [], "geometry": []},
            crs=public_gdf.crs,
        )

    # Create dual representation
    public_dual_gdf, public_to_public_dict = dual_graph(
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


# ===============================================================================
# MAIN MORPHOLOGICAL GRAPH FUNCTIONS
# ===============================================================================

def _prepare_barrier_geometry(
    segments_gdf: gpd.GeoDataFrame,
    public_geom_col: str,
) -> tuple[gpd.GeoDataFrame, str]:
    """Prepare barrier geometry for tessellation creation."""
    # Check if public_geom_col exists in segments_gdf
    if public_geom_col not in segments_gdf.columns:
        logger.warning(
            "Column '%s' not found in segments_gdf. Using 'geometry' column instead.",
            public_geom_col,
        )
        public_geom_col = "geometry"

    # Create a GeoDataFrame with the barrier geometry for tessellation
    if public_geom_col == "geometry":
        # If using the main geometry column, just use the segments as-is
        barrier_gdf = segments_gdf.copy()
    else:
        # Create a completely new GeoDataFrame with the barrier geometry
        barrier_gdf = gpd.GeoDataFrame(
            segments_gdf.drop(columns=["geometry"]),
            geometry=segments_gdf[public_geom_col].values,
            crs=segments_gdf.crs,
        )

    return barrier_gdf, public_geom_col


def _filter_tessellation_by_adjacency(
    enclosed_tess: gpd.GeoDataFrame,
    segments_subset: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Filter tessellation to only include cells adjacent to segments."""
    if enclosed_tess.empty or segments_subset.empty:
        return enclosed_tess.copy()

    # Get the enclosure indices that are adjacent to the segments using a spatial join
    adjacent_enclosure_indices = gpd.sjoin(
        enclosed_tess, segments_subset, how="inner", predicate="intersects",
    )

    # Check if the tessellation has an enclosure_index column (from enclosed tessellation)
    # If not, use the tessellation index as the enclosure index
    if "enclosure_index" in adjacent_enclosure_indices.columns:
        enclosure_indices = adjacent_enclosure_indices["enclosure_index"].unique()
        # Get enclosed tessellation for the adjacent enclosures
        if "enclosure_index" in enclosed_tess.columns:
            return enclosed_tess[enclosed_tess["enclosure_index"].isin(enclosure_indices)]
    else:
        # For morphological tessellation, use the tessellation index
        enclosure_indices = adjacent_enclosure_indices.index.unique()
        # For morphological tessellation, filter by index
        return enclosed_tess[enclosed_tess.index.isin(enclosure_indices)]

    return enclosed_tess.copy()


def _prepare_tessellation_with_buildings(
    tess_subset: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
    keep_buildings: bool,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Prepare tessellation data, optionally adding building information."""
    if keep_buildings:
        tess_with_buildings = _add_buildings_to_tessellation(tess_subset, buildings_gdf)
        # Use the original tessellation for graph operations to avoid column conflicts
        tess_for_graphs = tess_subset
        # But use the enhanced tessellation for the final output
        return tess_with_buildings, tess_for_graphs
    return tess_subset, tess_subset


def morphological_graph(
    buildings_gdf: gpd.GeoDataFrame,
    segments_gdf: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame = None,
    distance: float | None = None,
    private_id_col: str = "tess_id",
    public_id_col: str = "id",
    public_geom_col: str = "barrier_geometry",
    contiguity: str = "queen",
    keep_buildings: bool = False,
) -> dict:
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
    contiguity : str, default='queen'
        Type of contiguity for private-to-private connections, either 'queen' or 'rook'.
    keep_buildings : bool, default=False
        If True, performs a spatial left join between tessellation and buildings to include
        building attributes in the tessellation data. Building index columns are dropped.

    Returns
    -------
    dict
        Dictionary containing the following elements:
        - 'nodes': Dictionary with 'private' and 'public' GeoDataFrames
        - 'edges': Dictionary with edge relationships between node types

    Notes
    -----
    - If center_point is not provided, all segments will be included.
    - The barrier_geometry column should contain the processed geometries for road segments.
      If it doesn't exist, the function will use the column specified in public_geom_col.
    - The function requires the city2graph package and depends on create_tessellation,
      filter_graph_by_distance, create_private_to_private, create_public_to_public,
      and create_private_to_public functions.
    """
    # Prepare barrier geometry for tessellation
    barrier_gdf, public_geom_col = _prepare_barrier_geometry(segments_gdf, public_geom_col)

    # Ensure barrier_gdf has the same CRS as buildings_gdf
    if barrier_gdf.crs != buildings_gdf.crs:
        barrier_gdf = barrier_gdf.to_crs(buildings_gdf.crs)

    # Create tessellations based on buildings and road barrier geometries
    enclosed_tess = create_tessellation(buildings_gdf, primary_barriers=barrier_gdf)

    # Optionally filter the network by distance from a specified center point
    if center_point is not None and distance is not None:
        segments_subset = filter_graph_by_distance(
            segments_gdf, center_point, distance=distance,
        )
    else:
        segments_subset = segments_gdf.copy()

    # Filter tessellation to only include cells adjacent to segments
    tess_subset = _filter_tessellation_by_adjacency(enclosed_tess, segments_subset)

    # Prepare tessellation data, optionally adding building information
    tess_final, tess_for_graphs = _prepare_tessellation_with_buildings(
        tess_subset, buildings_gdf, keep_buildings,
    )

    # Create connections between adjacent private spaces (tessellation cells)
    private_to_private = private_to_private_graph(
        tess_for_graphs,
        private_id_col=private_id_col,
        group_col="enclosure_index",
        contiguity=contiguity,
    )

    # Create connections between street segments
    public_to_public = public_to_public_graph(
        segments_subset,
        public_id_col=public_id_col,
    )

    # Create connections between private spaces and public spaces
    private_to_public = private_to_public_graph(
        tess_for_graphs,
        segments_subset,
        private_id_col=private_id_col,
        public_id_col=public_id_col,
        public_geom_col=public_geom_col,
    )

    # Return structure compatible with pyg_to_gdf expectations
    return {
        "nodes": {
            "private": tess_final,
            "public": segments_subset,
        },
        "edges": {
            ("private", "touched_to", "private"): private_to_private,
            ("public", "connected_to", "public"): public_to_public,
            ("private", "faced_to", "public"): private_to_public,
        },
    }


# ===============================================================================
# BUILDING INTEGRATION FUNCTIONS
# ===============================================================================

def _add_buildings_to_tessellation(
    tessellation_gdf: gpd.GeoDataFrame,
    buildings_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Add building information to tessellation cells using spatial left join.

    Parameters
    ----------
    tessellation_gdf : gpd.GeoDataFrame
        Tessellation cells to add building information to
    buildings_gdf : gpd.GeoDataFrame
        Buildings to spatially join with tessellation cells

    Returns
    -------
    gpd.GeoDataFrame
        Tessellation with building information joined, including building_geometry column
    """
    if buildings_gdf.empty:
        return tessellation_gdf.copy()

    # Perform spatial left join with buildings
    joined = gpd.sjoin(
        tessellation_gdf,
        buildings_gdf,
        how="left",
        predicate="intersects",
    )

    # Add building_geometry column as a separate GeoSeries
    if "index_right" in joined.columns:
        # Create mapping from building index to geometry
        building_geom_map = buildings_gdf["geometry"].to_dict()

        # Map building geometries to tessellation cells
        joined["building_geometry"] = joined["index_right"].map(building_geom_map)

        # Convert to GeoSeries with proper CRS
        joined["building_geometry"] = gpd.GeoSeries(
            joined["building_geometry"],
            crs=buildings_gdf.crs,
        )

        # Remove the index_right column to prevent conflicts
        joined = joined.drop(columns=["index_right"])

    return joined
