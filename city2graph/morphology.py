"""Network analysis functions for creating and manipulating network graphs."""

import logging
from typing import Any

import geopandas as gpd
import libpysal
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from city2graph.utils import create_tessellation
from city2graph.utils import dual_graph
from city2graph.utils import filter_graph_by_distance
from city2graph.utils import nx_to_gdf

logger = logging.getLogger(__name__)

__all__ = [
    "morphological_graph",
    "private_to_private_graph",
    "private_to_public_graph",
    "public_to_public_graph",
]

# ===============================================================================
# MAIN MORPHOLOGICAL GRAPH FUNCTION
# ===============================================================================

def morphological_graph(
    buildings_gdf: gpd.GeoDataFrame,
    segments_gdf: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    distance: float | None = None,
    private_id_col: str = "tess_id",
    public_id_col: str = "id",
    public_geom_col: str = "barrier_geometry",
    contiguity: str = "queen",
    keep_buildings: bool = False,
) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]:
    """Create morphological network from buildings and road segments."""
    _validate_inputs(buildings_gdf, "buildings_gdf")
    _validate_inputs(segments_gdf, "segments_gdf")

    # Ensure compatible CRS
    if segments_gdf.crs != buildings_gdf.crs:
        segments_gdf = segments_gdf.to_crs(buildings_gdf.crs)

    # Prepare barriers and create tessellation
    barrier_gdf = _prepare_barriers(segments_gdf, public_geom_col)
    tess = create_tessellation(buildings_gdf, primary_barriers=barrier_gdf)

    # Filter segments by distance if specified
    if center_point is not None and distance is not None:
        segments_subset = filter_graph_by_distance(segments_gdf, center_point, distance)
    else:
        segments_subset = segments_gdf.copy()

    # Filter tessellation and optionally add building info
    tess_subset = _filter_adjacent_tessellation(tess, segments_subset)
    tess_final = (
        _add_building_info(tess_subset, buildings_gdf) if keep_buildings else tess_subset
    )

    # Create graph components
    private_private = private_to_private_graph(
        tess_subset,
        private_id_col=private_id_col,
        group_col="enclosure_index",
        contiguity=contiguity,
    )

    public_public = public_to_public_graph(segments_subset, public_id_col=public_id_col)

    private_public = private_to_public_graph(
        tess_subset,
        segments_subset,
        private_id_col=private_id_col,
        public_id_col=public_id_col,
        public_geom_col=public_geom_col,
    )

    # Prepare output
    nodes_dict = {
        "private": _set_index_if_exists(tess_final, private_id_col),
        "public": _set_index_if_exists(segments_subset, public_id_col),
    }

    edges_dict = {
        ("private", "touched_to", "private"): _set_edge_index(
            private_private, "from_private_id", "to_private_id",
        ),
        ("public", "connected_to", "public"): _set_edge_index(
            public_public, "from_public_id", "to_public_id",
        ),
        ("private", "faced_to", "public"): _set_edge_index(
            private_public, "private_id", "public_id",
        ),
    }

    return nodes_dict, edges_dict


# ===============================================================================
# PRIVATE TO PRIVATE GRAPH
# ===============================================================================

def private_to_private_graph(
    private_gdf: gpd.GeoDataFrame,
    private_id_col: str | None = None,
    group_col: str | None = None,
    contiguity: str = "queen",
) -> gpd.GeoDataFrame:
    """Create contiguity-based connections between private spaces."""
    _validate_inputs(private_gdf, "private_gdf")

    if private_gdf.empty:
        return _empty_edges_gdf(
            private_gdf.crs,
            "from_private_id",
            "to_private_id",
            [group_col or "group"],
        )

    if contiguity.lower() not in ["queen", "rook"]:
        msg = "contiguity must be 'queen' or 'rook'"
        raise ValueError(msg)

    # Ensure ID column
    private_gdf, private_id_col = _ensure_id_column(private_gdf, private_id_col)

    # Handle grouping
    if group_col is None:
        groups = {"all": private_gdf}
        group_col = "group"
    else:
        if group_col not in private_gdf.columns:
            msg = f"group_col '{group_col}' not found"
            raise ValueError(msg)
        groups = {
            group_name: group_gdf
            for group_name, group_gdf in private_gdf.groupby(group_col)
        }

    # Create edges for each group
    all_edges = []
    for group_name, group_gdf in groups.items():
        if len(group_gdf) < 2:
            continue

        edges = _create_contiguity_edges(
            group_gdf, private_id_col, contiguity, group_name, group_col,
        )
        if not edges.empty:
            all_edges.append(edges)

    if all_edges:
        return gpd.GeoDataFrame(pd.concat(all_edges, ignore_index=True))

    return _empty_edges_gdf(
        private_gdf.crs, "from_private_id", "to_private_id", [group_col],
    )


def _create_contiguity_edges(
    gdf: gpd.GeoDataFrame,
    id_col: str,
    contiguity: str,
    group_name: str,
    group_col: str,
) -> gpd.GeoDataFrame:
    """Create contiguity edges for a group."""
    # Create spatial weights
    if contiguity.lower() == "queen":
        w = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    else:
        w = libpysal.weights.Rook.from_dataframe(gdf, use_index=False)

    if w.n_components == 0 or len(w.neighbors) == 0:
        return gpd.GeoDataFrame()

    # Convert to NetworkX
    w_nx = w.to_networkx()
    if w_nx.number_of_edges() == 0:
        return gpd.GeoDataFrame()

    # Set positions for geometry creation
    centroids = gdf.geometry.centroid
    positions = {i: (centroids.iloc[i].x, centroids.iloc[i].y) for i in range(len(gdf))}
    nx.set_node_attributes(w_nx, positions, "pos")

    # Create edge geometries
    edge_geoms = {}
    for u, v in w_nx.edges():
        edge_geoms[(u, v)] = LineString([positions[u], positions[v]])
    nx.set_edge_attributes(w_nx, edge_geoms, "geometry")
    w_nx.graph["crs"] = gdf.crs

    # Convert to GeoDataFrame
    _, edges_gdf = nx_to_gdf(w_nx)

    if edges_gdf.empty:
        return gpd.GeoDataFrame()

    # Map indices to actual IDs
    id_mapping = gdf[id_col].to_dict()

    if isinstance(edges_gdf.index, pd.MultiIndex):
        edges_gdf["from_private_id"] = edges_gdf.index.get_level_values(0).map(id_mapping)
        edges_gdf["to_private_id"] = edges_gdf.index.get_level_values(1).map(id_mapping)
    elif "node_start" in edges_gdf.columns and "node_end" in edges_gdf.columns:
        edges_gdf["from_private_id"] = edges_gdf["node_start"].map(id_mapping)
        edges_gdf["to_private_id"] = edges_gdf["node_end"].map(id_mapping)
    else:
        logger.warning("Could not extract edge connections")
        return gpd.GeoDataFrame()

    edges_gdf[group_col] = group_name
    return edges_gdf[["from_private_id", "to_private_id", group_col, "geometry"]].reset_index(drop=True)


# ===============================================================================
# PRIVATE TO PUBLIC GRAPH
# ===============================================================================

def private_to_public_graph(
    private_gdf: gpd.GeoDataFrame,
    public_gdf: gpd.GeoDataFrame,
    private_id_col: str | None = None,
    public_id_col: str | None = None,
    public_geom_col: str | None = None,
    tolerance: float = 1.0,
) -> gpd.GeoDataFrame:
    """Create connections between private and public spaces."""
    _validate_inputs(private_gdf, "private_gdf")
    _validate_inputs(public_gdf, "public_gdf")

    if private_gdf.empty or public_gdf.empty:
        return _empty_edges_gdf(private_gdf.crs, "private_id", "public_id")

    # Ensure compatible CRS
    public_gdf = _ensure_crs_match(private_gdf, public_gdf)

    # Ensure ID columns
    private_gdf, private_id_col = _ensure_id_column(private_gdf, private_id_col)
    public_gdf, public_id_col = _ensure_id_column(public_gdf, public_id_col)

    # Create dual graph for public spaces
    try:
        public_dual_gdf, _ = dual_graph(public_gdf, id_col=public_id_col, tolerance=tolerance)
    except (ValueError, TypeError) as e:
        logger.warning("Failed to create dual graph: %s", e)
        return _empty_edges_gdf(private_gdf.crs, "private_id", "public_id")

    if public_dual_gdf.empty:
        return _empty_edges_gdf(private_gdf.crs, "private_id", "public_id")

    # Find adjacencies
    adjacency = _find_adjacencies(
        private_gdf, public_gdf, private_id_col, public_id_col, public_geom_col, tolerance,
    )

    if not adjacency:
        return _empty_edges_gdf(private_gdf.crs, "private_id", "public_id")

    # Create connections
    return _create_connections(private_gdf, public_dual_gdf, adjacency, private_id_col, public_id_col)


def _find_adjacencies(
    private_gdf: gpd.GeoDataFrame,
    public_gdf: gpd.GeoDataFrame,
    private_id_col: str,
    public_id_col: str,
    public_geom_col: str | None,
    buffer_dist: float,
) -> dict:
    """Find adjacent public spaces for each private space."""
    # Prepare public geometries
    public_for_join = public_gdf.copy()

    if public_geom_col and public_geom_col in public_gdf.columns:
        public_for_join.geometry = public_gdf[public_geom_col]

    # Buffer for intersection
    public_for_join.geometry = public_for_join.geometry.buffer(buffer_dist)

    # Spatial join
    joined = gpd.sjoin(public_for_join, private_gdf, how="inner", predicate="intersects")

    if joined.empty:
        return {}

    # Group by private space
    private_col = "index_right" if private_id_col == "id" else private_id_col
    return (joined.groupby(private_col)[public_id_col]
                  .apply(lambda x: x.unique().tolist())
                  .to_dict())


def _create_connections(
    private_gdf: gpd.GeoDataFrame,
    public_dual_gdf: gpd.GeoDataFrame,
    adjacency: dict,
    private_id_col: str,
    public_id_col: str,
) -> gpd.GeoDataFrame:
    """Create LineString connections between private and public spaces."""
    # Build connections list
    connections = []
    for private_id, public_ids in adjacency.items():
        connections.extend([(private_id, public_id) for public_id in public_ids])

    if not connections:
        return _empty_edges_gdf(private_gdf.crs, "private_id", "public_id")

    # Convert to DataFrame
    conn_df = pd.DataFrame(connections, columns=["private_id", "public_id"])

    # Get geometries
    private_centroids = private_gdf.set_index(private_id_col).geometry.centroid
    if public_id_col in public_dual_gdf.columns:
        public_geoms = public_dual_gdf.set_index(public_id_col).geometry
    else:
        public_geoms = public_dual_gdf.geometry

    # Join geometries
    conn_df = conn_df.join(private_centroids.rename("private_geom"), on="private_id")
    conn_df = conn_df.join(public_geoms.rename("public_geom"), on="public_id")

    # Filter valid connections
    valid_df = conn_df.dropna(subset=["private_geom", "public_geom"]).copy()

    if valid_df.empty:
        return _empty_edges_gdf(private_gdf.crs, "private_id", "public_id")

    # Create LineString geometries
    geometries = []
    for _, row in valid_df.iterrows():
        p1 = row["private_geom"]
        p2 = row["public_geom"]
        geometries.append(LineString([(p1.x, p1.y), (p2.x, p2.y)]))

    valid_df["geometry"] = geometries

    return gpd.GeoDataFrame(
        valid_df[["private_id", "public_id", "geometry"]],
        geometry="geometry",
        crs=private_gdf.crs,
    )


# ===============================================================================
# PUBLIC TO PUBLIC GRAPH
# ===============================================================================

def public_to_public_graph(
    public_gdf: gpd.GeoDataFrame,
    public_id_col: str | None = None,
    tolerance: float = 1e-8,
) -> gpd.GeoDataFrame:
    """Create connections between public spaces using dual graph."""
    _validate_inputs(public_gdf, "public_gdf")

    if public_gdf.empty:
        return _empty_edges_gdf(public_gdf.crs, "from_public_id", "to_public_id")

    # Ensure ID column
    public_gdf, public_id_col = _ensure_id_column(public_gdf, public_id_col)

    # Create dual graph
    try:
        public_dual_gdf, connections_dict = dual_graph(
            public_gdf, id_col=public_id_col, tolerance=tolerance,
        )
    except (ValueError, TypeError) as e:
        logger.warning("Failed to create dual graph: %s", e)
        return _empty_edges_gdf(public_gdf.crs, "from_public_id", "to_public_id")

    if public_dual_gdf.empty or not connections_dict:
        return _empty_edges_gdf(public_gdf.crs, "from_public_id", "to_public_id")

    # Convert connections to edges
    edges_list = []
    for from_id, to_ids in connections_dict.items():
        edges_list.extend([(from_id, to_id) for to_id in to_ids])

    if not edges_list:
        return _empty_edges_gdf(public_gdf.crs, "from_public_id", "to_public_id")

    # Create DataFrame and remove duplicates
    edges_df = pd.DataFrame(edges_list, columns=["from_public_id", "to_public_id"])

    # Remove duplicate undirected edges
    edge_keys = np.sort(edges_df[["from_public_id", "to_public_id"]].values, axis=1)
    edges_df["edge_key"] = [tuple(row) for row in edge_keys]
    edges_df = edges_df.drop_duplicates(subset=["edge_key"]).drop(columns=["edge_key"])

    # Get node coordinates and filter valid connections
    node_coords = public_dual_gdf.geometry.apply(lambda p: (p.x, p.y)).to_dict()
    valid_mask = (
        edges_df["from_public_id"].isin(node_coords) &
        edges_df["to_public_id"].isin(node_coords)
    )
    valid_edges = edges_df[valid_mask].copy()

    if valid_edges.empty:
        return _empty_edges_gdf(public_gdf.crs, "from_public_id", "to_public_id")

    # Create geometries
    geometries = []
    for _, row in valid_edges.iterrows():
        from_coord = node_coords[row["from_public_id"]]
        to_coord = node_coords[row["to_public_id"]]
        geometries.append(LineString([from_coord, to_coord]))

    valid_edges["geometry"] = geometries

    return gpd.GeoDataFrame(
        valid_edges[["from_public_id", "to_public_id", "geometry"]],
        geometry="geometry",
        crs=public_dual_gdf.crs,
    )


# ===============================================================================
# HELPER FUNCTIONS
# ===============================================================================

def _prepare_barriers(segments_gdf: gpd.GeoDataFrame, public_geom_col: str) -> gpd.GeoDataFrame:
    """Prepare barrier geometry for tessellation."""
    if public_geom_col not in segments_gdf.columns:
        logger.warning("Column '%s' not found, using 'geometry'", public_geom_col)
        return segments_gdf.copy()

    if public_geom_col == "geometry":
        return segments_gdf.copy()

    return gpd.GeoDataFrame(
        segments_gdf.drop(columns=["geometry"]),
        geometry=segments_gdf[public_geom_col],
        crs=segments_gdf.crs,
    )


def _filter_adjacent_tessellation(
    tess: gpd.GeoDataFrame, segments: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Filter tessellation to cells adjacent to segments."""
    if tess.empty or segments.empty:
        return tess.copy()

    adjacent = gpd.sjoin(tess, segments, how="inner", predicate="intersects")

    if "enclosure_index" in tess.columns and "enclosure_index" in adjacent.columns:
        # For enclosed tessellation
        enclosure_indices = adjacent["enclosure_index"].unique()
        return tess[tess["enclosure_index"].isin(enclosure_indices)]

    # For morphological tessellation
    tess_indices = adjacent.index.unique()
    return tess[tess.index.isin(tess_indices)]


def _add_building_info(
    tess_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add building information to tessellation."""
    if buildings_gdf.empty:
        return tess_gdf.copy()

    joined = gpd.sjoin(tess_gdf, buildings_gdf, how="left", predicate="intersects")

    if "index_right" in joined.columns:
        building_geom_map = buildings_gdf.geometry.to_dict()
        joined["building_geometry"] = joined["index_right"].map(building_geom_map)
        joined["building_geometry"] = gpd.GeoSeries(
            joined["building_geometry"], crs=buildings_gdf.crs,
        )
        joined = joined.drop(columns=["index_right"])

    return joined


def _set_edge_index(gdf: gpd.GeoDataFrame, from_col: str, to_col: str) -> gpd.GeoDataFrame:
    """Set MultiIndex on edge GeoDataFrame if columns exist."""
    if not gdf.empty and from_col in gdf.columns and to_col in gdf.columns:
        return gdf.set_index([from_col, to_col])
    return gdf

def _ensure_id_column(gdf: gpd.GeoDataFrame, id_col: str | None) -> tuple[gpd.GeoDataFrame, str]:
    """Ensure GeoDataFrame has ID column."""
    if gdf.empty:
        return gdf, id_col or "id"

    if id_col is None or id_col not in gdf.columns:
        gdf_copy = gdf.reset_index(drop=True)
        gdf_copy["id"] = gdf_copy.index
        return gdf_copy, "id"

    return gdf.copy(), id_col


def _validate_inputs(gdf: gpd.GeoDataFrame, name: str) -> None:
    """Validate GeoDataFrame inputs."""
    if not isinstance(gdf, gpd.GeoDataFrame):
        msg = f"{name} must be a GeoDataFrame"
        raise TypeError(msg)
    if gdf.empty:
        logger.warning("%s is empty", name)


def _ensure_crs_match(target: gpd.GeoDataFrame, source: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure source has same CRS as target."""
    if source.crs != target.crs:
        logger.warning("Converting CRS from %s to %s", source.crs, target.crs)
        return source.to_crs(target.crs)
    return source


def _empty_edges_gdf(crs: Any,
                     from_col: str,
                     to_col: str,
                     extra_cols: list | None = None) -> gpd.GeoDataFrame:
    """Create empty edges GeoDataFrame."""
    cols = [from_col, to_col] + (extra_cols or []) + ["geometry"]
    return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=crs)


def _set_index_if_exists(gdf: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
    """Set index if column exists."""
    return gdf.set_index(col) if col in gdf.columns else gdf
