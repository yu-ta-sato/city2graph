import logging
import warnings
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import geopandas as gpd
import libpysal
import networkx as nx
from shapely.geometry import LineString

from .utils import create_tessellation, dual_graph, filter_graph_by_distance

__all__ = [
    "morphological_graph",
    "private_to_private_graph",
    "private_to_public_graph",
    "public_to_public_graph",
]

logger = logging.getLogger(__name__)

# ==============================================================================
# MAIN FUNCTIONS
# ==============================================================================

def morphological_graph(
    buildings_gdf: gpd.GeoDataFrame,
    segments_gdf: gpd.GeoDataFrame,
    center_point: gpd.GeoSeries | gpd.GeoDataFrame | None = None,
    distance: float | None = None,
    private_id_col: str | None = None,
    public_id_col: str | None = None,
    public_geom_col: str | None = "barrier_geometry",
    contiguity: str = "queen",
    keep_buildings: bool = False,
) -> Tuple[Dict[str, gpd.GeoDataFrame], Dict[Tuple[str, str, str], gpd.GeoDataFrame]]:
    """Create a morphological graph and return node/edge dictionaries."""

    _ensure_gdf(buildings_gdf, "buildings_gdf")
    _ensure_gdf(segments_gdf, "segments_gdf")

    if segments_gdf.crs != buildings_gdf.crs:
        warnings.warn("CRS mismatch between buildings and segments", RuntimeWarning)
        segments_gdf = segments_gdf.to_crs(buildings_gdf.crs)

    private_id_col = private_id_col or "tess_id"
    public_id_col = public_id_col or "id"

    barriers = _prepare_barriers(segments_gdf, public_geom_col)
    tessellation = create_tessellation(
        buildings_gdf,
        primary_barriers=barriers if not barriers.empty else None,
    )

    tessellation, private_id_col = _ensure_id_column(tessellation, private_id_col, "tess_id")

    if center_point is not None and distance is not None and not segments_gdf.empty:
        segs = filter_graph_by_distance(segments_gdf, center_point, distance)
    else:
        segs = segments_gdf

    segs, public_id_col = _ensure_id_column(segs, public_id_col, "id")

    tessellation = _filter_adjacent_tessellation(tessellation, segs)
    if keep_buildings:
        tessellation = _add_building_info(tessellation, buildings_gdf)

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
    if priv_pub.empty:
        warnings.warn("No private to public connections found", RuntimeWarning)

    nodes = {
        "private": _set_index_if_exists(tessellation, private_id_col),
        "public": _set_index_if_exists(segs, public_id_col),
    }
    edges = {
        ("private", "touched_to", "private"): _set_edge_index(
            priv_priv, "from_private_id", "to_private_id"
        ),
        ("public", "connected_to", "public"): _set_edge_index(
            pub_pub, "from_public_id", "to_public_id"
        ),
        ("private", "faced_to", "public"): _set_edge_index(
            priv_pub, "private_id", "public_id"
        ),
    }
    return nodes, edges


# ==============================================================================
# PRIVATE TO PRIVATE
# ==============================================================================

def private_to_private_graph(
    private_gdf: gpd.GeoDataFrame,
    private_id_col: str | None = None,
    group_col: str | None = None,
    contiguity: str = "queen",
) -> gpd.GeoDataFrame:
    """Return edges between contiguous private polygons."""
    _ensure_gdf(private_gdf, "private_gdf")
    private_id_col = private_id_col or "tess_id"

    if private_gdf.empty or len(private_gdf) < 2:
        return _empty_edges_gdf(private_gdf.crs, "from_private_id", "to_private_id", [group_col or "group"])

    if contiguity not in {"queen", "rook"}:
        raise ValueError("contiguity must be 'queen' or 'rook'")

    private_gdf, private_id_col = _ensure_id_column(private_gdf, private_id_col, "tess_id")

    if group_col and group_col not in private_gdf.columns:
        raise ValueError(f"group_col '{group_col}' not found")

    groups = {
        "all": private_gdf
    } if group_col is None else dict(tuple(private_gdf.groupby(group_col)))

    # drop duplicates to avoid Series return when indexing by private_id_col
    centroids = (
        private_gdf
        .drop_duplicates(subset=private_id_col)
        .set_index(private_id_col)
        .geometry.centroid
    )
    edges: list[dict] = []
    seen: set[tuple[Any, Any]] = set()

    for name, gdf in groups.items():
        if len(gdf) < 2:
            continue
        gdf = gdf.reset_index(drop=True)
        w = (
            libpysal.weights.Queen.from_dataframe(gdf)
            if contiguity == "queen"
            else libpysal.weights.Rook.from_dataframe(gdf)
        )
        if not w.neighbors:
            continue
        for i, neighbors in w.neighbors.items():
            id1 = gdf.loc[i, private_id_col]
            for j in neighbors:
                id2 = gdf.loc[j, private_id_col]
                if id1 == id2:
                    continue
                pair = tuple(sorted((id1, id2)))
                if pair in seen:
                    continue
                seen.add(pair)
                p1 = centroids[id1]
                p2 = centroids[id2]
                edges.append(
                    {
                        "from_private_id": pair[0],
                        "to_private_id": pair[1],
                        group_col or "group": name,
                        "geometry": LineString([(p1.x, p1.y), (p2.x, p2.y)]),
                    }
                )

    if not edges:
        return _empty_edges_gdf(
            private_gdf.crs,
            "from_private_id",
            "to_private_id",
            [group_col or "group"],
        )

    return gpd.GeoDataFrame(edges, geometry="geometry", crs=private_gdf.crs)


# ==============================================================================
# PRIVATE TO PUBLIC
# ==============================================================================

def private_to_public_graph(
    private_gdf: gpd.GeoDataFrame,
    public_gdf: gpd.GeoDataFrame,
    private_id_col: str | None = None,
    public_id_col: str | None = None,
    public_geom_col: str | None = None,
    tolerance: float = 1.0,
) -> gpd.GeoDataFrame:
    """Connect private polygons to nearby public geometries."""

    _ensure_gdf(private_gdf, "private_gdf")
    _ensure_gdf(public_gdf, "public_gdf")
    if private_gdf.empty or public_gdf.empty:
        return _empty_edges_gdf(private_gdf.crs, "private_id", "public_id")

    private_id_col = private_id_col or "tess_id"
    public_id_col = public_id_col or "id"
    private_gdf, private_id_col = _ensure_id_column(private_gdf, private_id_col, "tess_id")
    public_gdf, public_id_col = _ensure_id_column(public_gdf, public_id_col, "id")
    public_gdf = _ensure_crs_match(private_gdf, public_gdf)

    join_geom = public_gdf[public_geom_col] if public_geom_col and public_geom_col in public_gdf.columns else public_gdf.geometry
    pubs = gpd.GeoDataFrame({public_id_col: public_gdf[public_id_col]}, geometry=join_geom.buffer(tolerance), crs=public_gdf.crs)
    joined = gpd.sjoin(private_gdf[[private_id_col, "geometry"]], pubs, how="inner", predicate="intersects")
    if joined.empty:
        return _empty_edges_gdf(private_gdf.crs, "private_id", "public_id")

    priv_cent = private_gdf.drop_duplicates(subset=private_id_col).set_index(private_id_col).geometry.centroid
    pub_cent = public_gdf.drop_duplicates(subset=public_id_col).set_index(public_id_col).geometry.centroid

    lines: list[dict] = []
    for pid, pubs in joined.groupby(private_id_col)[public_id_col]:
        for pubid in pubs.unique():
            if pid not in priv_cent.index or pubid not in pub_cent.index:
                continue
            p1 = priv_cent[pid]
            p2 = pub_cent[pubid]
            lines.append({
                "private_id": pid,
                "public_id": pubid,
                "geometry": LineString([(p1.x, p1.y), (p2.x, p2.y)]),
            })

    if not lines:
        return _empty_edges_gdf(private_gdf.crs, "private_id", "public_id")
    return gpd.GeoDataFrame(lines, geometry="geometry", crs=private_gdf.crs)


# ==============================================================================
# PUBLIC TO PUBLIC
# ==============================================================================

def public_to_public_graph(
    public_gdf: gpd.GeoDataFrame,
    public_id_col: str | None = None,
    tolerance: float = 1e-8,
) -> gpd.GeoDataFrame:
    """Create edges between connected public segments."""

    _ensure_gdf(public_gdf, "public_gdf")
    if public_gdf.empty or len(public_gdf) < 2:
        return _empty_edges_gdf(public_gdf.crs, "from_public_id", "to_public_id")

    public_id_col = public_id_col or "id"
    public_gdf, public_id_col = _ensure_id_column(public_gdf, public_id_col, "id")

    nodes, connections = dual_graph(public_gdf, id_col=public_id_col, tolerance=tolerance)
    if nodes.empty or not connections:
        return _empty_edges_gdf(public_gdf.crs, "from_public_id", "to_public_id")

    edges: list[dict] = []
    seen: set[Tuple[Any, Any]] = set()
    for from_id, to_ids in connections.items():
        for to_id in to_ids:
            pair = tuple(sorted((from_id, to_id)))
            if pair in seen:
                continue
            seen.add(pair)
            p1 = nodes.loc[from_id].geometry
            p2 = nodes.loc[to_id].geometry
            edges.append({
                "from_public_id": pair[0],
                "to_public_id": pair[1],
                "geometry": LineString([(p1.x, p1.y), (p2.x, p2.y)]),
            })
    if not edges:
        return _empty_edges_gdf(public_gdf.crs, "from_public_id", "to_public_id")
    return gpd.GeoDataFrame(edges, geometry="geometry", crs=public_gdf.crs)


# ==============================================================================
# HELPER UTILITIES
# ==============================================================================

def _ensure_gdf(obj: Any, name: str) -> None:
    if not isinstance(obj, gpd.GeoDataFrame):
        raise TypeError(f"{name} must be a GeoDataFrame")


def _ensure_id_column(
    gdf: gpd.GeoDataFrame, column: str | None, default: str
) -> Tuple[gpd.GeoDataFrame, str]:
    """Ensure ``column`` exists on ``gdf`` without altering existing values."""

    col = column or default
    if col in gdf.columns:
        return gdf, col

    gdf = gdf.copy()
    gdf[col] = range(len(gdf))
    return gdf, col


def _ensure_crs_match(target: gpd.GeoDataFrame, source: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if source.crs != target.crs:
        warnings.warn("CRS mismatch", RuntimeWarning)
        return source.to_crs(target.crs)
    return source


def _prepare_barriers(segments: gpd.GeoDataFrame, geom_col: str | None) -> gpd.GeoDataFrame:
    if geom_col and geom_col in segments.columns and geom_col != "geometry":
        return gpd.GeoDataFrame(segments.drop(columns=["geometry"]), geometry=segments[geom_col], crs=segments.crs)
    return segments.copy()


def _filter_adjacent_tessellation(tess: gpd.GeoDataFrame, segments: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if tess.empty or segments.empty:
        return tess.copy()
    joined = gpd.sjoin(tess, segments, how="inner", predicate="intersects")
    return tess.loc[joined.index.unique()]


def _add_building_info(tess: gpd.GeoDataFrame, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if buildings.empty:
        return tess.copy()
    joined = gpd.sjoin(tess, buildings, how="left", predicate="intersects")
    if "index_right" in joined.columns:
        mapping = buildings.geometry.to_dict()
        joined["building_geometry"] = joined["index_right"].map(mapping)
        joined = joined.drop(columns=["index_right"])
    return joined


def _empty_edges_gdf(crs: Any, from_col: str, to_col: str, extra_cols: Iterable[str] | None = None) -> gpd.GeoDataFrame:
    cols = [from_col, to_col] + list(extra_cols or []) + ["geometry"]
    return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs=crs)


def _set_index_if_exists(gdf: gpd.GeoDataFrame, col: str) -> gpd.GeoDataFrame:
    return gdf.set_index(col) if col in gdf.columns else gdf


def _set_edge_index(gdf: gpd.GeoDataFrame, from_col: str, to_col: str) -> gpd.GeoDataFrame:
    if not gdf.empty and from_col in gdf.columns and to_col in gdf.columns:
        return gdf.set_index([from_col, to_col])
    return gdf


# Additional helper used in tests ------------------------------------------------

def _prep_contiguity_graph(
    gdf: gpd.GeoDataFrame,
    id_col: str | None,
    contiguity: str,
) -> Tuple[nx.Graph | None, Dict[int, Any] | None]:
    if gdf.empty or len(gdf) < 2:
        return None, None
    if contiguity not in {"queen", "rook"}:
        raise ValueError("contiguity must be 'queen' or 'rook'")
    gdf = gdf.reset_index(drop=True)
    w = libpysal.weights.Queen.from_dataframe(gdf) if contiguity == "queen" else libpysal.weights.Rook.from_dataframe(gdf)
    if not w.neighbors:
        return None, None
    mapping = {i: gdf.iloc[i][id_col] if id_col else i for i in range(len(gdf))}
    return w.to_networkx(), mapping


def _validate_inputs(privates: Any, publics: Any, buffer_dist: Any) -> None:
    _ensure_gdf(privates, "privates")
    _ensure_gdf(publics, "publics")
    if not isinstance(buffer_dist, (int, float)):
        raise TypeError("buffer_dist must be a number")


def _check_empty_dataframes(privates: gpd.GeoDataFrame, publics: gpd.GeoDataFrame) -> bool:
    if privates.empty or publics.empty:
        warnings.warn("One or both GeoDataFrames are empty", RuntimeWarning)
        return True
    return False


def _validate_columns(
    public_gdf: gpd.GeoDataFrame,
    private_gdf: gpd.GeoDataFrame,
    public_id_col: str | None,
    private_id_col: str | None,
    public_geom_col: str | None,
) -> None:
    if public_id_col and public_id_col not in public_gdf.columns:
        raise ValueError(f"public_id_col '{public_id_col}' not found in publics")
    if private_id_col and private_id_col not in private_gdf.columns:
        raise ValueError(f"private_id_col '{private_id_col}' not found in privates")
    if public_geom_col and public_geom_col not in public_gdf.columns:
        raise ValueError(f"public_geom_col '{public_geom_col}' not found in publics")


def _validate_geometries(private_gdf: gpd.GeoDataFrame, public_gdf: gpd.GeoDataFrame) -> None:
    if not all(private_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])):
        warnings.warn("Invalid geometries in privates", RuntimeWarning)
    if not all(public_gdf.geometry.type.isin(["LineString", "MultiLineString", "Point"])):
        warnings.warn("Invalid geometries in publics", RuntimeWarning)

