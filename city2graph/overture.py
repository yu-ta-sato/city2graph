# overture.py
"""
Utility functions for downloading and post-processing Overture Maps data.

Public API
----------
load_overture_data
process_overture_segments
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import (
    LineString,
    MultiLineString,
    Polygon,
)
from shapely.ops import substring  # shapely ≥ 2

# -----------------------------------------------------------------------------#
# Configuration & globals
# -----------------------------------------------------------------------------#

LOG = logging.getLogger(__name__)
WGS84 = "EPSG:4326"

VALID_OVERTURE_TYPES: set[str] = {
    "address",
    "bathymetry",
    "building",
    "building_part",
    "division",
    "division_area",
    "division_boundary",
    "place",
    "segment",
    "connector",
    "infrastructure",
    "land",
    "land_cover",
    "land_use",
    "water",
}

# -----------------------------------------------------------------------------#
# Shared helpers
# -----------------------------------------------------------------------------#


def _validate_types(types: list[str] | None) -> list[str]:
    """Return a validated list of overture types (or every type if None)."""
    if types is None:
        return sorted(VALID_OVERTURE_TYPES)

    invalid = set(types) - VALID_OVERTURE_TYPES
    if invalid:
        raise ValueError(
            f"Invalid Overture Maps data type(s): {sorted(invalid)}. "
            f"Valid choices: {sorted(VALID_OVERTURE_TYPES)}"
        )
    return types


def _to_bbox_and_poly(
    area: list[float] | Polygon,
) -> tuple[list[float], Polygon | None]:
    """Return `[minx, miny, maxx, maxy]` and the original polygon (if given)."""
    if isinstance(area, Polygon):
        if getattr(area, "crs", WGS84) != WGS84:  # type: ignore[attr-defined]
            area = area.to_crs(WGS84)  # type: ignore[assignment]
        minx, miny, maxx, maxy = area.bounds
        return [round(v, 10) for v in (minx, miny, maxx, maxy)], area
    # area is already a bbox list
    if len(area) != 4:
        raise ValueError("Bounding box must be length-4 [minx, miny, maxx, maxy]")
    return list(map(float, area)), None


def _read_geojson(raw: str | Path) -> gpd.GeoDataFrame:
    """Wrapper around geopandas.read_file with empty fallback."""
    try:
        return gpd.read_file(raw)
    except Exception as exc:  # pragma: no cover
        LOG.warning("Failed to parse GeoJSON (%s). Returning empty frame.", exc)
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)


# -----------------------------------------------------------------------------#
# Downloading
# -----------------------------------------------------------------------------#


def _download_one(
    data_type: str,
    bbox: list[float],
    out_dir: Path,
    prefix: str,
    save: bool,
) -> gpd.GeoDataFrame:
    """Download one overture *data_type* layer and return a GeoDataFrame."""
    bbox_str = ",".join(map(str, bbox))
    out_path = out_dir / f"{prefix}{data_type}.geojson" if save else None

    cmd: list[str] = [
        "overturemaps",
        "download",
        f"--bbox={bbox_str}",
        "--format=geojson",
        f"--type={data_type}",
    ]
    if save:
        cmd += ["-o", str(out_path)]

    LOG.debug("Running: %s", " ".join(cmd))
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE if not save else subprocess.DEVNULL,
            text=True,
        )
    except subprocess.CalledProcessError as exc:  # pragma: no cover
        LOG.warning("Overture CLI failed for %s (%s)", data_type, exc)
        return gpd.GeoDataFrame(geometry=[], crs=WGS84)

    if save:
        return _read_geojson(out_path)
    return _read_geojson(proc.stdout)


# -----------------------------------------------------------------------------#
# Public: load_overture_data
# -----------------------------------------------------------------------------#


def load_overture_data(
    area: list[float] | Polygon,
    types: list[str] | None = None,
    output_dir: str | Path = ".",
    prefix: str = "",
    save_to_file: bool = True,
    return_data: bool = True,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Download selected Overture Maps layers intersecting *area*.

    *area* may be a WGS-84 polygon or a `[minx, miny, maxx, maxy]` list.
    """
    types = _validate_types(types)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox, poly = _to_bbox_and_poly(area)
    result: dict[str, gpd.GeoDataFrame] = {}

    for t in types:
        gdf = _download_one(t, bbox, output_dir, prefix, save_to_file)
        if poly is not None and not gdf.empty:
            gdf = gpd.clip(gdf, poly)
        if return_data:
            result[t] = gdf

    return result


# -----------------------------------------------------------------------------#
# Barrier / connector utilities
# -----------------------------------------------------------------------------#


def _barrier_mask(level_rules: str | None) -> list[list[float]]:
    """
    Return non-barrier intervals from *level_rules*.

    A 0-valued rule marks a barrier; intervals are returned as complement.
    """
    if not level_rules:
        return [[0.0, 1.0]]

    try:
        rules = json.loads(level_rules.replace("'", '"'))
        rules = rules if isinstance(rules, list) else [rules]
    except (json.JSONDecodeError, TypeError):
        return [[0.0, 1.0]]

    # collect barrier sub-intervals
    blocked: list[tuple[float, float]] = [
        tuple(map(float, r["between"]))
        for r in rules
        if r.get("value") != 0 and r.get("between") is not None
    ]

    if not blocked:
        return [[0.0, 1.0]]

    blocked.sort()
    free: list[list[float]] = []
    cur = 0.0
    for a, b in blocked:
        if a > cur:
            free.append([cur, a])
        cur = max(cur, b)
    if cur < 1.0:
        free.append([cur, 1.0])
    return free


def _connector_mask(connectors: str | None) -> list[float]:
    """Return sorted breakpoints [0.0, ..., 1.0] from connectors JSON."""
    if not connectors:
        return [0.0, 1.0]
    try:
        items = json.loads(connectors.replace("'", '"'))
        items = items if isinstance(items, list) else [items]
        pts = sorted(float(d["at"]) for d in items if "at" in d)
    except (json.JSONDecodeError, TypeError, ValueError):
        pts = []
    return [0.0, *pts, 1.0]


def _substring(
    line: LineString,
    start: float,
    end: float,
) -> LineString | None:
    """Return *line* substring from *start* to *end* fractions."""
    if start >= end or line.is_empty:
        return None
    if start <= 0 and end >= 1:
        return line
    try:
        return substring(line, start, end, normalized=True)  # shapely ≥ 2
    except Exception:  # pragma: no cover
        # Fallback for shapely < 2
        length = line.length
        return line.interpolate(start * length), line.interpolate(end * length)


# -----------------------------------------------------------------------------#
# Splitting segments at connectors
# -----------------------------------------------------------------------------#


def _split_one_segment(row: pd.Series, valid_ids: set[Any]) -> list[pd.Series]:
    """Return 1-n split rows for a single segment."""
    connectors_json = row.get("connectors")
    if not connectors_json:
        return [row]

    try:
        items = json.loads(str(connectors_json).replace("'", '"'))
        items = items if isinstance(items, list) else [items]
    except (json.JSONDecodeError, TypeError):
        return [row]

    cuts = sorted(
        float(it["at"])
        for it in items
        if isinstance(it, dict)
        and it.get("connector_id") in valid_ids
        and "at" in it
    )
    if not cuts:
        return [row]

    parts: list[pd.Series] = []
    start = 0.0
    mask = [0.0, *cuts, 1.0]
    counter = 1
    for stop in cuts + [1.0]:
        geom_part = _substring(row.geometry, start, stop)
        if geom_part:
            new = row.copy()
            new.geometry = geom_part
            new["connector_mask"] = mask
            new["split_from"], new["split_to"] = start, stop
            new["barrier_mask"] = _recalc_mask(
                row["barrier_mask"], start, stop
            )
            new["id"] = f"{row.get('id', row.name)}_{counter}"
            parts.append(new)
            counter += 1
        start = stop
    return parts


def _recalc_mask(mask: list[list[float]], s: float, e: float) -> list[list[float]]:
    """Shrink barrier mask to sub-interval [s, e]."""
    if mask == [[0.0, 1.0]]:
        return mask
    seg_len = e - s
    new: list[list[float]] = []
    for a, b in mask:
        a2, b2 = max(a, s), min(b, e)
        if a2 < b2:
            new.append([(a2 - s) / seg_len, (b2 - s) / seg_len])
    return new or [[0.0, 1.0]]


def _split_segments(
    segments: gpd.GeoDataFrame, connectors: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Vectorised splitter returning a fresh GeoDataFrame."""
    valid = set(connectors["id"])
    out_rows: list[pd.Series] = []
    for _, row in segments.iterrows():
        out_rows.extend(_split_one_segment(row, valid))
    return gpd.GeoDataFrame(out_rows, crs=segments.crs).reset_index(drop=True)


# -----------------------------------------------------------------------------#
# Endpoint snapping
# -----------------------------------------------------------------------------#


def _snap_endpoints(
    gdf: gpd.GeoDataFrame,
    threshold: float,
) -> gpd.GeoDataFrame:
    """Cluster endpoints within *threshold* and rebuild geometries."""
    mask = gdf.geometry.type == "LineString"
    if not mask.any():
        return gdf

    work = gdf.loc[mask].copy()
    work["seg_id"] = np.arange(len(work))

    # endpoints → DataFrame
    coords = np.concatenate(
        [np.vstack(work.geometry.apply(lambda g: [g.coords[0], g.coords[-1]]))]
    )
    ep = pd.DataFrame(
        coords, columns=["x", "y"]
    )
    ep["seg_id"] = np.repeat(work["seg_id"].values, 2)
    ep["pos"] = np.tile(["start", "end"], len(work))

    # quantise
    ep["bin_x"] = np.rint(ep["x"] / threshold)
    ep["bin_y"] = np.rint(ep["y"] / threshold)
    ep["bin"] = list(zip(ep["bin_x"], ep["bin_y"]))

    centroids = ep.groupby("bin")[["x", "y"]].mean().rename(
        columns={"x": "cx", "y": "cy"}
    )
    ep = ep.join(centroids, on="bin")

    pivot = ep.pivot_table(
        index="seg_id",
        columns="pos",
        values=["cx", "cy"],
    )

    def rebuild(row):
        a = (pivot.loc[row.seg_id, ("cx", "start")], pivot.loc[row.seg_id, ("cy", "start")])
        b = (pivot.loc[row.seg_id, ("cx", "end")], pivot.loc[row.seg_id, ("cy", "end")])
        coords = list(row.geometry.coords)
        return LineString([a, *coords[1:-1], b] if len(coords) > 2 else [a, b])

    work.geometry = work.apply(rebuild, axis=1)
    gdf.update(work)
    return gdf


# -----------------------------------------------------------------------------#
# Public: process_overture_segments
# -----------------------------------------------------------------------------#


def process_overture_segments(
    segments_gdf: gpd.GeoDataFrame,
    *,
    get_barriers: bool = True,
    connectors_gdf: gpd.GeoDataFrame | None = None,
    threshold: float = 1.0,
) -> gpd.GeoDataFrame:
    """
    Split Overture road `segments` at connectors, compute lengths and barriers.
    """
    segments = segments_gdf.copy()

    if get_barriers:
        segments["barrier_mask"] = segments["level_rules"].astype(str).apply(_barrier_mask)

    if connectors_gdf is not None:
        segments = _split_segments(segments, connectors_gdf)
        segments = _snap_endpoints(segments, threshold)

    segments["length"] = segments.geometry.length

    if get_barriers:
        segments["barrier_geometry"] = segments.apply(
            lambda r: _substring(r.geometry, *r.barrier_mask[0])
            if r.barrier_mask != [[0.0, 1.0]]
            else r.geometry,
            axis=1,
        )

    return segments.reset_index(drop=True)
