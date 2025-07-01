"""transportation.py

Utility functions to ingest GTFS feeds, generate origin-destination (OD) pairs,
and build simple stop-to-stop graphs.  The implementation purposefully sticks to
plain functions—no classes—to keep the public surface small, the state
transparent and the code easy to re-use in notebooks or larger pipelines.

The main public helpers are:

load_gtfs(...)                -> dict[str, pd/Geo DataFrame]
get_od_pairs(...)             -> pd.DataFrame | gpd.GeoDataFrame
travel_summary_graph(...)     -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] |
                                 networkx.Graph

`travel_summary_graph` now complies with the return-type convention used in
utils.py: it can yield either a pair (nodes_gdf, edges_gdf) or a fully fledged
NetworkX graph.
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime
from datetime import timedelta
from pathlib import Path  # noqa: TC003

import geopandas as gpd
import networkx as nx  # noqa: TC002
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import Point

from .utils import gdf_to_nx

# --------------------------------------------------------------------------- #
# CONFIGURATION
# --------------------------------------------------------------------------- #
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = [
    "get_od_pairs",
    "load_gtfs",
    "travel_summary_graph",
]

# --------------------------------------------------------------------------- #
# GENERIC HELPERS
# --------------------------------------------------------------------------- #


def _read_csv_bytes(buf: bytes) -> pd.DataFrame:
    """Read a CSV residing in memory into a *string*-typed DataFrame."""
    return pd.read_csv(io.BytesIO(buf), dtype=str, encoding="utf-8-sig")


def _time_to_seconds(value: str | float | None) -> float:
    """Convert a GTFS HH:MM:SS string into seconds (24 h+ time supported)."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    h, m, s = map(int, value.split(":"))
    return (h * 3600) + (m * 60) + s


def _timestamp(gtfs_time: str, service_date: datetime) -> datetime | None:
    """Combine a GTFS time string with a date object."""
    if pd.isna(gtfs_time):
        return None
    h, m, s = map(int, gtfs_time.split(":"))
    day_offset, h = divmod(h, 24)
    try:
        ts = service_date.replace(hour=h, minute=m, second=s) + timedelta(days=day_offset)
    except ValueError:  # date out of range
        return None
    return ts


# --------------------------------------------------------------------------- #
# GTFS READING & BASIC CLEAN-UP
# --------------------------------------------------------------------------- #


def _load_gtfs_zip(path: str | Path) -> dict[str, pd.DataFrame]:
    """Unzip every *.txt file and load it into a raw pandas.DataFrame."""
    gtfs: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".txt") and not name.endswith("/"):
                try:
                    gtfs[name.rsplit("/", 1)[-1].replace(".txt", "")] = _read_csv_bytes(
                        zf.read(name)
                    )
                except Exception:  # pragma: no cover
                    logger.exception("Could not read %s", name)
    return gtfs


def _coerce_types(gtfs: dict[str, pd.DataFrame]) -> None:
    """Cast the most common numeric / boolean columns in-place."""
    # stops.txt
    if (stops := gtfs.get("stops")) is not None:
        for col in ("stop_lat", "stop_lon"):
            stops[col] = pd.to_numeric(stops[col], errors="coerce")

    # routes.txt
    if (routes := gtfs.get("routes")) is not None and "route_type" in routes:
        routes["route_type"] = pd.to_numeric(routes["route_type"], errors="coerce")

    # calendar.txt
    if (cal := gtfs.get("calendar")) is not None:
        for d in ("monday", "tuesday", "wednesday", "thursday",
                  "friday", "saturday", "sunday"):
            if d in cal.columns:
                cal[d] = cal[d].astype(float).fillna(0).astype(bool)


def _point_geometries(df: pd.DataFrame) -> gpd.GeoSeries:
    """Return EPSG:4326 GeoSeries built from `stop_lon`, `stop_lat`."""
    pts = [
        Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
        for lon, lat in zip(df["stop_lon"], df["stop_lat"], strict=False)
    ]
    return gpd.GeoSeries(pts, crs="EPSG:4326")


def _linestring_geometries(shapes: pd.DataFrame) -> gpd.GeoSeries:
    """Build LineStrings grouped by shape_id ordered by sequence."""
    shapes = shapes.copy()
    shapes[["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]] = shapes[
        ["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    ].apply(pd.to_numeric, errors="coerce")
    shapes = shapes.dropna().sort_values(["shape_id", "shape_pt_sequence"])

    geoms: dict[str, LineString] = {}
    for sid, grp in shapes.groupby("shape_id"):
        pts = [Point(lon, lat) for lon, lat in zip(grp["shape_pt_lon"], grp["shape_pt_lat"])]
        if len(pts) >= 2:
            geoms[sid] = LineString(pts)
    return gpd.GeoSeries(geoms, crs="EPSG:4326")


# --------------------------------------------------------------------------- #
# PUBLIC: LOAD GTFS
# --------------------------------------------------------------------------- #


def load_gtfs(path: str | Path) -> dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    """
    Parse a GTFS zip file and enrich `stops` / `shapes` with geometry columns.

    Returns
    -------
    dict
        Keys are the original file names (w/o extension); values are pandas or
        GeoPandas DataFrames ready for downstream analysis.
    """
    gtfs = _load_gtfs_zip(path)
    if not gtfs:
        logger.warning("No GTFS files found in %s", path)
        return gtfs

    _coerce_types(gtfs)

    # attach geometries ------------------------------------------------------ #
    if (stops := gtfs.get("stops")) is not None and {"stop_lat", "stop_lon"} <= set(stops):
        geo = _point_geometries(stops)
        gtfs["stops"] = gpd.GeoDataFrame(stops.dropna(subset=["stop_lat", "stop_lon"]),
                                         geometry=geo, crs="EPSG:4326")

    if (shapes := gtfs.get("shapes")) is not None:
        geo = _linestring_geometries(shapes)
        if not geo.empty:
            gtfs["shapes"] = gpd.GeoDataFrame(shapes, geometry=geo, crs="EPSG:4326")

    logger.info("GTFS loaded: %s", ", ".join(gtfs))
    return gtfs


# --------------------------------------------------------------------------- #
# ORIGIN-DESTINATION PAIR GENERATION
# --------------------------------------------------------------------------- #


def _create_basic_od(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Within each trip build successive stop pairs."""
    st = stop_times.copy()
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.dropna(subset=["stop_sequence"]).sort_values(["trip_id", "stop_sequence"])

    lead = st.groupby("trip_id").shift(-1)
    mask = lead["trip_id"].notna()

    od = pd.DataFrame({
        "trip_id": st.loc[mask, "trip_id"],
        "service_id": st.loc[mask, "trip_id"].map(
            stop_times.set_index("trip_id")["service_id"],
        ),
        "orig_stop_id": st.loc[mask, "stop_id"],
        "dest_stop_id": lead.loc[mask, "stop_id"],
        "departure_time": st.loc[mask, "departure_time"],
        "arrival_time": lead.loc[mask, "arrival_time"],
    })
    return od.reset_index(drop=True)


def _service_day_map(calendar: pd.DataFrame,
                     start: datetime,
                     end: datetime) -> dict[str, list[datetime]]:
    """Return {service_id: [date, ...]} within the given period."""
    # Create a mapping of weekday index to day name
    day_idx = {0: "monday", 1: "tuesday", 2: "wednesday", 3: "thursday",
               4: "friday", 5: "saturday", 6: "sunday"}

    # Create a mapping of service_id to active dates
    srv_dates: dict[str, list[datetime]] = {}

    for _, row in calendar.iterrows():
        sid = row["service_id"]
        s = datetime.strptime(str(row["start_date"]), "%Y%m%d")
        e = datetime.strptime(str(row["end_date"]), "%Y%m%d")
        for d in (max(s, start),
                  min(e, end)):
            pass  # variables needed for mypy only
        cur = max(start, s)
        while cur <= min(end, e):
            if row.get(day_idx[cur.weekday()], False):
                srv_dates.setdefault(sid, []).append(cur)
            cur += timedelta(days=1)
    return srv_dates


def _apply_calendar_exceptions(calendar_dates: pd.DataFrame,
                               srv_dates: dict[str, list[datetime]]) -> None:
    """Mutate `srv_dates` in-place with exception rules."""
    for _, row in calendar_dates.iterrows():
        sid = row["service_id"]
        date = datetime.strptime(str(row["date"]), "%Y%m%d")
        t = int(row["exception_type"])
        srv_dates.setdefault(sid, [])
        if t == 1 and date not in srv_dates[sid]:
            srv_dates[sid].append(date)
        if t == 2 and date in srv_dates[sid]:
            srv_dates[sid].remove(date)


def _expand_with_dates(od: pd.DataFrame,
                       srv_dates: dict[str, list[datetime]]) -> pd.DataFrame:
    """Cartesian multiply OD rows by their active service dates."""
    rows = []
    for _, r in od.iterrows():
        for d in srv_dates.get(r["service_id"], []):
            dep = _timestamp(r["departure_time"], d)
            arr = _timestamp(r["arrival_time"], d)
            if dep and arr:
                rows.append({
                    "trip_id": r["trip_id"],
                    "service_id": r["service_id"],
                    "orig_stop_id": r["orig_stop_id"],
                    "dest_stop_id": r["dest_stop_id"],
                    "departure_ts": dep,
                    "arrival_ts": arr,
                    "travel_time_sec": (arr - dep).total_seconds(),
                    "date": d.strftime("%Y-%m-%d"),
                })
    return pd.DataFrame(rows)


def get_od_pairs(
    gtfs: dict[str, pd.DataFrame | gpd.GeoDataFrame],
    start_date: str | None = None,
    end_date: str | None = None,
    include_geometry: bool = True,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
   Materialise origin-destination pairs for every trip and service day.

    If `include_geometry` is True, the result is a GeoDataFrame whose geometry
    is the LineString connecting the two stops.
    """
    if not {"stop_times", "trips", "stops", "calendar"} <= gtfs.keys():
        logger.error("GTFS feed incomplete – need stop_times, trips, stops & calendar")
        return pd.DataFrame()

    od = _create_basic_od(gtfs["stop_times"].merge(
        gtfs["trips"][["trip_id", "service_id"]], on="trip_id",
    ))

    # calendar expansion ----------------------------------------------------- #
    cal = gtfs["calendar"]
    start_dt = datetime.strptime(start_date, "%Y%m%d") if start_date else \
        datetime.strptime(cal["start_date"].min(), "%Y%m%d")
    end_dt = datetime.strptime(end_date, "%Y%m%d") if end_date else \
        datetime.strptime(cal["end_date"].max(), "%Y%m%d")

    srv_dates = _service_day_map(cal, start_dt, end_dt)
    if (cal_dates := gtfs.get("calendar_dates")) is not None:
        _apply_calendar_exceptions(cal_dates, srv_dates)

    od_full = _expand_with_dates(od, srv_dates)

    if not include_geometry:
        return od_full

    # attach LineString geometry -------------------------------------------- #
    stops = gtfs["stops"].set_index("stop_id")
    line_geoms = [
        LineString([stops.loc[o].geometry, stops.loc[d].geometry])
        if o in stops.index and d in stops.index else None
        for o, d in zip(od_full["orig_stop_id"], od_full["dest_stop_id"], strict=False)
    ]
    return gpd.GeoDataFrame(od_full, geometry=line_geoms, crs="EPSG:4326")


# --------------------------------------------------------------------------- #
# TRAVEL SUMMARY GRAPH
# --------------------------------------------------------------------------- #


def travel_summary_graph(
    gtfs: dict[str, pd.DataFrame | gpd.GeoDataFrame],
    start_time: str | None = None,
    end_time: str | None = None,
    calendar_start: str | None = None,
    calendar_end: str | None = None,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """
   Aggregate stop-to-stop travel time & frequency into an edge list.

    Returns
    -------
    (nodes_gdf, edges_gdf) OR networkx.Graph
        • Nodes: every stop with a valid geometry.
        • Edges: columns = from_stop_id, to_stop_id, travel_time, frequency, geometry.
    """
    if "stop_times" not in gtfs or "stops" not in gtfs:
        msg = "GTFS must contain at least stop_times and stops"
        raise ValueError(msg)

    st = gtfs["stop_times"].copy()
    st["arrival_sec"] = st["arrival_time"].map(_time_to_seconds)
    st["departure_sec"] = st["departure_time"].map(_time_to_seconds)

    if start_time:
        st = st[st["departure_sec"] >= _time_to_seconds(start_time)]
    if end_time:
        st = st[st["arrival_sec"] <= _time_to_seconds(end_time)]

    st = st.sort_values(["trip_id", "stop_sequence"])
    st["next_stop_id"] = st.groupby("trip_id")["stop_id"].shift(-1)
    st["next_arr_sec"] = st.groupby("trip_id")["arrival_sec"].shift(-1)
    st = st.dropna(subset=["next_stop_id", "next_arr_sec"]).copy()
    st["travel_time"] = st["next_arr_sec"] - st["departure_sec"]

    # calendar weighting ----------------------------------------------------- #
    st = st.merge(gtfs["trips"][["trip_id", "service_id"]], on="trip_id", how="left")
    cal_start = calendar_start or gtfs["calendar"]["start_date"].min()
    cal_end = calendar_end or gtfs["calendar"]["end_date"].max()
    srv_dates = _service_day_map(
        gtfs["calendar"],
        datetime.strptime(cal_start, "%Y%m%d"),
        datetime.strptime(cal_end, "%Y%m%d"),
    )
    counts = pd.Series({k: len(v) for k, v in srv_dates.items()}, name="service_days")
    st = st.merge(counts, left_on="service_id", right_index=True, how="left").fillna(0)
    st["weighted_time"] = st["travel_time"] * st["service_days"]

    agg = (
        st.groupby(["stop_id", "next_stop_id"])
        .agg(total_weight=("weighted_time", "sum"),
             freq=("service_days", "sum"))
        .reset_index()
    )
    agg["travel_time"] = agg["total_weight"] / agg["freq"]

    # build GeoDataFrames ---------------------------------------------------- #
    nodes_gdf: gpd.GeoDataFrame = gtfs["stops"].set_index("stop_id")
    edges_geom = [
        LineString([nodes_gdf.loc[u].geometry, nodes_gdf.loc[v].geometry])
        for u, v in zip(agg["stop_id"], agg["next_stop_id"], strict=False)
    ]
    edges_gdf = gpd.GeoDataFrame(
        {
            "from_stop_id": agg["stop_id"],
            "to_stop_id": agg["next_stop_id"],
            "travel_time": agg["travel_time"],
            "frequency": agg["freq"],
        },
        geometry=edges_geom,
        crs=nodes_gdf.crs,
    ).set_index(["from_stop_id", "to_stop_id"])

    return nodes_gdf, edges_gdf if not as_nx else gdf_to_nx(nodes_gdf, edges_gdf)
