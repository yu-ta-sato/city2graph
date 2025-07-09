"""
Transportation Network Analysis Module.

This module provides comprehensive functionality for processing General Transit Feed
Specification (GTFS) data and creating transportation network representations. It
specializes in converting public transit data into graph structures suitable for
network analysis and accessibility studies.

The module emphasizes functional programming principles, using plain functions
without class dependencies to ensure explicit state management and maintain a
minimal public API surface. All functions return ready-to-use pandas/GeoPandas
objects or NetworkX graphs that can be seamlessly integrated into analysis
pipelines, notebooks, or model training workflows.
"""
# Future annotations for type hints
from __future__ import annotations

# Standard library imports
import contextlib
import io
import logging
import zipfile
from datetime import datetime
from datetime import timedelta
from typing import TYPE_CHECKING

# Third-party imports
import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import Point

# Local imports
from .utils import gdf_to_nx  # pragma: no cover

# Type checking imports
if TYPE_CHECKING:
    from pathlib import Path

    import networkx as nx

# Module logger configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

__all__ = ["get_od_pairs", "load_gtfs", "travel_summary_graph"]


# =============================================================================
# INTERNAL HELPER FUNCTIONS
# =============================================================================
# All helper functions are private (underscore-prefixed) and support the
# main public API functions. They handle low-level data processing tasks
# such as CSV parsing, time conversion, and GTFS data validation.

# -----------------------------------------------------------------------------
# CSV Processing Utilities
# -----------------------------------------------------------------------------


def _read_csv_bytes(buf: bytes) -> pd.DataFrame:
    """Read a CSV sitting entirely in memory and return *string-typed* columns."""
    return pd.read_csv(io.BytesIO(buf), dtype=str, encoding="utf-8-sig")


def _time_to_seconds(value: str | float | None) -> float:
    """Convert a GTFS ``HH:MM:SS`` string (24 h+ supported) into seconds."""
    # Convert None and numeric values to float directly
    if not isinstance(value, str):
        return float(value or 0)

    # At this point, value must be a string - parse time
    h, m, s = map(int, value.split(":"))
    return h * 3600 + m * 60 + s


def _timestamp(gtfs_time: str | float | None, service_date: datetime) -> datetime | None:
    """Combine a GTFS time string with a **date** producing a proper timestamp.

    Times beyond 24 : 00 : 00 are correctly rolled over to the next day.
    """
    if not isinstance(gtfs_time, str):
        # Handle None or numeric values by converting to seconds first
        seconds = _time_to_seconds(gtfs_time)
        h, remainder = divmod(int(seconds), 3600)
        m, s = divmod(remainder, 60)
        day_offset, h = divmod(h, 24)
        return service_date.replace(hour=h, minute=m, second=s) + timedelta(days=day_offset)

    h, m, s = map(int, gtfs_time.split(":"))
    day_offset, h = divmod(h, 24)
    return service_date.replace(hour=h, minute=m, second=s) + timedelta(days=day_offset)


# ---------------------------------------------------------------------------
# GTFS reading & basic clean-up helpers
# ---------------------------------------------------------------------------


def _load_gtfs_zip(path: str | Path) -> dict[str, pd.DataFrame]:
    """Unzip every *.txt* file found inside *path* into a raw DataFrame."""
    gtfs: dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith(".txt") and not name.endswith("/"):
                try:
                    key = name.rsplit("/", 1)[-1].replace(".txt", "")
                    gtfs[key] = _read_csv_bytes(zf.read(name))
                except Exception:  # pragma: no cover
                    logger.exception("Could not read %s", name)
    return gtfs


def _coerce_types(gtfs: dict[str, pd.DataFrame]) -> None:
    """Cast common numeric / boolean columns **in-place** for easier analysis."""
    # Coerce stops.txt coordinates
    if (stops := gtfs.get("stops")) is not None:
        for col in ("stop_lat", "stop_lon"):
            stops[col] = pd.to_numeric(stops[col], errors="coerce")

    # Coerce routes.txt route_type
    if (routes := gtfs.get("routes")) is not None and "route_type" in routes:
        routes["route_type"] = pd.to_numeric(routes["route_type"], errors="coerce")

    # Coerce calendar.txt weekday flags
    if (cal := gtfs.get("calendar")) is not None:
        weekdays = (
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        )
        for d in weekdays:
            if d in cal.columns:
                cal[d] = cal[d].astype(float).fillna(0).astype(bool)

def _get_service_counts(gtfs_data: dict[str, pd.DataFrame | gpd.GeoDataFrame], start_date_str: str, end_date_str: str) -> pd.Series:
    """Calculate the number of active days for each service_id within a date range."""
    calendar = gtfs_data.get("calendar")
    calendar_dates = gtfs_data.get("calendar_dates")

    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    # Create a date range and map each date to its weekday name
    all_dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq="D"))
    date_df = pd.DataFrame({"date": all_dates, "day_name": all_dates.strftime("%A").str.lower()})

    service_days = pd.Series(dtype=int)

    # Process regular service days from calendar.txt
    if calendar is not None:
        cal_df = calendar.melt(
            id_vars=["service_id", "start_date", "end_date"],
            value_vars=["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"],
            var_name="day_name",
            value_name="is_active",
        )
        cal_df = cal_df[cal_df["is_active"] == 1]

        # Merge with the date range to find active dates
        merged = cal_df.merge(date_df, on="day_name")
        merged["start_date"] = pd.to_datetime(merged["start_date"], format="%Y%m%d")
        merged["end_date"] = pd.to_datetime(merged["end_date"], format="%Y%m%d")

        active_days = merged[
            (merged["date"] >= merged["start_date"]) & (merged["date"] <= merged["end_date"])
        ]
        service_days = active_days.groupby("service_id").size()

    return service_days.astype(int).clip(lower=0)

# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------


def _point_geometries(df: pd.DataFrame) -> gpd.GeoSeries:
    """Return an EPSG:4326 GeoSeries built from ``stop_lon`` / ``stop_lat``."""
    pts = [
        Point(lon, lat) if pd.notna(lon) and pd.notna(lat) else None
        for lon, lat in zip(df["stop_lon"], df["stop_lat"], strict=False)
    ]
    return gpd.GeoSeries(pts, crs="EPSG:4326")


def _linestring_geometries(shapes: pd.DataFrame) -> gpd.GeoSeries:
    """Create LineStrings grouped by *shape_id* ordered by *shape_pt_sequence*."""
    shapes = shapes.copy()
    shapes[["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]] = shapes[
        ["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    ].apply(pd.to_numeric, errors="coerce")

    shapes = shapes.dropna().sort_values(["shape_id", "shape_pt_sequence"])

    geoms: dict[str, LineString] = {}
    for sid, grp in shapes.groupby("shape_id"):
        pts = [Point(lon, lat) for lon, lat in zip(grp["shape_pt_lon"], grp["shape_pt_lat"], strict=False)]
        if len(pts) >= 2:
            geoms[sid] = LineString(pts)
    return gpd.GeoSeries(geoms, crs="EPSG:4326")


# ===========================================================================
# PUBLIC HELPERS
# ===========================================================================


def load_gtfs(path: str | Path) -> dict[str, pd.DataFrame | gpd.GeoDataFrame]:
    """Parse a GTFS zip file and *enrich* ``stops`` / ``shapes`` with geometry.

    Parameters
    ----------
    path : str or pathlib.Path
        Location of the zipped GTFS feed (e.g. ``"./rome_gtfs.zip"``).

    Returns
    -------
    dict
        Keys are the original GTFS file names (without extension) and values
        are pandas or GeoPandas DataFrames ready for analysis.

    Notes
    -----
    *   The function never mutates the original file - everything is kept
        in memory.
    *   Geometry columns are added **only** when the relevant coordinate
        columns are present and valid.

    Examples
    --------
    >>> from pathlib import Path
    >>> gtfs = load_gtfs(Path("data/rome_gtfs.zip"))
    >>> print(list(gtfs))
    ['agency', 'routes', 'trips', 'stops', ...]
    >>> gtfs['stops'].head(3)[['stop_name', 'geometry']]
           stop_name                     geometry
    0  Termini (MA)  POINT (12.50118 41.90088)
    1   Colosseo(MB)  POINT (12.49224 41.89021)
    """
    gtfs = _load_gtfs_zip(path)
    if not gtfs:
        logger.warning("No GTFS files found in %s", path)
        return gtfs

    _coerce_types(gtfs)

    # Attach geometries to stops and shapes
    if (stops := gtfs.get("stops")) is not None and {"stop_lat", "stop_lon"} <= set(stops):
        geo = _point_geometries(stops)
        gtfs["stops"] = gpd.GeoDataFrame(
            stops.dropna(subset=["stop_lat", "stop_lon"]), geometry=geo, crs="EPSG:4326",
        )

    if (shapes := gtfs.get("shapes")) is not None:
        geo = _linestring_geometries(shapes)
        if not geo.empty:
            gtfs["shapes"] = gpd.GeoDataFrame(shapes, geometry=geo, crs="EPSG:4326")

    logger.info("GTFS loaded: %s", ", ".join(gtfs))
    return gtfs


# ---------------------------------------------------------------------------
# ORIGIN-DESTINATION PAIRS
# ---------------------------------------------------------------------------


def _create_basic_od(stop_times: pd.DataFrame) -> pd.DataFrame:
    """Within each trip build *successive* stop pairs (u → v)."""
    st = stop_times.copy()
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.dropna(subset=["stop_sequence"]).sort_values(["trip_id", "stop_sequence"])

    lead = st.groupby("trip_id").shift(1)
    mask = lead["stop_id"].notna()

    od = pd.DataFrame(
        {
            "trip_id": st.loc[mask, "trip_id"],
            "service_id": st.loc[mask, "service_id"],
            "orig_stop_id": st.loc[mask, "stop_id"],
            "dest_stop_id": lead.loc[mask, "stop_id"],
            "departure_time": st.loc[mask, "departure_time"],
            "arrival_time": lead.loc[mask, "arrival_time"],
        },
    )
    return od.reset_index(drop=True)


def _service_day_map(
    calendar: pd.DataFrame, start: datetime, end: datetime,
) -> dict[str, list[datetime]]:
    """Return ``{service_id: [date, …]}`` within the given period."""
    day_idx = {
        0: "monday",
        1: "tuesday",
        2: "wednesday",
        3: "thursday",
        4: "friday",
        5: "saturday",
        6: "sunday",
    }
    srv_dates: dict[str, list[datetime]] = {}

    for _, row in calendar.iterrows():
        sid = row["service_id"]
        s = datetime.strptime(str(row["start_date"]), "%Y%m%d")
        e = datetime.strptime(str(row["end_date"]), "%Y%m%d")

        cur = max(start, s)
        while cur <= min(end, e):
            if row.get(day_idx[cur.weekday()], False):
                srv_dates.setdefault(sid, []).append(cur)
            cur += timedelta(days=1)
    return srv_dates


def _apply_calendar_exceptions(
    calendar_dates: pd.DataFrame, srv_dates: dict[str, list[datetime]],
) -> None:
    """Mutate *srv_dates* **in-place** with exception rules."""
    for _, row in calendar_dates.iterrows():
        sid = row["service_id"]
        date = datetime.strptime(str(row["date"]), "%Y%m%d")
        t = int(row["exception_type"])
        srv_dates.setdefault(sid, [])
        # Only removal of service exceptions is supported
        if t == 2:
            # Remove service exception: safely remove if present
            with contextlib.suppress(ValueError):
                srv_dates[sid].remove(date)


def _expand_with_dates(od: pd.DataFrame, srv_dates: dict[str, list[datetime]]) -> pd.DataFrame:
    """Cartesian multiply *od* rows by their active **service dates**."""
    rows: list[dict[str, object]] = []
    for _, r in od.iterrows():
        for d in srv_dates.get(r["service_id"], []):
            dep = _timestamp(r["departure_time"], d)
            arr = _timestamp(r["arrival_time"], d)
            if dep and arr:
                rows.append(
                    {
                        "trip_id": r["trip_id"],
                        "service_id": r["service_id"],
                        "orig_stop_id": r["orig_stop_id"],
                        "dest_stop_id": r["dest_stop_id"],
                        "departure_ts": dep,
                        "arrival_ts": arr,
                        "travel_time_sec": (arr - dep).total_seconds(),
                        "date": d.strftime("%Y-%m-%d"),
                    },
                )
    return pd.DataFrame(rows)


def get_od_pairs(
    gtfs: dict[str, pd.DataFrame | gpd.GeoDataFrame],
    start_date: str | None = None,
    end_date: str | None = None,
    include_geometry: bool = True,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Materialise origin-destination pairs for every trip and service day.

    Parameters
    ----------
    gtfs : dict
        Dictionary returned by :func:`load_gtfs`.
    start_date, end_date : str or None, optional
        Restrict the calendar expansion to the closed interval
        ``[start_date, end_date]`` (format *YYYYMMDD*).  When *None* the period
        is inferred from ``calendar.txt``.
    include_geometry : bool, default True
        If *True* the result is a GeoDataFrame whose geometry is a *straight*
        LineString connecting the two stops.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
        One row per *trip-day-leg* with departure / arrival timestamps,
        travel time in seconds and, optionally, geometry.

    Examples
    --------
    >>> gtfs = load_gtfs("data/rome_gtfs.zip")
    >>> od = get_od_pairs(gtfs, start_date="20230101", end_date="20230107")
    >>> od.head(3)[['orig_stop_id', 'dest_stop_id', 'travel_time_sec']]
      orig_stop_id dest_stop_id  travel_time_sec
    0      7045490      7045491            120.0
    1      7045491      7045492            180.0
    2      7045492      7045493            240.0
    """
    required = {"stop_times", "trips", "stops", "calendar"}
    if not required <= gtfs.keys():
        logger.error("GTFS feed incomplete - need %s", ", ".join(required))
        return pd.DataFrame()

    # 1. Create successive stop pairs from stop_times
    od = _create_basic_od(
        gtfs["stop_times"].merge(gtfs["trips"][["trip_id", "service_id"]], on="trip_id"),
    )

    # 2. Expand OD pairs with calendar dates
    cal = gtfs["calendar"]
    start_dt = (
        datetime.strptime(start_date, "%Y%m%d")
        if start_date
        else datetime.strptime(cal["start_date"].min(), "%Y%m%d")
    )
    end_dt = (
        datetime.strptime(end_date, "%Y%m%d")
        if end_date
        else datetime.strptime(cal["end_date"].max(), "%Y%m%d")
    )

    srv_dates = _service_day_map(cal, start_dt, end_dt)
    if (cal_dates := gtfs.get("calendar_dates")) is not None:
        _apply_calendar_exceptions(cal_dates, srv_dates)

    od_full = _expand_with_dates(od, srv_dates)

    if not include_geometry:
        return od_full

    # 3. Append LineString geometries
    stops = gtfs["stops"].set_index("stop_id")
    line_geoms = [
        LineString([stops.loc[o].geometry, stops.loc[d].geometry])
        if o in stops.index and d in stops.index
        else None
        for o, d in zip(od_full["orig_stop_id"], od_full["dest_stop_id"], strict=False)
    ]
    return gpd.GeoDataFrame(od_full, geometry=line_geoms, crs="EPSG:4326")


# ---------------------------------------------------------------------------
# TRAVEL SUMMARY GRAPH
# ---------------------------------------------------------------------------


def travel_summary_graph(
    gtfs: dict[str, pd.DataFrame | gpd.GeoDataFrame],
    start_time: str | None = None,
    end_time: str | None = None,
    calendar_start: str | None = None,
    calendar_end: str | None = None,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """Aggregate stop-to-stop travel time & frequency into an edge list.

    Parameters
    ----------
    gtfs : dict
        A dictionary produced by :func:`load_gtfs` - must contain at least
        ``stop_times`` and ``stops``.
    start_time, end_time : str or None, optional
        Consider only trips whose **departure** falls inside
        ``[start_time, end_time]`` (format *HH:MM:SS*).  When *None* the whole
        service day is used.
    calendar_start, calendar_end : str or None, optional
        Period over which service-days are counted (format *YYYYMMDD*).  If
        omitted it spans the native range in ``calendar.txt``.
    as_nx : bool, default False
        If *True* return a NetworkX graph, otherwise two GeoDataFrames
        ``(nodes_gdf, edges_gdf)``.  The latter follow the convention used in
        *utils.py*.

    Returns
    -------
    (nodes_gdf, edges_gdf) or networkx.Graph
        • **Nodes** - every stop with a valid geometry.
        • **Edges** - columns = ``from_stop_id, to_stop_id, travel_time,
          frequency, geometry``.

    Examples
    --------
    >>> gtfs = load_gtfs("data/rome_gtfs.zip")
    >>> nodes, edges = travel_summary_graph(
    ...     gtfs,
    ...     start_time="07:00:00",
    ...     end_time="10:00:00",
    ... )
    >>> print(edges.head(3)[['travel_time', 'frequency']])
                       travel_time  frequency
    from_stop_id to_stop_id
    7045490      7045491        120.0        42
    7045491      7045492        180.0        42
    7045492      7045493        240.0        42

    You can directly obtain a NetworkX object too:

    >>> G = travel_summary_graph(gtfs, as_nx=True)
    >>> print(G.number_of_nodes(), G.number_of_edges())
    2564 3178
    """
    # 1. Validate Required Data
    required_keys = {"stop_times", "stops"}
    if not required_keys.issubset(gtfs.keys()):
        missing = required_keys - gtfs.keys()
        msg = f"GTFS must contain at least stop_times and stops. Missing: {', '.join(missing)}"
        raise ValueError(msg)

    # 2. Preprocess Data
    stop_times = gtfs["stop_times"].copy()
    trips = gtfs["trips"][["trip_id", "service_id"]].copy()

    stop_times["departure_time_sec"] = stop_times["departure_time"].map(_time_to_seconds)
    stop_times["arrival_time_sec"] = stop_times["arrival_time"].map(_time_to_seconds)

    # 3. Filter by Time of Day
    if start_time is not None:
        stop_times = stop_times[stop_times["departure_time_sec"] >= _time_to_seconds(start_time)]
    if end_time is not None:
        stop_times = stop_times[stop_times["arrival_time_sec"] <= _time_to_seconds(end_time)]

    # 4. Merge Data and Calculate Service Counts
    stop_times = stop_times.merge(trips, on="trip_id", how="inner")

    if calendar_start and calendar_end:
        service_counts = _get_service_counts(gtfs, calendar_start, calendar_end)
        stop_times["service_count"] = stop_times["service_id"].map(service_counts).fillna(0)
        stop_times = stop_times[stop_times["service_count"] > 0]
    else:
        stop_times["service_count"] = 1

    # 5. Calculate Travel Time Between Consecutive Stops
    stop_times = stop_times.sort_values(["trip_id", "stop_sequence"])
    grouped = stop_times.groupby("trip_id")

    valid_pairs = stop_times.assign(
        next_stop_id=grouped["stop_id"].shift(-1),
        next_arrival_time_sec=grouped["arrival_time_sec"].shift(-1),
    ).dropna(subset=["next_stop_id", "next_arrival_time_sec"])

    valid_pairs = valid_pairs.assign(
        travel_time=valid_pairs["next_arrival_time_sec"] - valid_pairs["departure_time_sec"],
    )
    valid_pairs = valid_pairs[valid_pairs["travel_time"] > 0]

    # 6. Aggregate Results
    agg_calcs = valid_pairs.groupby(["stop_id", "next_stop_id"]).agg(
        weighted_time_sum=("travel_time", lambda x: (x * valid_pairs.loc[x.index, "service_count"]).sum()),
        total_service_count=("service_count", "sum"),
    ).reset_index()

    agg_calcs["travel_time"] = agg_calcs["weighted_time_sum"] / agg_calcs["total_service_count"]
    agg_calcs["frequency"] = agg_calcs["total_service_count"]

    # 7. Create GeoDataFrames for nodes and edges
    nodes_gdf: gpd.GeoDataFrame = gtfs["stops"].set_index("stop_id")
    edges_geom = [
        LineString([nodes_gdf.loc[u].geometry, nodes_gdf.loc[v].geometry])
        for u, v in zip(agg_calcs["stop_id"], agg_calcs["next_stop_id"], strict=False)
    ]
    edges_gdf = gpd.GeoDataFrame(
        {
            "from_stop_id": agg_calcs["stop_id"],
            "to_stop_id": agg_calcs["next_stop_id"],
            "travel_time": agg_calcs["travel_time"],
            "frequency": agg_calcs["frequency"],
        },
        geometry=edges_geom,
        crs=nodes_gdf.crs,
    ).set_index(["from_stop_id", "to_stop_id"])

    return (nodes_gdf, edges_gdf) if not as_nx else gdf_to_nx(nodes_gdf, edges_gdf)
