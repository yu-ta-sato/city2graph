"""Module for loading and processing geospatial data from Overture Maps."""

import json
import logging
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

# Define the public API for this module
__all__ = [
    "load_overture_data",
    "process_overture_segments",
]

# Valid Overture Maps data types
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

logger = logging.getLogger(__name__)


def _validate_overture_types(types: list[str] | None) -> list[str]:
    """Validate and return overture data types."""
    if types is None:
        return list(VALID_OVERTURE_TYPES)

    invalid_types = [t for t in types if t not in VALID_OVERTURE_TYPES]
    if invalid_types:
        msg = (
            f"Invalid Overture Maps data type(s): {invalid_types}. "
            f"Valid types are: {sorted(VALID_OVERTURE_TYPES)}"
        )
        raise ValueError(msg)
    return types


def _prepare_polygon_area(area: Polygon) -> tuple[list[float], Polygon | None]:
    """Transform polygon to WGS84 and extract bounding box."""
    wgs84_crs = "EPSG:4326"
    original_polygon = area

    if hasattr(area, "crs") and area.crs and area.crs != wgs84_crs:
        # Reproject polygon to WGS84
        original_polygon = area.to_crs(wgs84_crs)
        logger.info("Transformed polygon from %s to WGS84 (EPSG:4326)", area.crs)

    # Extract and round bounding box coordinates
    minx, miny, maxx, maxy = original_polygon.bounds
    bbox = [round(minx, 10), round(miny, 10), round(maxx, 10), round(maxy, 10)]
    return bbox, original_polygon


def _read_overture_data(
    output_path: str, process: subprocess.CompletedProcess, save_to_file: bool, data_type: str,
) -> gpd.GeoDataFrame:
    """Read data from file or stdout and return GeoDataFrame."""
    WGS84_CRS = "EPSG:4326"

    if save_to_file:
        if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
            return gpd.read_file(output_path)
        logger.warning("No data returned for %s", data_type)

    if process.stdout and process.stdout.strip():
        try:
            return gpd.read_file(process.stdout)
        except (ValueError, TypeError, KeyError, UnicodeDecodeError) as e:
            logger.warning("Could not parse GeoJSON for %s: %s", data_type, e)

    return gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)


def _clip_to_polygon(gdf: gpd.GeoDataFrame, polygon: Polygon, data_type: str) -> gpd.GeoDataFrame:
    """Clip GeoDataFrame to polygon boundaries."""
    WGS84_CRS = "EPSG:4326"

    if polygon is None or gdf.empty:
        return gdf

    mask = gpd.GeoDataFrame(geometry=[polygon], crs=WGS84_CRS)
    if gdf.crs != mask.crs:
        mask = mask.to_crs(gdf.crs)

    try:
        return gpd.clip(gdf, mask)
    except (ValueError, AttributeError, RuntimeError) as e:
        logger.warning("Error clipping %s to polygon: %s", data_type, e)
        return gdf


def _process_single_overture_type(
    data_type: str,
    bbox_str: str,
    output_dir: str,
    prefix: str,
    save_to_file: bool,
    return_data: bool,
    original_polygon: Polygon | None,
) -> gpd.GeoDataFrame | None:
    """Process a single overture data type."""
    WGS84_CRS = "EPSG:4326"

    def _raise_invalid_data_type(data_type: str) -> None:
        """Raise ValueError for invalid data type."""
        msg = f"Invalid data type: {data_type}"
        raise ValueError(msg)

    def _raise_invalid_bbox_format(error_msg: str = "Invalid bbox format") -> None:
        """Raise ValueError for invalid bbox format."""
        raise ValueError(error_msg)

    # Validate data_type against known safe values to prevent injection
    if data_type not in VALID_OVERTURE_TYPES:
        _raise_invalid_data_type(data_type)

    # Validate and sanitize bbox_str to prevent injection
    try:
        bbox_parts = bbox_str.split(",")
        if len(bbox_parts) != 4:
            _raise_invalid_bbox_format()
        # Validate that all parts are valid floats
        validated_bbox = [float(part.strip()) for part in bbox_parts]
        safe_bbox_str = ",".join(map(str, validated_bbox))
    except (ValueError, TypeError) as e:
        msg = f"Invalid bbox format: {e}"
        raise ValueError(msg) from e

    # Validate output directory and prefix to prevent path traversal
    safe_output_dir = Path(output_dir).resolve()
    safe_prefix = Path(prefix).name if prefix else ""

    output_filename = f"{safe_prefix}{data_type}.geojson" if safe_prefix else f"{data_type}.geojson"
    output_path = Path(safe_output_dir) / output_filename

    cmd_parts = [
        "overturemaps", "download", f"--bbox={safe_bbox_str}",
        "-f", "geojson", f"--type={data_type}",
    ]

    if save_to_file:
        cmd_parts.extend(["-o", str(output_path)])

    try:
        process = subprocess.run(
            cmd_parts,
            check=True,
            stdout=subprocess.PIPE if not save_to_file else None,
            text=True,
        )

        if not return_data:
            return None

        gdf = _read_overture_data(output_path, process, save_to_file, data_type)
        gdf = _clip_to_polygon(gdf, original_polygon, data_type)

        if gdf.empty and "geometry" not in gdf:
            gdf = gpd.GeoDataFrame(geometry=[], crs=gdf.crs or WGS84_CRS)

        # Successfully processed data type
        if not gdf.empty:
            logger.warning("Successfully processed %s", data_type)

    except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        logger.warning("Error processing %s data: %s", data_type, e)
        return gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS) if return_data else None
    except subprocess.CalledProcessError as e:
        logger.warning("Error downloading %s: %s", data_type, e)
        return gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS) if return_data else None
    else:
        return gdf


def load_overture_data(
    area: list[float] | Polygon,
    types: list[str] | None = None,
    output_dir: str = ".",
    prefix: str = "",
    save_to_file: bool = True,
    return_data: bool = True,
) -> dict[str, gpd.GeoDataFrame]:
    """
    Load data from Overture Maps using the CLI tool and optionally save to GeoJSON files.

    Can accept either a bounding box or a Polygon as the area parameter.

    Parameters
    ----------
    area : Union[List[float], Polygon]
        Either a bounding box as [min_lon, min_lat, max_lon, max_lat] in WGS84 coordinates
        or a Polygon in WGS84 coordinates (EPSG:4326). If provided in another CRS,
        it will be automatically transformed to WGS84.
        If a Polygon is provided, its bounding box will be used for the query and
        the results will be clipped to the Polygon boundaries.
    types : Optional[List[str]], default=None
        Types of data to download. If None, downloads all available types.
        Must be valid Overture Maps data types: address, bathymetry, building,
        building_part, division, division_area, division_boundary, place, segment,
        connector, infrastructure, land, land_cover, land_use, water.
    output_dir : str, default="."
        Directory to save the GeoJSON files
    prefix : str, default=""
        Prefix to add to the output filenames
    save_to_file : bool, default=True
        Whether to save the data to GeoJSON files
    return_data : bool, default=True
        Whether to return the data as GeoDataFrames

    Returns
    -------
    Dict[str, gpd.GeoDataFrame]
        Dictionary mapping types to GeoDataFrames if return_data is True,
        otherwise an empty dict

    Raises
    ------
    ValueError
        If any of the provided types are not valid Overture Maps data types

    Notes
    -----
    The Overture Maps API requires coordinates in WGS84 (EPSG:4326) format.
    For more information, see https://docs.overturemaps.org/
    """
    types = _validate_overture_types(types)

    if save_to_file and not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    if isinstance(area, Polygon):
        bbox, original_polygon = _prepare_polygon_area(area)
    else:
        bbox, original_polygon = area, None

    bbox_str = ",".join(map(str, bbox))
    result = {}

    for data_type in types:
        gdf = _process_single_overture_type(
            data_type, bbox_str, output_dir, prefix,
            save_to_file, return_data, original_polygon,
        )
        if return_data and gdf is not None:
            result[data_type] = gdf

    return result


def _get_substring(
    line: LineString, start_pct: float, end_pct: float,
) -> LineString | None:
    """
    Extract substring of a line between start_pct and end_pct.

    Parameters
    ----------
    line : LineString
        The input line
    start_pct : float
        Start percentage (0-1)
    end_pct : float
        End percentage (0-1)

    Returns
    -------
    Optional[LineString]
        The substring or None if invalid
    """
    # Validate input parameters
    if not isinstance(line, LineString) or not (0 <= start_pct < end_pct <= 1):
        return None

    # For full line or nearly full line, return the original
    if start_pct < 1e-9 and end_pct > 1 - 1e-9:
        return line

    try:
        return substring(line, start_pct, end_pct, normalized=True)
    except (ValueError, AttributeError, TypeError) as e:
        logger.warning("Error creating line substring: %s", e)
        return None


def _identify_barrier_mask(level_rules: str) -> list:
    """
    Compute non-barrier intervals (barrier mask) from level_rules JSON.

    Only rules with "value" equal to 0 are considered as barriers.
    If any such rule has "between" equal to null, then the entire interval [0, 1]
    is treated as non-barrier.

    Parameters
    ----------
    level_rules : str
        JSON string containing level rules with "value" and "between" fields.
        Example: '[{"value": 0, "between": [0.177, 0.836]}]'

    Returns
    -------
    list
        List of non-barrier intervals as [start, end] pairs.
        Each interval represents a continuous non-barrier section.

    Examples
    --------
    >>> level_rules = '[{"value": 0, "between": [0.177, 0.836]}, {"value": 0, "between": [0.957, 0.959]}]'
    >>> _identify_barrier_mask(level_rules)
    [[0.0, 0.177], [0.836, 0.957], [0.959, 1.0]]

    Notes
    -----
    If any rule for which "value" equals 0 has "between" as null, then
    the function returns [[0.0, 1.0]].

    The barrier intervals are extracted from rules where "value" != 0,
    and the returned intervals represent the complement (non-barrier sections).
    """
    if not isinstance(level_rules, str) or not level_rules.strip():
        return [[0.0, 1.0]]

    try:
        rules = json.loads(level_rules.replace("'", '"').replace("None", "null"))
        if not isinstance(rules, list):
            rules = [rules]
    except (json.JSONDecodeError, TypeError):
        logger.warning("JSON parse failed for level_rules: %s", level_rules)
        return [[0.0, 1.0]]

    barrier_intervals = []
    for rule in rules:
        if isinstance(rule, dict) and rule.get("value") != 0:
            between = rule.get("between")
            if between is None:
                return []  # This implies the entire segment is a barrier
            if isinstance(between, list) and len(between) == 2:
                barrier_intervals.append(tuple(map(float, between)))

    if not barrier_intervals:
        return [[0.0, 1.0]]

    barrier_intervals.sort()

    non_barrier_mask = []
    current = 0.0
    for start, end in barrier_intervals:
        if start > current:
            non_barrier_mask.append([current, start])
        current = max(current, end)

    if current < 1.0:
        non_barrier_mask.append([current, 1.0])

    return non_barrier_mask


def _extract_barriers_from_mask(line: LineString, mask: list) -> BaseGeometry | None:
    """
    Extract barrier parts from the line using the provided barrier mask.

    The mask is expected to be a list of [start, end] intervals.
    """
    parts = []
    for interval in mask:
        seg = _get_substring(line, interval[0], interval[1])
        if seg and not seg.is_empty:
            parts.append(seg)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return MultiLineString(parts)


def _get_barrier_geometry(row: pd.Series) -> BaseGeometry | None:
    if "barrier_mask" not in row:
        msg = "Column 'barrier_mask' not found in input row"
        raise KeyError(msg)
    barrier_mask = row["barrier_mask"]

    if not barrier_mask:
        return None

    if barrier_mask == [[0.0, 1.0]]:
        return row.geometry

    geom = row.geometry
    if geom is None or geom.is_empty:
        return None

    try:
        if isinstance(geom, MultiLineString):
            parts = [
                part
                for g in geom.geoms
                if (part := _extract_barriers_from_mask(g, barrier_mask))
            ]
            if not parts:
                return None
            if len(parts) == 1:
                return parts[0]
            return MultiLineString(parts)

        if isinstance(geom, LineString):
            return _extract_barriers_from_mask(geom, barrier_mask)

        return None

    except (ValueError, AttributeError, TypeError):
        return None


def _identify_connector_mask(connectors_info: str) -> list:
    """
    Parse connectors_info and return a connector mask list.

    Parameters
    ----------
    connectors_info : str
        JSON string containing connector information with "at" fields.
        Example: '[{"connector_id": "123", "at": 0.5}]'

    Returns
    -------
    list
        List of floats starting with 0.0 and ending with 1.0.
        If connectors_info is empty or invalid, returns [0.0, 1.0].

    Examples
    --------
    >>> connectors_info = '[{"connector_id": "123", "at": 0.3}, {"connector_id": "456", "at": 0.7}]'
    >>> _identify_connector_mask(connectors_info)
    [0.0, 0.3, 0.7, 1.0]
    """
    if not connectors_info or not isinstance(connectors_info, str) or not connectors_info.strip():
        return [0.0, 1.0]
    try:
        parsed = json.loads(connectors_info.replace("'", '"'))
        if isinstance(parsed, dict):
            connectors_list = [parsed]
        elif isinstance(parsed, list):
            connectors_list = parsed
        else:
            return [0.0, 1.0]

        valid_ps = {
            float(item["at"])
            for item in connectors_list
            if isinstance(item, dict) and "at" in item
        }
        sorted_ps = sorted(valid_ps)
        return [0.0, *sorted_ps, 1.0]

    except (json.JSONDecodeError, ValueError, TypeError, KeyError):
        return [0.0, 1.0]


def _recalc_barrier_mask(original_mask: list, sub_start: float, sub_end: float) -> list:
    """Recalculate barrier_mask for a subsegment defined by [sub_start, sub_end]."""
    if original_mask == [[0.0, 1.0]] or not original_mask:
        return original_mask
    new_mask = []
    seg_length = sub_end - sub_start
    for interval in original_mask:
        inter_start = max(interval[0], sub_start)
        inter_end = min(interval[1], sub_end)
        if inter_start < inter_end:
            new_mask.append(
                [
                    (inter_start - sub_start) / seg_length,
                    (inter_end - sub_start) / seg_length,
                ],
            )
    return new_mask


def _parse_connectors_info(connectors_info: str | None) -> list[dict]:
    """Parse and validate connectors info from row data."""
    if not connectors_info or not isinstance(connectors_info, str) or not connectors_info.strip():
        return []

    try:
        parsed = json.loads(str(connectors_info).replace("'", '"'))
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return parsed
        return []
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def _extract_valid_connectors(connectors_list: list[dict], valid_ids: set) -> list[float]:
    """Extract valid connector positions from connector list."""
    valid_connectors = set()
    for item in connectors_list:
        if not isinstance(item, dict):
            continue

        connector_id = item.get("connector_id")
        at_value = item.get("at")

        if connector_id is None or at_value is None or connector_id not in valid_ids:
            continue

        valid_connectors.add(float(at_value))

    return sorted(valid_connectors)


def _create_connector_mask(valid_connectors: list[float]) -> list[float]:
    """Create connector mask from valid connector positions."""
    mask = []
    if not valid_connectors or valid_connectors[0] != 0.0:
        mask.append(0.0)
    mask.extend(valid_connectors)
    if not mask or mask[-1] != 1.0:
        mask.append(1.0)
    return mask


def _create_split_row(row: pd.Series,
                      part: LineString,
                      start_pct: float,
                      end_pct: float,
                      mask: list[float],
                      barrier_mask: list,
                      original_id: str | int,
                      counter: int) -> pd.Series:
    """Create a new row for a split segment part."""
    new_row = row.copy()
    new_row.geometry = part
    new_row["split_from"] = start_pct
    new_row["split_to"] = end_pct
    new_row["connector_mask"] = mask
    new_row["barrier_mask"] = _recalc_barrier_mask(barrier_mask, start_pct, end_pct)
    new_row["id"] = f"{original_id}_{counter}"
    return new_row


def _process_segment(row: pd.Series, valid_ids: set) -> list[pd.Series]:
    """
    Process a single segment row for splitting by connectors.

    Parameters
    ----------
    row : pd.Series
        A row from the segments GeoDataFrame
    valid_ids : set
        Set of valid connector IDs

    Returns
    -------
    list[pd.Series]
        List of new rows created from splitting the segment
    """
    geom = row.geometry
    connectors_info = row.get("connectors")

    # Parse connectors info
    connectors_list = _parse_connectors_info(connectors_info)
    if not connectors_list:
        row["split_from"] = 0.0
        row["split_to"] = 1.0
        return [row]

    # Extract valid connectors
    valid_connectors = _extract_valid_connectors(connectors_list, valid_ids)
    if not valid_connectors:
        row["split_from"] = 0.0
        row["split_to"] = 1.0
        return [row]

    # Create connector mask
    mask = _create_connector_mask(valid_connectors)

    # Generate split geometries
    split_rows = []
    start_pct = 0.0
    counter = 1
    original_id = row.get("id", row.name)
    barrier_mask = row["barrier_mask"]

    # Process each connector split
    split_points = sorted(valid_connectors)
    for end_pct in split_points:
        if end_pct > start_pct:
            part = _get_substring(geom, start_pct, end_pct)
            if part and not part.is_empty:
                split_row = _create_split_row(row, part, start_pct, end_pct, mask, barrier_mask, original_id, counter)
                split_rows.append(split_row)
                counter += 1
        start_pct = end_pct

    # Process the last segment
    if start_pct < 1.0:
        part = _get_substring(geom, start_pct, 1.0)
        if part and not part.is_empty:
            split_row = _create_split_row(row, part, start_pct, 1.0, mask, barrier_mask, original_id, counter)
            split_rows.append(split_row)

    return split_rows if split_rows else [row]


def _split_segments_by_connectors(
    segments_gdf: gpd.GeoDataFrame, connectors_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Split segments at connector points and update barrier masks accordingly.

    Optimized for performance with batch processing.

    Parameters
    ----------
    segments_gdf : gpd.GeoDataFrame
        GeoDataFrame containing segments to be split
    connectors_gdf : gpd.GeoDataFrame
        GeoDataFrame containing connector points

    Returns
    -------
    gpd.GeoDataFrame
        New GeoDataFrame with split segments
    """
    # Precompute valid connector ids for a fast membership check
    valid_ids = set(connectors_gdf["id"])

    # Pre-process connectors_info and level_rules for all rows at once
    segments_gdf["connector_mask"] = segments_gdf["connectors"].fillna("").astype(str).apply(_identify_connector_mask)
    segments_gdf["barrier_mask"] = segments_gdf["level_rules"].fillna("").astype(str).apply(_identify_barrier_mask)

    # Prepare data structures
    new_rows_data = []

    # Process segments in batches to reduce memory pressure
    batch_size = 1000
    for i in range(0, len(segments_gdf), batch_size):
        batch = segments_gdf.iloc[i : i + batch_size]
        batch_results = batch.apply(
            lambda row: _process_segment(row, valid_ids), axis=1,
        )
        for rows in batch_results:
            new_rows_data.extend(rows)

    # Create a new GeoDataFrame from all processed rows, include split columns
    result_gdf = gpd.GeoDataFrame(new_rows_data, crs=segments_gdf.crs)

    # Reset the index of the resulting GeoDataFrame
    return result_gdf.reset_index(drop=True)


def _rebuild_geometry(
    seg_id: str | int,
    geom: LineString,
    pivot_df: pd.DataFrame) -> list[tuple[float, float]]:
    """
    Rebuild the geometry of a segment by replacing its endpoints with quantized centroids.

    Parameters
    ----------
    seg_id : Any
        Identifier for the segment in the pivot_df
    geom : LineString
        Original geometry of the segment
    pivot_df : pd.DataFrame
        DataFrame containing quantized centroid coordinates for endpoints

    Returns
    -------
    List[Tuple[float, float]]
        List of coordinate tuples for the rebuilt geometry
    """
    start = pivot_df.loc[seg_id, ("x_centroid", "start")], pivot_df.loc[seg_id, ("y_centroid", "start")]
    end = pivot_df.loc[seg_id, ("x_centroid", "end")], pivot_df.loc[seg_id, ("y_centroid", "end")]
    coords = list(geom.coords)
    if len(coords) < 2:
        return [start, end]
    return [start, *coords[1:-1], end]


def _adjust_segment_connectors(
    segments_gdf: gpd.GeoDataFrame, threshold: float,
) -> gpd.GeoDataFrame:
    """
    Adjust segment connector endpoints by clustering endpoints within a threshold distance.

    This function identifies endpoints that are within a threshold distance of each other
    and replaces them with their cluster's centroid, creating more precise connections
    between LineString segments.

    Parameters
    ----------
    segments_gdf : gpd.GeoDataFrame
        GeoDataFrame containing segment geometries (LineStrings)
    threshold : float
        Distance threshold for clustering endpoints. Endpoints whose coordinates
        quantize to the same bin (based on this threshold) will be merged.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with adjusted LineString geometries where endpoints
        that were within the threshold have been merged to a common point

    Notes
    -----
    The function works by:
    1. Extracting start and end points from all LineStrings
    2. Quantizing coordinates to bins based on the threshold
    3. Computing the centroid for each bin
    4. Rebuilding LineStrings with the new endpoint coordinates

    Only LineString geometries are processed; other geometry types are left unchanged.
    """
    # Filter to only process LineString geometries
    mask = segments_gdf.geometry.type == "LineString"
    if not mask.any():
        return segments_gdf

    valid_gdf = segments_gdf.loc[mask].copy()
    valid_gdf["seg_id"] = valid_gdf.index

    # Extract start and end points from all LineStrings
    starts = valid_gdf.geometry.apply(lambda geom: geom.coords[0])
    ends = valid_gdf.geometry.apply(lambda geom: geom.coords[-1])

    # Create DataFrame with all endpoints for easier processing
    endpoints_df = pd.DataFrame(
        {
            "seg_id": np.concatenate([valid_gdf["seg_id"], valid_gdf["seg_id"]]),
            "pos": ["start"] * len(valid_gdf) + ["end"] * len(valid_gdf),
            "x": np.concatenate([starts.apply(lambda p: p[0]), ends.apply(lambda p: p[0])]),
            "y": np.concatenate([starts.apply(lambda p: p[1]), ends.apply(lambda p: p[1])]),
        },
    )

    # Quantize coordinates to bins based on threshold
    endpoints_df["bin_x"] = (endpoints_df["x"] / threshold).round().astype(int)
    endpoints_df["bin_y"] = (endpoints_df["y"] / threshold).round().astype(int)

    # Calculate centroids for each bin
    centroids = endpoints_df.groupby(["bin_x", "bin_y"])[["x", "y"]].transform("mean")
    endpoints_df["x_centroid"] = centroids["x"]
    endpoints_df["y_centroid"] = centroids["y"]

    # Pivot the dataframe to get centroid coordinates by segment and position
    pivot_df = endpoints_df.pivot_table(
        index="seg_id", columns="pos", values=["x_centroid", "y_centroid"],
    )

    # Rebuild geometries using the centroid coordinates
    new_geometries = valid_gdf.apply(
        lambda row: LineString(
            _rebuild_geometry(row["seg_id"], row.geometry, pivot_df),
        ),
        axis=1,
    )

    # Update the original GeoDataFrame with the new geometries
    segments_gdf.loc[mask, "geometry"] = new_geometries
    return segments_gdf


def process_overture_segments(
    segments_gdf: gpd.GeoDataFrame,
    get_barriers: bool = True,
    connectors_gdf: gpd.GeoDataFrame | None = None,
    threshold: float = 1.0,
) -> gpd.GeoDataFrame:
    """
    Process segments from Overture Maps to be split by connectors and extract barriers.

    Parameters
    ----------
    segments_gdf : gpd.GeoDataFrame
        Input segments with 'subtype' and 'level_rules'.
    get_barriers : bool
        If True, add 'barrier_geometry' column to output.
    connectors_gdf : Optional[gpd.GeoDataFrame]
        Connectors for splitting; if None connectors step is skipped.
    threshold : float
        Distance threshold for adjusting connectors.

    Returns
    -------
    gpd.GeoDataFrame
        Processed road segments, including 'length' and optional 'barrier_geometry'.
    """
    if "level_rules" not in segments_gdf.columns:
        segments_gdf["level_rules"] = None

    if get_barriers:
        segments_gdf["barrier_mask"] = segments_gdf["level_rules"].fillna("").astype(str).apply(_identify_barrier_mask)

    if connectors_gdf is not None and not connectors_gdf.empty:
        segments_gdf = _split_segments_by_connectors(segments_gdf, connectors_gdf)
        segments_gdf = _adjust_segment_connectors(segments_gdf, threshold=threshold)

    segments_gdf["length"] = segments_gdf.geometry.length

    if get_barriers:
        barrier_geoms = segments_gdf.apply(_get_barrier_geometry, axis=1)
        segments_gdf["barrier_geometry"] = gpd.GeoSeries(barrier_geoms, crs=segments_gdf.crs)

    return segments_gdf
