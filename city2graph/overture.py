"""Module for loading and processing geospatial data from Overture Maps."""

import json
import logging
import subprocess
from pathlib import Path

import geopandas as gpd
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
VALID_OVERTURE_TYPES = {
    "address", "bathymetry", "building", "building_part", "division",
    "division_area", "division_boundary", "place", "segment", "connector",
    "infrastructure", "land", "land_cover", "land_use", "water",
}

logger = logging.getLogger(__name__)
WGS84_CRS = "EPSG:4326"


def _validate_and_prepare_area(area, types):
    """Validate types and prepare area for processing."""
    # Validate types
    if types is None:
        types = list(VALID_OVERTURE_TYPES)
    else:
        invalid_types = [t for t in types if t not in VALID_OVERTURE_TYPES]
        if invalid_types:
            raise ValueError(f"Invalid Overture Maps data type(s): {invalid_types}. "
                           f"Valid types are: {sorted(VALID_OVERTURE_TYPES)}")

    # Prepare area
    if isinstance(area, Polygon):
        original_polygon = area
        if hasattr(area, "crs") and area.crs and area.crs != WGS84_CRS:
            original_polygon = area.to_crs(WGS84_CRS)
            logger.info("Transformed polygon from %s to WGS84", area.crs)

        minx, miny, maxx, maxy = original_polygon.bounds
        bbox = [round(coord, 10) for coord in [minx, miny, maxx, maxy]]
        return types, bbox, original_polygon
    return types, area, None


def _download_and_process_data(data_type, bbox_str, output_dir, prefix, save_to_file, return_data, original_polygon):
    """Download and process a single overture data type."""
    # Validate inputs
    if data_type not in VALID_OVERTURE_TYPES:
        raise ValueError(f"Invalid data type: {data_type}")

    try:
        bbox_parts = [float(part.strip()) for part in bbox_str.split(",")]
        if len(bbox_parts) != 4:
            raise ValueError("Bbox must have 4 coordinates")
        safe_bbox_str = ",".join(map(str, bbox_parts))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid bbox format: {e}") from e

    # Prepare output path
    safe_output_dir = Path(output_dir).resolve()
    safe_prefix = Path(prefix).name if prefix else ""
    output_filename = f"{safe_prefix}{data_type}.geojson" if safe_prefix else f"{data_type}.geojson"
    output_path = safe_output_dir / output_filename

    # Build command
    cmd = ["overturemaps", "download", f"--bbox={safe_bbox_str}", "-f", "geojson", f"--type={data_type}"]
    if save_to_file:
        cmd.extend(["-o", str(output_path)])

    try:
        # Execute command
        process = subprocess.run(cmd, check=True,
                               stdout=subprocess.PIPE if not save_to_file else None, text=True)

        if not return_data:
            return None

        # Read data
        gdf = gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)
        if save_to_file and output_path.exists() and output_path.stat().st_size > 0:
            gdf = gpd.read_file(output_path)
        elif process.stdout and process.stdout.strip():
            try:
                gdf = gpd.read_file(process.stdout)
            except Exception as e:
                logger.warning("Could not parse GeoJSON for %s: %s", data_type, e)

        # Clip to polygon if provided
        if original_polygon is not None and not gdf.empty:
            try:
                mask = gpd.GeoDataFrame(geometry=[original_polygon], crs=WGS84_CRS)
                if gdf.crs != mask.crs:
                    mask = mask.to_crs(gdf.crs)
                gdf = gpd.clip(gdf, mask)
            except Exception as e:
                logger.warning("Error clipping %s to polygon: %s", data_type, e)

        if not gdf.empty:
            logger.info("Successfully processed %s", data_type)

        return gdf

    except subprocess.CalledProcessError as e:
        logger.warning("Error downloading %s: %s", data_type, e)
        return gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS) if return_data else None
    except Exception as e:
        logger.warning("Error processing %s data: %s", data_type, e)
        return gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS) if return_data else None


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

    Parameters
    ----------
    area : Union[List[float], Polygon]
        Either a bounding box as [min_lon, min_lat, max_lon, max_lat] in WGS84 coordinates
        or a Polygon in WGS84 coordinates (EPSG:4326). If provided in another CRS,
        it will be automatically transformed to WGS84.
    types : Optional[List[str]], default=None
        Types of data to download. If None, downloads all available types.
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
        Dictionary mapping types to GeoDataFrames if return_data is True

    Raises
    ------
    ValueError
        If any of the provided types are not valid Overture Maps data types
    """
    # Validate and prepare inputs
    types, bbox, original_polygon = _validate_and_prepare_area(area, types)

    if save_to_file and not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    bbox_str = ",".join(map(str, bbox))
    result = {}

    # Process each data type
    for data_type in types:
        gdf = _download_and_process_data(
            data_type, bbox_str, output_dir, prefix, save_to_file, return_data, original_polygon,
        )
        if return_data and gdf is not None:
            result[data_type] = gdf

    return result


def _parse_json_safely(json_str: str, default=None):
    """Parse JSON string safely with fallback."""
    if not json_str or not isinstance(json_str, str) or not json_str.strip():
        return default
    try:
        return json.loads(json_str.replace("'", '"').replace("None", "null"))
    except (json.JSONDecodeError, TypeError):
        return default


def _get_line_substring(line: LineString, start_pct: float, end_pct: float) -> LineString | None:
    """Extract substring of a line between start_pct and end_pct."""
    if not isinstance(line, LineString) or not (0 <= start_pct < end_pct <= 1):
        return None

    if start_pct < 1e-9 and end_pct > 1 - 1e-9:
        return line

    try:
        return substring(line, start_pct, end_pct, normalized=True)
    except Exception as e:
        logger.warning("Error creating line substring: %s", e)
        return None


def _compute_barrier_mask(level_rules: str) -> list:
    """Compute non-barrier intervals from level_rules JSON."""
    rules = _parse_json_safely(level_rules, [])
    if not rules:
        return [[0.0, 1.0]]

    if not isinstance(rules, list):
        rules = [rules]

    # Extract barrier intervals (where value != 0)
    barrier_intervals = []
    for rule in rules:
        if isinstance(rule, dict) and rule.get("value") != 0:
            between = rule.get("between")
            if between is None:
                return []  # Entire segment is a barrier
            if isinstance(between, list) and len(between) == 2:
                barrier_intervals.append(tuple(map(float, between)))

    if not barrier_intervals:
        return [[0.0, 1.0]]

    # Compute non-barrier intervals (complement of barriers)
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


def _extract_barrier_geometry(geometry: BaseGeometry, barrier_mask: list) -> BaseGeometry | None:
    """Extract barrier geometry from line using barrier mask."""
    if not barrier_mask or geometry is None or geometry.is_empty:
        return None

    if barrier_mask == [[0.0, 1.0]]:
        return geometry

    def extract_parts_from_line(line, mask):
        parts = []
        for interval in mask:
            seg = _get_line_substring(line, interval[0], interval[1])
            if seg and not seg.is_empty:
                parts.append(seg)
        return parts

    try:
        if isinstance(geometry, MultiLineString):
            all_parts = []
            for geom in geometry.geoms:
                parts = extract_parts_from_line(geom, barrier_mask)
                all_parts.extend(parts)

            if not all_parts:
                return None
            return all_parts[0] if len(all_parts) == 1 else MultiLineString(all_parts)

        if isinstance(geometry, LineString):
            parts = extract_parts_from_line(geometry, barrier_mask)
            if not parts:
                return None
            return parts[0] if len(parts) == 1 else MultiLineString(parts)

        return None
    except Exception:
        return None


def _compute_connector_mask(connectors_info: str) -> list:
    """Parse connectors_info and return connector positions."""
    parsed = _parse_json_safely(connectors_info, [])
    if not parsed:
        return [0.0, 1.0]

    if isinstance(parsed, dict):
        parsed = [parsed]
    elif not isinstance(parsed, list):
        return [0.0, 1.0]

    # Extract valid connector positions
    positions = set()
    for item in parsed:
        if isinstance(item, dict) and "at" in item:
            try:
                positions.add(float(item["at"]))
            except (ValueError, TypeError):
                continue

    if not positions:
        return [0.0, 1.0]

    sorted_positions = sorted(positions)
    return [0.0] + sorted_positions + [1.0]


def _recalculate_barrier_mask(original_mask: list, sub_start: float, sub_end: float) -> list:
    """Recalculate barrier mask for a subsegment."""
    if original_mask == [[0.0, 1.0]] or not original_mask:
        return original_mask

    new_mask = []
    seg_length = sub_end - sub_start

    for interval in original_mask:
        inter_start = max(interval[0], sub_start)
        inter_end = min(interval[1], sub_end)
        if inter_start < inter_end:
            new_mask.append([
                (inter_start - sub_start) / seg_length,
                (inter_end - sub_start) / seg_length,
            ])

    return new_mask


def _split_segment_by_connectors(row: pd.Series, valid_connector_ids: set) -> list[pd.Series]:
    """Split a single segment by its connectors."""
    geometry = row.geometry
    connectors_info = row.get("connectors", "")

    # Parse connector information
    connectors = _parse_json_safely(connectors_info, [])
    if not connectors:
        row["split_from"] = 0.0
        row["split_to"] = 1.0
        return [row]

    if isinstance(connectors, dict):
        connectors = [connectors]

    # Extract valid connector positions
    valid_positions = []
    for connector in connectors:
        if (isinstance(connector, dict) and
            connector.get("connector_id") in valid_connector_ids and
            "at" in connector):
            try:
                valid_positions.append(float(connector["at"]))
            except (ValueError, TypeError):
                continue

    if not valid_positions:
        row["split_from"] = 0.0
        row["split_to"] = 1.0
        return [row]

    # Create split segments
    valid_positions = sorted(set(valid_positions))
    split_rows = []
    original_id = row.get("id", row.name)
    barrier_mask = row.get("barrier_mask", [[0.0, 1.0]])

    # Split at each connector position
    start_pct = 0.0
    for i, end_pct in enumerate(valid_positions):
        if end_pct > start_pct:
            part = _get_line_substring(geometry, start_pct, end_pct)
            if part and not part.is_empty:
                new_row = row.copy()
                new_row.geometry = part
                new_row["split_from"] = start_pct
                new_row["split_to"] = end_pct
                new_row["barrier_mask"] = _recalculate_barrier_mask(barrier_mask, start_pct, end_pct)
                new_row["id"] = f"{original_id}_{i+1}"
                split_rows.append(new_row)
        start_pct = end_pct

    # Add final segment if needed
    if start_pct < 1.0:
        part = _get_line_substring(geometry, start_pct, 1.0)
        if part and not part.is_empty:
            new_row = row.copy()
            new_row.geometry = part
            new_row["split_from"] = start_pct
            new_row["split_to"] = 1.0
            new_row["barrier_mask"] = _recalculate_barrier_mask(barrier_mask, start_pct, 1.0)
            new_row["id"] = f"{original_id}_{len(split_rows)+1}"
            split_rows.append(new_row)

    return split_rows if split_rows else [row]


def _split_segments_by_connectors(segments_gdf: gpd.GeoDataFrame, connectors_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Split segments at connector points and update barrier masks."""
    valid_connector_ids = set(connectors_gdf["id"])

    # Pre-compute barrier masks
    segments_gdf["barrier_mask"] = segments_gdf["level_rules"].fillna("").astype(str).apply(_compute_barrier_mask)

    # Process segments in batches
    all_split_rows = []
    batch_size = 1000

    for i in range(0, len(segments_gdf), batch_size):
        batch = segments_gdf.iloc[i:i + batch_size]
        for _, row in batch.iterrows():
            split_rows = _split_segment_by_connectors(row, valid_connector_ids)
            all_split_rows.extend(split_rows)

    return gpd.GeoDataFrame(all_split_rows, crs=segments_gdf.crs).reset_index(drop=True)


def _adjust_segment_connectors(segments_gdf: gpd.GeoDataFrame, threshold: float) -> gpd.GeoDataFrame:
    """Adjust segment endpoints by clustering nearby endpoints."""
    # Filter to only process LineString geometries
    linestring_mask = segments_gdf.geometry.type == "LineString"
    if not linestring_mask.any():
        return segments_gdf

    # Extract endpoints from all LineStrings
    endpoints_data = []
    for idx, geom in segments_gdf.loc[linestring_mask, "geometry"].items():
        coords = list(geom.coords)
        if len(coords) >= 2:
            endpoints_data.extend([
                {"seg_id": idx, "pos": "start", "x": coords[0][0], "y": coords[0][1]},
                {"seg_id": idx, "pos": "end", "x": coords[-1][0], "y": coords[-1][1]},
            ])

    if not endpoints_data:
        return segments_gdf

    endpoints_df = pd.DataFrame(endpoints_data)

    # Quantize coordinates to bins and compute centroids
    endpoints_df["bin_x"] = (endpoints_df["x"] / threshold).round().astype(int)
    endpoints_df["bin_y"] = (endpoints_df["y"] / threshold).round().astype(int)

    # Calculate centroids for each bin
    bin_centroids = endpoints_df.groupby(["bin_x", "bin_y"])[["x", "y"]].mean()
    endpoints_df = endpoints_df.merge(
        bin_centroids, on=["bin_x", "bin_y"], suffixes=("", "_centroid"),
    )

    # Create lookup for new endpoint coordinates
    endpoint_lookup = {}
    for _, row in endpoints_df.iterrows():
        endpoint_lookup[(row["seg_id"], row["pos"])] = (row["x_centroid"], row["y_centroid"])

    # Rebuild geometries with adjusted endpoints
    def rebuild_geometry(row):
        geom = row.geometry
        if not isinstance(geom, LineString):
            return geom

        coords = list(geom.coords)
        if len(coords) < 2:
            return geom

        seg_id = row.name
        start_coord = endpoint_lookup.get((seg_id, "start"), coords[0])
        end_coord = endpoint_lookup.get((seg_id, "end"), coords[-1])

        return LineString([start_coord] + coords[1:-1] + [end_coord])

    segments_gdf.loc[linestring_mask, "geometry"] = segments_gdf.loc[linestring_mask].apply(rebuild_geometry, axis=1)
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
    # Ensure level_rules column exists
    if "level_rules" not in segments_gdf.columns:
        segments_gdf["level_rules"] = ""

    # Split segments by connectors if provided
    if connectors_gdf is not None and not connectors_gdf.empty:
        segments_gdf = _split_segments_by_connectors(segments_gdf, connectors_gdf)
        segments_gdf = _adjust_segment_connectors(segments_gdf, threshold)

    # Add length column
    segments_gdf["length"] = segments_gdf.geometry.length

    # Generate barrier geometry if requested
    if get_barriers:
        # Compute barrier masks if not already done
        if "barrier_mask" not in segments_gdf.columns:
            segments_gdf["barrier_mask"] = segments_gdf["level_rules"].fillna("").astype(str).apply(_compute_barrier_mask)

        # Generate barrier geometries
        def get_barrier_geometry(row):
            if "barrier_mask" not in row:
                return None
            return _extract_barrier_geometry(row.geometry, row["barrier_mask"])

        barrier_geoms = segments_gdf.apply(get_barrier_geometry, axis=1)
        segments_gdf["barrier_geometry"] = gpd.GeoSeries(barrier_geoms, crs=segments_gdf.crs)

    return segments_gdf
