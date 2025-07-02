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
from shapely.ops import substring

__all__ = ["load_overture_data", "process_overture_segments"]

VALID_OVERTURE_TYPES = {
    "address", "bathymetry", "building", "building_part", "division",
    "division_area", "division_boundary", "place", "segment", "connector",
    "infrastructure", "land", "land_cover", "land_use", "water",
}

logger = logging.getLogger(__name__)
WGS84_CRS = "EPSG:4326"


def _process_overture_type(data_type, area, output_dir, prefix, save_to_file, return_data):
    """Process a single overture data type with integrated validation and processing."""
    # Validate data type
    if data_type not in VALID_OVERTURE_TYPES:
        raise ValueError(f"Invalid data type: {data_type}")

    # Prepare area and bbox
    if isinstance(area, Polygon):
        polygon = area
        if hasattr(area, "crs") and area.crs and area.crs != WGS84_CRS:
            polygon = area.to_crs(WGS84_CRS)
            logger.info("Transformed polygon from %s to WGS84", area.crs)
        bbox = [round(c, 10) for c in polygon.bounds]
    else:
        polygon = None
        bbox = area

    # Validate bbox format
    try:
        bbox_parts = [float(str(b).strip()) for b in bbox]
        if len(bbox_parts) != 4:
            raise ValueError("Bbox must have 4 coordinates")
        bbox_str = ",".join(map(str, bbox_parts))
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid bbox format: {e}") from e

    # Setup output path
    output_dir = Path(output_dir).resolve()
    prefix = Path(prefix).name if prefix else ""
    filename = f"{prefix}{data_type}.geojson" if prefix else f"{data_type}.geojson"
    output_path = output_dir / filename

    # Build and execute command
    cmd = ["overturemaps", "download", f"--bbox={bbox_str}", "-f", "geojson", f"--type={data_type}"]
    if save_to_file:
        cmd.extend(["-o", str(output_path)])

    try:
        result = subprocess.run(cmd, check=True, capture_output=not save_to_file, text=True)

        if not return_data:
            return None

        # Load data
        gdf = gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)
        if save_to_file and output_path.exists() and output_path.stat().st_size > 0:
            gdf = gpd.read_file(output_path)
        elif result.stdout and result.stdout.strip():
            try:
                gdf = gpd.read_file(result.stdout)
            except Exception as e:
                logger.warning("Could not parse GeoJSON for %s: %s", data_type, e)

        # Clip to polygon if needed
        if polygon is not None and not gdf.empty:
            try:
                mask = gpd.GeoDataFrame(geometry=[polygon], crs=WGS84_CRS)
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
        logger.warning("Error processing %s: %s", data_type, e)
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
    # Validate types
    if types is None:
        types = list(VALID_OVERTURE_TYPES)
    else:
        invalid_types = [t for t in types if t not in VALID_OVERTURE_TYPES]
        if invalid_types:
            raise ValueError(f"Invalid Overture Maps data type(s): {invalid_types}. "
                           f"Valid types are: {sorted(VALID_OVERTURE_TYPES)}")

    # Create output directory if needed
    if save_to_file and not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    # Process each data type
    result = {}
    for data_type in types:
        gdf = _process_overture_type(data_type, area, output_dir, prefix, save_to_file, return_data)
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


def _compute_barrier_mask_and_geometry(level_rules: str, geometry=None):
    """Compute barrier mask and optionally extract barrier geometry."""
    # Parse level rules
    rules = _parse_json_safely(level_rules, [])
    if not rules:
        mask = [[0.0, 1.0]]
        return mask, geometry if geometry else None

    if not isinstance(rules, list):
        rules = [rules]

    # Extract barrier intervals (where value != 0)
    barrier_intervals = []
    for rule in rules:
        if isinstance(rule, dict) and rule.get("value") != 0:
            between = rule.get("between")
            if between is None:
                return [], None  # Entire segment is a barrier
            if isinstance(between, list) and len(between) == 2:
                barrier_intervals.append(tuple(map(float, between)))

    if not barrier_intervals:
        mask = [[0.0, 1.0]]
        return mask, geometry if geometry else None

    # Compute non-barrier intervals (complement of barriers)
    barrier_intervals.sort()
    mask = []
    current = 0.0

    for start, end in barrier_intervals:
        if start > current:
            mask.append([current, start])
        current = max(current, end)

    if current < 1.0:
        mask.append([current, 1.0])

    # Extract barrier geometry if requested
    if geometry is None:
        return mask, None

    if not mask or geometry.is_empty:
        return mask, None

    if mask == [[0.0, 1.0]]:
        return mask, geometry

    # Extract parts based on mask
    def extract_parts(line):
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
                all_parts.extend(extract_parts(geom))
            if not all_parts:
                return mask, None
            barrier_geom = all_parts[0] if len(all_parts) == 1 else MultiLineString(all_parts)
        elif isinstance(geometry, LineString):
            parts = extract_parts(geometry)
            if not parts:
                return mask, None
            barrier_geom = parts[0] if len(parts) == 1 else MultiLineString(parts)
        else:
            barrier_geom = None

        return mask, barrier_geom
    except Exception:
        return mask, None


def _split_and_adjust_segments(segments_gdf: gpd.GeoDataFrame, connectors_gdf: gpd.GeoDataFrame, threshold: float) -> gpd.GeoDataFrame:
    """Split segments by connectors and adjust endpoints in one integrated function."""
    if connectors_gdf is None or connectors_gdf.empty:
        return segments_gdf

    valid_connector_ids = set(connectors_gdf["id"])

    # Process segments and split by connectors
    all_split_rows = []
    for _, row in segments_gdf.iterrows():
        # Parse connector information
        connectors = _parse_json_safely(row.get("connectors", ""), [])
        if not connectors:
            row["split_from"] = 0.0
            row["split_to"] = 1.0
            all_split_rows.append(row)
            continue

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
            all_split_rows.append(row)
            continue

        # Create split segments
        valid_positions = sorted(set(valid_positions))
        original_id = row.get("id", row.name)
        barrier_mask = row.get("barrier_mask", [[0.0, 1.0]])

        # Split at each connector position
        start_pct = 0.0
        for i, end_pct in enumerate(valid_positions):
            if end_pct > start_pct:
                part = _get_line_substring(row.geometry, start_pct, end_pct)
                if part and not part.is_empty:
                    new_row = row.copy()
                    new_row.geometry = part
                    new_row["split_from"] = start_pct
                    new_row["split_to"] = end_pct
                    # Recalculate barrier mask for subsegment
                    if barrier_mask != [[0.0, 1.0]] and barrier_mask:
                        seg_length = end_pct - start_pct
                        new_mask = []
                        for interval in barrier_mask:
                            inter_start = max(interval[0], start_pct)
                            inter_end = min(interval[1], end_pct)
                            if inter_start < inter_end:
                                new_mask.append([
                                    (inter_start - start_pct) / seg_length,
                                    (inter_end - start_pct) / seg_length,
                                ])
                        new_row["barrier_mask"] = new_mask
                    else:
                        new_row["barrier_mask"] = barrier_mask
                    new_row["id"] = f"{original_id}_{i+1}"
                    all_split_rows.append(new_row)
            start_pct = end_pct

        # Add final segment if needed
        if start_pct < 1.0:
            part = _get_line_substring(row.geometry, start_pct, 1.0)
            if part and not part.is_empty:
                new_row = row.copy()
                new_row.geometry = part
                new_row["split_from"] = start_pct
                new_row["split_to"] = 1.0
                # Recalculate barrier mask for final subsegment
                if barrier_mask != [[0.0, 1.0]] and barrier_mask:
                    seg_length = 1.0 - start_pct
                    new_mask = []
                    for interval in barrier_mask:
                        inter_start = max(interval[0], start_pct)
                        inter_end = min(interval[1], 1.0)
                        if inter_start < inter_end:
                            new_mask.append([
                                (inter_start - start_pct) / seg_length,
                                (inter_end - start_pct) / seg_length,
                            ])
                    new_row["barrier_mask"] = new_mask
                else:
                    new_row["barrier_mask"] = barrier_mask
                new_row["id"] = f"{original_id}_{len([r for r in all_split_rows if str(r.get('id', '')).startswith(str(original_id))])+1}"
                all_split_rows.append(new_row)

    # Create new GeoDataFrame with split segments
    result_gdf = gpd.GeoDataFrame(all_split_rows, crs=segments_gdf.crs).reset_index(drop=True)

    # Adjust endpoints by clustering nearby endpoints
    linestring_mask = result_gdf.geometry.type == "LineString"
    if not linestring_mask.any():
        return result_gdf

    # Extract and process endpoints
    endpoints_data = []
    for idx, geom in result_gdf.loc[linestring_mask, "geometry"].items():
        coords = list(geom.coords)
        if len(coords) >= 2:
            endpoints_data.extend([
                {"seg_id": idx, "pos": "start", "x": coords[0][0], "y": coords[0][1]},
                {"seg_id": idx, "pos": "end", "x": coords[-1][0], "y": coords[-1][1]},
            ])

    if endpoints_data:
        endpoints_df = pd.DataFrame(endpoints_data)

        # Quantize and compute centroids
        endpoints_df["bin_x"] = (endpoints_df["x"] / threshold).round().astype(int)
        endpoints_df["bin_y"] = (endpoints_df["y"] / threshold).round().astype(int)
        bin_centroids = endpoints_df.groupby(["bin_x", "bin_y"])[["x", "y"]].mean()
        endpoints_df = endpoints_df.merge(bin_centroids, on=["bin_x", "bin_y"], suffixes=("", "_centroid"))

        # Create lookup and rebuild geometries
        endpoint_lookup = {(row["seg_id"], row["pos"]): (row["x_centroid"], row["y_centroid"])
                          for _, row in endpoints_df.iterrows()}

        def rebuild_geometry(row):
            geom = row.geometry
            if not isinstance(geom, LineString):
                return geom
            coords = list(geom.coords)
            if len(coords) < 2:
                return geom
            start_coord = endpoint_lookup.get((row.name, "start"), coords[0])
            end_coord = endpoint_lookup.get((row.name, "end"), coords[-1])
            return LineString([start_coord] + coords[1:-1] + [end_coord])

        result_gdf.loc[linestring_mask, "geometry"] = result_gdf.loc[linestring_mask].apply(rebuild_geometry, axis=1)

    return result_gdf


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

    # Compute barrier masks and geometries if needed
    if get_barriers or (connectors_gdf is not None and not connectors_gdf.empty):
        barrier_data = segments_gdf["level_rules"].fillna("").astype(str).apply(
            lambda x: _compute_barrier_mask_and_geometry(x, None),
        )
        segments_gdf["barrier_mask"] = [data[0] for data in barrier_data]

    # Split segments by connectors if provided
    if connectors_gdf is not None and not connectors_gdf.empty:
        segments_gdf = _split_and_adjust_segments(segments_gdf, connectors_gdf, threshold)

    # Add length column
    segments_gdf["length"] = segments_gdf.geometry.length

    # Generate barrier geometries if requested
    if get_barriers:
        def get_barrier_geometry(row):
            if "barrier_mask" not in row:
                return None
            _, barrier_geom = _compute_barrier_mask_and_geometry(
                row.get("level_rules", ""), row.geometry,
            )
            return barrier_geom

        barrier_geoms = segments_gdf.apply(get_barrier_geometry, axis=1)
        segments_gdf["barrier_geometry"] = gpd.GeoSeries(barrier_geoms, crs=segments_gdf.crs)

    return segments_gdf
