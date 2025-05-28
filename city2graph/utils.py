"""
Module for utilities that handle GeoDataFrames and spatial operations.
"""

import json
import warnings
import os
import subprocess
from typing import List, Tuple, Union, Optional, Dict, Set, Any

import numpy as np
import pandas as pd  # needed for default series in split_segments_by_connectors
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString, Polygon, Point
from shapely.geometry.base import BaseGeometry
import networkx as nx
import momepy

# Define the public API for this module
__all__ = [
    "load_overture_data",
    "get_barrier_geometry",
    "identify_connector_mask",
    "identify_barrier_mask",
    "split_segments_by_connectors",
    "adjust_segment_connectors",
    "create_tessellation",
    "filter_network_by_distance",
]

# Valid Overture Maps data types
VALID_OVERTURE_TYPES: Set[str] = {
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


def load_overture_data(
    area: Union[List[float], Polygon],
    types: Optional[List[str]] = None,
    output_dir: str = ".",
    prefix: str = "",
    save_to_file: bool = True,
    return_data: bool = True,
) -> Dict[str, gpd.GeoDataFrame]:
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
    if types is None:
        types = list(VALID_OVERTURE_TYPES)
    else:
        # Validate input types
        invalid_types = [t for t in types if t not in VALID_OVERTURE_TYPES]
        if invalid_types:
            raise ValueError(
                f"Invalid Overture Maps data type(s): {invalid_types}. "
                f"Valid types are: {sorted(VALID_OVERTURE_TYPES)}"
            )

    # Create output directory if it doesn't exist
    if save_to_file and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract bounding box if area is a Polygon
    original_polygon = None

    if isinstance(area, Polygon):
        # Ensure the polygon is in WGS84 coordinates (EPSG:4326)
        wgs84_crs = "EPSG:4326"
        original_polygon = area

        # Check if we need to transform the polygon to WGS84
        if hasattr(area, "crs") and area.crs and area.crs != wgs84_crs:
            # Create a GeoDataFrame with the polygon to transform it
            temp_gdf = gpd.GeoDataFrame(geometry=[area], crs=area.crs)
            temp_gdf = temp_gdf.to_crs(wgs84_crs)
            original_polygon = temp_gdf.geometry.iloc[0]
            print(f"Transformed polygon from {area.crs} to WGS84 (EPSG:4326)")

        # Get the bounding box in WGS84 coordinates
        minx, miny, maxx, maxy = original_polygon.bounds
        bbox = [minx, miny, maxx, maxy]
    else:
        # Assume the provided bounding box is already in WGS84
        bbox = area

    # Format bbox as string for CLI
    bbox_str = ",".join(map(str, bbox))

    # Define WGS84 CRS constant
    WGS84_CRS = "EPSG:4326"

    result = {}

    for data_type in types:
        output_filename = (
            f"{prefix}{data_type}.geojson" if prefix else f"{data_type}.geojson"
        )
        output_path = os.path.join(output_dir, output_filename)

        # Build the CLI command
        cmd_parts = [
            "overturemaps",
            "download",
            f"--bbox={bbox_str}",
            "-f",
            "geojson",
            f"--type={data_type}",
        ]

        if save_to_file:
            cmd_parts.extend(["-o", output_path])

        try:
            # Execute the command
            process = subprocess.run(
                cmd_parts,
                check=True,
                stdout=subprocess.PIPE if not save_to_file else None,
                text=True,
            )

            # Read data if requested
            if return_data:
                try:
                    if save_to_file:
                        # Read from the saved file
                        if (
                            os.path.exists(output_path)
                            and os.path.getsize(output_path) > 0
                        ):
                            gdf = gpd.read_file(output_path)
                        else:
                            warnings.warn(f"No data returned for {data_type}")
                            # Create empty GeoDataFrame with proper geometry column and CRS
                            gdf = gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)
                    else:
                        # Parse from stdout
                        if process.stdout and process.stdout.strip():
                            try:
                                gdf = gpd.read_file(process.stdout)
                            except Exception as e:
                                warnings.warn(
                                    f"Could not parse GeoJSON for {data_type}: {e}"
                                )
                                gdf = gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)
                        else:
                            gdf = gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)

                    # Clip to original polygon if provided
                    if original_polygon is not None and not gdf.empty:
                        # Create a GeoDataFrame from the original polygon for clipping
                        mask = gpd.GeoDataFrame(
                            geometry=[original_polygon], crs=WGS84_CRS
                        )

                        # Ensure the mask CRS matches the data CRS
                        if gdf.crs != mask.crs:
                            mask = mask.to_crs(gdf.crs)

                        try:
                            gdf = gpd.clip(gdf, mask)
                        except Exception as e:
                            warnings.warn(f"Error clipping {data_type} to polygon: {e}")

                    # Ensure empty GeoDataFrames still have a valid geometry column
                    if gdf.empty and "geometry" not in gdf:
                        gdf = gpd.GeoDataFrame(geometry=[], crs=gdf.crs or WGS84_CRS)

                    result[data_type] = gdf

                except Exception as e:
                    warnings.warn(f"Error processing {data_type} data: {e}")
                    # Return an empty GeoDataFrame with proper setup
                    result[data_type] = gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)

                print(f"Successfully processed {data_type}")

        except subprocess.CalledProcessError as e:
            warnings.warn(f"Error downloading {data_type}: {e}")
            if return_data:
                # Create a properly formatted empty GeoDataFrame
                result[data_type] = gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)

    return result


def _extract_line_segment(
    line: LineString,
    start_point: Point,
    end_point: Point,
    start_dist: float,
    end_dist: float,
) -> Optional[LineString]:
    """
    Create a LineString segment between two points on a line.

    Parameters
    ----------
    line : LineString
        Original line
    start_point : Point
        Starting point on the line
    end_point : Point
        Ending point on the line
    start_dist : float
        Distance of start_point from the start of line
    end_dist : float
        Distance of end_point from the start of line

    Returns
    -------
    Optional[LineString]
        The extracted line segment
    """
    coords = list(line.coords)
    new_coords = []

    # Add the start point
    new_coords.append((start_point.x, start_point.y))

    # Find all intermediate vertices
    current_dist = 0
    for i in range(len(coords) - 1):
        p1, p2 = coords[i], coords[i + 1]
        seg = LineString([p1, p2])
        seg_length = seg.length
        next_dist = current_dist + seg_length

        # If this segment is after our start point and before our end point
        if next_dist > start_dist and current_dist < end_dist:
            # If this vertex is after start but before end, include it
            if current_dist >= start_dist:
                new_coords.append(p1)

            # If next vertex is after end, add the endpoint and break
            if next_dist >= end_dist:
                new_coords.append((end_point.x, end_point.y))
                break

        current_dist = next_dist

    # If we have at least two points, create a LineString
    if len(new_coords) >= 2:
        return LineString(new_coords)
    elif len(new_coords) == 1:
        # Edge case: create a very short line
        p = new_coords[0]
        return LineString([(p[0], p[1]), (p[0] + 1e-9, p[1] + 1e-9)])
    else:
        return None


def _get_substring(
    line: LineString, start_pct: float, end_pct: float
) -> Optional[LineString]:
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
    if not isinstance(line, LineString):
        return None

    if start_pct < 0 or end_pct > 1 or start_pct >= end_pct:
        return None

    # For full line or nearly full line, return the original
    if abs(start_pct) < 1e-9 and abs(end_pct - 1) < 1e-9:
        return line

    # Calculate distances along the line
    total_length = line.length
    start_dist = start_pct * total_length
    end_dist = end_pct * total_length

    if abs(end_dist - start_dist) < 1e-9:
        return None

    try:
        # Get points at the specified distances
        start_point = line.interpolate(start_dist)
        end_point = line.interpolate(end_dist)

        # Handle case where start and end are at endpoints
        if start_dist <= 1e-9 and end_dist >= total_length - 1e-9:
            return line

        return _extract_line_segment(line, start_point, end_point, start_dist, end_dist)

    except Exception as e:
        warnings.warn(f"Error creating line substring: {e}")
        return None


def identify_barrier_mask(level_rules: str) -> list:
    """
    Compute non-barrier intervals (barrier mask) from level_rules JSON.
    Only rules with "value" equal to 0 are considered as barriers.
    If any such rule has "between" equal to null, then the entire interval [0, 1]
    is treated as non-barrier.

    For example, if level_rules is:

    '[{"value": 0, "between": [0.17760709099999999, 0.83631280600000002]},
      {"value": 0, "between": [0.95722406000000004, 0.95967328100000004]}]'

    then the barrier intervals are [(0.17760709, 0.836312806), (0.95722406, 0.959673281)],
    and the returned non-barrier intervals will be:
    [[0.0, 0.17760709], [0.836312806, 0.95722406], [0.959673281, 1.0]].

    If any rule for which "value" equals 0 has "between" as null, then
    the function returns [[0.0, 1.0]].
    """
    if not isinstance(level_rules, str) or level_rules.strip().lower() in (
        "",
        "none",
        "null",
    ):
        return [[0.0, 1.0]]
    # Normalize Python None to JSON null for proper JSON parsing
    s = level_rules.replace("'", '"').replace('None', 'null')
    try:
        rules = json.loads(s)
    except Exception as e:
        warnings.warn(f"JSON parse failed for level_rules: {e}")
        return [[0.0, 1.0]]
    if not isinstance(rules, list):
        rules = [rules]
    barrier_intervals = []
    for rule in rules:
        if isinstance(rule, dict) and rule.get("value") is not None:
            if rule.get("value") != 0:
                between = rule.get("between")
                if between is None:
                    return []
                if isinstance(between, list) and len(between) == 2:
                    barrier_intervals.append((float(between[0]), float(between[1])))
    if not barrier_intervals:
        return [[0.0, 1.0]]
    barrier_intervals.sort(key=lambda x: x[0])
    result = []
    current = 0.0
    for start, end in barrier_intervals:
        if start > current:
            result.append([current, start])
        current = max(current, end)
    if current < 1.0:
        result.append([current, 1.0])
    return result


def _extract_barriers_from_mask(line: LineString, mask: list) -> Optional[BaseGeometry]:
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
    elif len(parts) == 1:
        return parts[0]
    else:
        return MultiLineString(parts)


def _get_barrier_geometry(row):
    if "barrier_mask" not in row:
        raise KeyError("Column 'barrier_mask' not found in input row")
    barrier_mask = row["barrier_mask"]

    if barrier_mask is None:
        return None

    if barrier_mask == [[0.0, 1.0]]:
        return row.geometry

    try:
        geom = row.geometry
        if isinstance(geom, MultiLineString):
            parts = []
            for part in geom.geoms:
                clipped = _extract_barriers_from_mask(part, barrier_mask)
                if clipped:
                    parts.extend(
                        clipped.geoms
                        if isinstance(clipped, MultiLineString)
                        else [clipped]
                    )
            return (
                None
                if not parts
                else parts[0] if len(parts) == 1 else MultiLineString(parts)
            )

        else:
            return _extract_barriers_from_mask(geom, barrier_mask)

    except Exception:
        return None


def get_barrier_geometry(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    # Process each row of the GeoDataFrame
    barrier_geoms = gdf.apply(_get_barrier_geometry, axis=1)
    return gpd.GeoSeries(barrier_geoms, crs=gdf.crs)


def identify_connector_mask(connectors_info: str) -> list:
    """
    Parse connectors_info and return a connector mask list.
    Returns a list of floats starting with 0.0 and ending with 1.0.
    If connectors_info is empty or invalid, returns [0.0, 1.0].
    """
    if not connectors_info or not str(connectors_info).strip():
        return [0.0, 1.0]
    try:
        parsed = json.loads(connectors_info.replace("'", '"'))
        if isinstance(parsed, dict):
            connectors_list = [parsed]
        elif isinstance(parsed, list):
            connectors_list = parsed
        else:
            return [0.0, 1.0]
        valid_ps = []
        for item in connectors_list:
            if isinstance(item, dict):
                at_val = item.get("at")
                if at_val is not None:
                    valid_ps.append(float(at_val))
        valid_ps.sort()
        return [0.0] + valid_ps + [1.0]
    except Exception:
        return [0.0, 1.0]


def _recalc_barrier_mask(original_mask: list, sub_start: float, sub_end: float) -> list:
    """
    Recalculate barrier_mask for a subsegment defined by [sub_start, sub_end].
    """
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
                ]
            )
    return new_mask


def _process_segment(row, valid_ids):
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
    list
        List of new rows created from splitting the segment
    """
    geom = row.geometry
    connectors_info = row.get("connectors")
    # Skip rows without connectors data
    if not connectors_info or not str(connectors_info).strip():
        return [row]

    # Parse JSON only once per row
    try:
        parsed = json.loads(str(connectors_info).replace("'", '"'))
        if isinstance(parsed, dict):
            connectors_list = [parsed]
        elif isinstance(parsed, list):
            connectors_list = parsed
        else:
            return [row]
    except Exception:
        return [row]

    # Find valid connectors
    valid_connectors = set()
    for item in connectors_list:
        if isinstance(item, dict):
            connector_id = item.get("connector_id")
            at_value = item.get("at")
            if connector_id is None or at_value is None:
                continue
            # Use pre-indexed valid_ids for fast lookup
            if connector_id not in valid_ids:
                continue
            valid_connectors.add(float(at_value))

    valid_connectors = sorted(valid_connectors)
    if not valid_connectors:
        return [row]

    # Recompute connector_mask without duplicates
    mask = []
    if not valid_connectors or valid_connectors[0] != 0.0:
        mask.append(0.0)
    mask.extend(valid_connectors)
    if not mask or mask[-1] != 1.0:
        mask.append(1.0)

    # Generate split geometries
    split_rows = []
    start_pct = 0.0
    counter = 1
    original_id = row.get("id", row.name)
    barrier_mask = row["barrier_mask"]

    for at in valid_connectors:
        part = _get_substring(geom, start_pct, at)
        if part is not None and not part.is_empty:
            # Create only the necessary data for each new row
            new_row = row.copy()
            new_row.geometry = part
            new_row["split_from"] = start_pct
            new_row["split_to"] = at
            new_row["connector_mask"] = mask
            new_row["barrier_mask"] = _recalc_barrier_mask(barrier_mask, start_pct, at)
            new_row["id"] = f"{original_id}_{counter}"
            counter += 1
            split_rows.append(new_row)
        start_pct = at

    # Process the last segment
    part = _get_substring(geom, start_pct, 1.0)
    if part is not None and not part.is_empty:
        new_row = row.copy()
        new_row.geometry = part
        new_row["split_from"] = start_pct
        new_row["split_to"] = 1.0
        new_row["connector_mask"] = mask
        new_row["barrier_mask"] = _recalc_barrier_mask(barrier_mask, start_pct, 1.0)
        new_row["id"] = f"{original_id}_{counter}"
        split_rows.append(new_row)

    return split_rows


def split_segments_by_connectors(
    segments_gdf: gpd.GeoDataFrame, connectors_gdf: gpd.GeoDataFrame
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
    if "connectors" in segments_gdf.columns:
        conn_series = segments_gdf["connectors"].astype(str)
    else:
        conn_series = pd.Series([""] * len(segments_gdf), index=segments_gdf.index)
    segments_gdf["connector_mask"] = conn_series.apply(identify_connector_mask)

    if "level_rules" in segments_gdf.columns:
        lvl_series = segments_gdf["level_rules"].astype(str)
    else:
        lvl_series = pd.Series([""] * len(segments_gdf), index=segments_gdf.index)
    segments_gdf["barrier_mask"] = lvl_series.apply(identify_barrier_mask)

    # Prepare data structures
    new_rows_data = []

    # Process segments in batches to reduce memory pressure
    batch_size = 1000
    for i in range(0, len(segments_gdf), batch_size):
        batch = segments_gdf.iloc[i : i + batch_size]
        batch_results = batch.apply(
            lambda row: _process_segment(row, valid_ids), axis=1
        )
        for rows in batch_results:
            new_rows_data.extend(rows)

    # Create a new GeoDataFrame from all processed rows, include split columns
    result_gdf = gpd.GeoDataFrame(new_rows_data, crs=segments_gdf.crs)

    # Reset the index of the resulting GeoDataFrame
    return result_gdf.reset_index(drop=True)


def _rebuild_geometry(seg_id: Any, geom: LineString, pivot_df: pd.DataFrame) -> List[Tuple[float, float]]:
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
    start = (
        pivot_df.loc[seg_id, ("x_centroid", "start")],
        pivot_df.loc[seg_id, ("y_centroid", "start")],
    )
    end = (
        pivot_df.loc[seg_id, ("x_centroid", "end")],
        pivot_df.loc[seg_id, ("y_centroid", "end")],
    )
    coords = list(geom.coords)
    if len(coords) > 2:
        new_coords = [start] + coords[1:-1] + [end]
    else:
        new_coords = [start, end]
    return new_coords


def adjust_segment_connectors(
    segments_gdf: gpd.GeoDataFrame, 
    threshold: float
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
    
    valid = segments_gdf.loc[mask].copy()
    valid["seg_id"] = valid.index

    # Extract start and end points from all LineStrings
    starts = [(geom.coords[0][0], geom.coords[0][1]) for geom in valid.geometry]
    ends = [(geom.coords[-1][0], geom.coords[-1][1]) for geom in valid.geometry]

    # Create DataFrame with all endpoints for easier processing
    endpoints_df = pd.DataFrame(
        {
            "seg_id": list(valid["seg_id"]) * 2,
            "pos": ["start"] * len(valid) + ["end"] * len(valid),
            "x": [pt[0] for pt in starts] + [pt[0] for pt in ends],
            "y": [pt[1] for pt in starts] + [pt[1] for pt in ends],
        }
    )

    # Quantize coordinates to bins based on threshold
    endpoints_df["bin_x"] = np.rint(endpoints_df["x"] / threshold).astype(int)
    endpoints_df["bin_y"] = np.rint(endpoints_df["y"] / threshold).astype(int)
    endpoints_df["bin"] = list(zip(endpoints_df["bin_x"], endpoints_df["bin_y"]))

    # Calculate centroids for each bin
    centroids = (
        endpoints_df.groupby("bin")[["x", "y"]]
        .mean()
        .rename(columns={"x": "x_centroid", "y": "y_centroid"})
    )
    endpoints_df = endpoints_df.join(centroids, on="bin")

    # Pivot the dataframe to get centroid coordinates by segment and position
    pivot_df = endpoints_df.pivot(
        index="seg_id", columns="pos", values=["x_centroid", "y_centroid"]
    )

    # Rebuild geometries using the centroid coordinates
    valid["geometry"] = valid.apply(
        lambda row: LineString(
            _rebuild_geometry(row["seg_id"], row.geometry, pivot_df)
        ),
        axis=1,
    )
    
    # Update the original GeoDataFrame with the new geometries
    segments_gdf.update(valid)
    return segments_gdf


def create_tessellation(
    geometry: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    primary_barriers: Optional[Union[gpd.GeoDataFrame, gpd.GeoSeries]] = None,
    shrink: float = 0.4,
    segment: float = 0.5,
    threshold: float = 0.05,
    n_jobs: int = -1,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """
    Create tessellations from the given geometries.
    If primary_barriers are provided, enclosed tessellations are created.
    If not, morphological tessellations are created.
    For more details, see momepy.enclosed_tessellation and momepy.morphological_tessellation.

    Parameters
    ----------
    geometry : Union[geopandas.GeoDataFrame, geopandas.GeoSeriesgeopandas       Input gegeopandastries to create a tessellation for. Should contain the geometries to tessellate.
    primary_barriers : Optional[Union[gpd.GeoDataFrame, gpd.GeoSeries]], default=None
        Optional GeoDataFrame or GeoSeries containing barriers to use for enclosed tessellation.
        If provided, the function will create enclosed tessellation using these barriers.
        If None, morphological tessellation will be created using the input geometries.
    shrink : float, default=0.4
        Distance for negative buffer of tessellations. By default, 0.4.
        Used for both momepy.enclosed_tessellation and momepy.morphological_tessellation.
    segment : float, default=0.5
        Maximum distance between points for tessellations. By default, 0.5.
        Used for both momepy.enclosed_tessellation and momepy.morphological_tessellation.
    threshold : float, default=0.05
        Minimum threshold for a building to be considered within an enclosure. By default, 0.05.
        Used for both momepy.enclosed_tessellation.
    n_jobs : int, default=-1
        Number of parallel jobs to run. By default, -1 (use all available cores).
        Used for both momepy.enclosed_tessellation.
    **kwargs : Any
        Additional keyword arguments to pass to momepy.enclosed_tessellation.
        These can include parameters specific to the tessellation method used.
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame containing the tessellation polygons.
    """
    # Create tessellation using momepy based on whether primary_barriers are provided
    if primary_barriers is not None:
        # Convert primary_barriers to GeoDataFrame if it's a GeoSeries
        if isinstance(primary_barriers, gpd.GeoSeries):
            primary_barriers = gpd.GeoDataFrame(
                geometry=primary_barriers, crs=primary_barriers.crs
            )

        # Ensure the barriers are in the same CRS as the input geometry
        if geometry.crs != primary_barriers.crs:
            raise ValueError(
                "CRS mismatch: geometry and barriers must have the same CRS."
            )

        # Create enclosures for enclosed tessellation
        enclosures = momepy.enclosures(
            primary_barriers=primary_barriers,
            limit=None,
            additional_barriers=None,
            enclosure_id="eID",
            clip=False,
        )

        tessellation = momepy.enclosed_tessellation(
            geometry=geometry,
            enclosures=enclosures,
            shrink=shrink,
            segment=segment,
            threshold=threshold,
            n_jobs=n_jobs,
            **kwargs,
        )

        # Apply ID handling for enclosed tessellation
        tessellation["tess_id"] = [
            f"{i}_{j}"
            for i, j in zip(tessellation["enclosure_index"], tessellation.index)
        ]
        tessellation.reset_index(drop=True, inplace=True)
    else:
        # Create morphological tessellation
        tessellation = momepy.morphological_tessellation(
            geometry=geometry, clip="bounding_box", shrink=shrink, segment=segment
        )
        tessellation["tess_id"] = tessellation.index

    return tessellation


def _get_nearest_node(point, nodes_gdf, node_id="node_id"):
    """
    Helper function to find the nearest node in a GeoDataFrame.
    """
    if isinstance(point, gpd.GeoSeries):
        point = point.iloc[0]
    nearest_idx = nodes_gdf.distance(point).idxmin()
    return nodes_gdf.loc[nearest_idx, node_id]


def filter_network_by_distance(
    network: Union[gpd.GeoDataFrame, nx.Graph],
    center_point: Union[Point, gpd.GeoSeries, gpd.GeoDataFrame],
    distance: float,
    edge_attr: str = "length",
    node_id_col: Optional[str] = None,
) -> Union[gpd.GeoDataFrame, nx.Graph]:
    """
    Extracts a filtered network containing only elements
    within a given shortest-path distance from specified center point(s).

    Parameters
    ----------
    network : Union[gpd.GeoDataFrame, nx.Graph]
        Input network data as either a GeoDataFrame of edges or a NetworkX graph.
    center_point : Union[Point, gpd.GeoSeries, gpd.GeoDataFrame]
        Center point(s) for distance calculation.
        Can be a single Point, GeoSeries of points, or GeoDataFrame with point geometries.
    distance : float, default=1000
        Maximum shortest-path distance from any center node.
    edge_attr : str, default="length"
        Edge attribute to use as weight for distance calculation.
    node_id_col : Optional[str], default=None
        Column name in nodes GeoDataFrame to use as node identifier.
        If None, will use auto-generated node IDs.

    Returns
    -------
    Union[gpd.GeoDataFrame, nx.Graph]
        Filtered network containing only elements within distance of any center point.
        Returns the same type as the input (either GeoDataFrame or NetworkX graph).
    """
    # Determine input type
    is_graph_input = isinstance(network, nx.Graph)

    # Convert to NetworkX graph if input is GeoDataFrame
    if is_graph_input:
        G = network
        original_crs = None
    else:
        G = momepy.gdf_to_nx(network)
        original_crs = network.crs

    # Build a GeoDataFrame for the nodes using their 'x' and 'y' attributes
    node_ids, node_geometries = zip(
        *[
            (nid, Point([attrs.get("x"), attrs.get("y")]))
            for nid, attrs in G.nodes(data=True)
        ]
    )

    # Use provided node_id_col or default
    node_id_name = node_id_col or "node_id"

    nodes_gdf = gpd.GeoDataFrame(
        {node_id_name: node_ids, "geometry": node_geometries}, crs=original_crs
    )

    # Initialize a set to collect nodes within distance from any center point
    nodes_within_distance = set()

    # Handle different types of center_point input
    center_points = center_point
    if isinstance(center_point, gpd.GeoSeries) or isinstance(
        center_point, gpd.GeoDataFrame
    ):
        # If it's a GeoDataFrame, convert to GeoSeries
        if isinstance(center_point, gpd.GeoDataFrame):
            center_points = center_point.geometry
    else:
        # Convert single point to a list
        center_points = [center_point]

    # Process each center point
    for point in center_points:
        # Find the nearest node to this center
        nearest_node = _get_nearest_node(point, nodes_gdf, node_id=node_id_name)

        # Compute shortest path lengths from this center
        try:
            distance_dict = nx.shortest_path_length(G, nearest_node, weight=edge_attr)
            # Add nodes within distance from this center
            nodes_within_distance.update(
                k for k, v in distance_dict.items() if v < distance
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            warnings.warn(
                f"Could not compute paths from a center point: {e}", RuntimeWarning
            )

    # Extract subgraph for nodes within distance from any center
    if nodes_within_distance:
        # Create a subgraph from the original graph
        subgraph = G.subgraph(nodes_within_distance)

        # Return the result in the same format as the input
        if is_graph_input:
            return subgraph
        else:
            filtered_gdf = momepy.nx_to_gdf(subgraph, points=False)

            # Ensure that the geometry column is properly set as GeoSeries
            if not isinstance(filtered_gdf.geometry, gpd.GeoSeries):
                filtered_gdf = gpd.GeoDataFrame(
                    filtered_gdf, geometry="geometry", crs=original_crs
                )

            return filtered_gdf
    else:
        # Return empty result in the same format as the input
        if is_graph_input:
            return nx.Graph()
        else:
            return gpd.GeoDataFrame(geometry=[], crs=original_crs)
