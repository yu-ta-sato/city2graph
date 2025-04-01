"""
Geometry utilities for handling GeoDataFrames and spatial operations.
"""

import json
import warnings
import os
import subprocess
from typing import List, Tuple, Union, Optional, Dict, Set, Any, Sequence, cast

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Polygon, Point
from shapely.geometry.base import BaseGeometry
import networkx as nx
from shapely.ops import unary_union
import momepy

# Define the public API for this module
__all__ = [
    "load_overture_data",
    "get_barrier_geometry",
    "identify_connector_mask",
    "identify_barrier_mask",
    "split_segments_by_connectors",
    "create_tessellation",
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


# New helper: safely parse JSON strings
def _parse_json_safe(s: str) -> Any:
    try:
        return json.loads(s.replace("'", '"'))
    except Exception as e:
        warnings.warn(f"JSON parse failed: {e}")
        return None


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


# Replace the existing identify_barrier_mask function with the following,
# which returns the full interval [[0,1]] if any barrier rule's "between" is null.
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
    try:
        rules = json.loads(level_rules.replace("'", '"'))
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


# New helper: Recalculate barrier_mask for a subsegment.
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


def split_segments_by_connectors(
    segments_gdf: gpd.GeoDataFrame, connectors_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    new_rows = []
    for idx, row in segments_gdf.iterrows():
        geom = row.geometry
        connectors_info = row.get("connectors")
        level_rules = row.get("level_rules")  # use level_rules
        connector_mask = identify_connector_mask(str(connectors_info))
        barrier_mask = identify_barrier_mask(str(level_rules))
        # If no connectors data, keep original row.
        if not connectors_info or not str(connectors_info).strip():
            new_row = row.copy()
            new_row["connector_mask"] = connector_mask
            new_row["barrier_mask"] = barrier_mask
            new_rows.append(new_row)
            continue
        try:
            parsed = json.loads(str(connectors_info).replace("'", '"'))
            if isinstance(parsed, dict):
                connectors_list = [parsed]
            elif isinstance(parsed, list):
                connectors_list = parsed
            else:
                new_row = row.copy()
                new_row["connector_mask"] = connector_mask
                new_row["barrier_mask"] = barrier_mask
                new_rows.append(new_row)
                continue
        except Exception as e:
            warnings.warn(
                f"Failed to parse connectors for segment {row.get('id', idx)}: {e}"
            )
            new_row = row.copy()
            new_row["connector_mask"] = connector_mask
            new_row["barrier_mask"] = barrier_mask
            new_rows.append(new_row)
            continue

        valid_connectors = set()
        for item in connectors_list:
            if isinstance(item, dict):
                connector_id = item.get("connector_id")
                at_value = item.get("at")
                if connector_id is None or at_value is None:
                    continue
                if connectors_gdf[connectors_gdf["id"] == connector_id].empty:
                    continue
                valid_connectors.add(float(at_value))
        valid_connectors = sorted(valid_connectors)
        # Recompute connector_mask without duplicates:
        mask = []
        if not valid_connectors or valid_connectors[0] != 0.0:
            mask.append(0.0)
        mask.extend(valid_connectors)
        if not mask or mask[-1] != 1.0:
            mask.append(1.0)
        connector_mask = mask

        split_geoms = []
        start_pct = 0.0
        for at in valid_connectors:
            part = _get_substring(geom, start_pct, at)
            if part is not None and not part.is_empty:
                split_geoms.append((part, start_pct, at))
            start_pct = at
        part = _get_substring(geom, start_pct, 1.0)
        if part is not None and not part.is_empty:
            split_geoms.append((part, start_pct, 1.0))
        counter = 1  # counter to number each split segment
        for part, s, e in split_geoms:
            new_row = row.copy()
            new_row.geometry = part
            new_row["split_from"] = s
            new_row["split_to"] = e
            new_row["connector_mask"] = connector_mask
            new_row["barrier_mask"] = _recalc_barrier_mask(barrier_mask, s, e)
            original_id = row.get("id", idx)
            new_row["id"] = f"{original_id}_{counter}"
            counter += 1
            new_rows.append(new_row)

    new_columns = (
        segments_gdf.columns.tolist()
    )  # + ["split_from", "split_to", "connector_mask", "barrier_mask"]
    result_gdf = gpd.GeoDataFrame(new_rows, columns=new_columns, crs=segments_gdf.crs)

    # Reset the index of the resulting GeoDataFrame
    result_gdf = result_gdf.reset_index(drop=True)

    return result_gdf


def get_walking_distance(
    point: Point, distance: float, street_network: nx.Graph
) -> BaseGeometry:
    """
    Compute an area reachable on foot from a point, based on topological distance of the street network.

    Parameters
    ----------
    point : Point
        The location from which to compute walking distance.
    distance : float
        The maximum travel distance along the street network.
    street_network : nx.Graph
        A networkx Graph where nodes have 'geometry' (Point) and edges have 'geometry' (LineString)
        and a 'length' attribute for distance.

    Returns
    -------
    BaseGeometry
        A unified geometry of all edges reachable within the specified distance.
    """
    # Find the nearest node in the street network
    nearest_node = min(
        street_network.nodes,
        key=lambda n: point.distance(street_network.nodes[n]["geometry"]),
    )

    # Get all nodes within the given distance
    lengths = nx.single_source_dijkstra_path_length(
        street_network, nearest_node, cutoff=distance, weight="length"
    )
    reachable_nodes = set(lengths.keys())

    # Collect edges between reachable nodes
    edge_geometries = []
    for u, v in street_network.edges():
        if u in reachable_nodes and v in reachable_nodes:
            edge_data = street_network[u][v]
            if "geometry" in edge_data:
                edge_geometries.append(edge_data["geometry"])

    # Merge edges into a single geometry
    return unary_union(edge_geometries)


def create_tessellation(
    geometry: Union[gpd.GeoDataFrame, gpd.GeoSeries],
    primary_barriers: Optional[Union[gpd.GeoDataFrame, gpd.GeoSeries]] = None,
    shrink: float=0.4,
    segment: float=0.5,
    threshold: float=0.05,
    n_jobs: int=-1,
    **kwargs: Any
) -> gpd.GeoDataFrame:
    """
,
    shrink: float=0.4,
    segment: float=0.5,
    threshold: float=0.05,
    n_jobs: int=-1,
    **kwargs: Any
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
        GeoDataFrageopandascontaining the tessellation polygeopandass.
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
        enclosures = momepy.enclosures(primary_barriers=primary_barriers,
                                       limit=None,
                                       additional_barriers=None,
                                       enclosure_id='eID',
                                       clip=False)
        
        tessellation = momepy.enclosed_tessellation(
            geometry=geometry,
            enclosures=enclosures,
            shrink=shrink,
            segment=segment,
            threshold=threshold,
            n_jobs=n_jobs,
            **kwargs
            )

        # Apply ID handling for enclosed tessellation
        tessellation["tess_id"] = [
            f"{i}_{j}"
            for i, j in zip(tessellation["enclosure_index"], tessellation.index)
        ]
        tessellation.reset_index(drop=True, inplace=True)
    else:
        # Create morphological tessellation
        tessellation = momepy.morphological_tessellation(geometry=geometry,
                                                         clip='bounding_box',
                                                         shrink=shrink,
                                                         segment=segment)
        tessellation["tess_id"] = tessellation.index

    return tessellation
