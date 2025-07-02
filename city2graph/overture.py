"""Module for loading and processing geospatial data from Overture Maps."""

import json
import logging
import subprocess
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Polygon
from shapely.ops import substring

__all__ = ["load_overture_data", "process_overture_segments"]

VALID_OVERTURE_TYPES = {
    "address", "bathymetry", "building", "building_part", "division",
    "division_area", "division_boundary", "place", "segment", "connector",
    "infrastructure", "land", "land_cover", "land_use", "water",
}

logger = logging.getLogger(__name__)
WGS84_CRS = "EPSG:4326"


def _download_overture_data(data_type, area, output_dir, prefix, save_to_file, return_data):
    """Download and process a single overture data type."""
    if data_type not in VALID_OVERTURE_TYPES:
        raise ValueError(f"Invalid data type: {data_type}")

    # Handle area input - convert to bbox
    if isinstance(area, Polygon):
        if hasattr(area, "crs") and area.crs and area.crs != WGS84_CRS:
            area = area.to_crs(WGS84_CRS)
            logger.info("Transformed polygon from %s to WGS84", area.crs)
        bbox = [round(c, 10) for c in area.bounds]
        clip_polygon = area
    else:
        bbox = area
        clip_polygon = None

    # Validate and format bbox
    try:
        bbox_str = ",".join(str(float(b)) for b in bbox)
        if len(bbox) != 4:
            raise ValueError("Bbox must have 4 coordinates")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid bbox format: {e}") from e

    # Setup output path
    output_dir = Path(output_dir).resolve()
    filename = f"{prefix}{data_type}.geojson" if prefix else f"{data_type}.geojson"
    output_path = output_dir / filename

    # Execute download command
    cmd = ["overturemaps", "download", f"--bbox={bbox_str}", "-f", "geojson", f"--type={data_type}"]
    if save_to_file:
        cmd.extend(["-o", str(output_path)])

    try:
        result = subprocess.run(cmd, check=True, capture_output=not save_to_file, text=True)
        
        if not return_data:
            return None

        # Load data from file or stdout
        gdf = gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)
        if save_to_file and output_path.exists() and output_path.stat().st_size > 0:
            gdf = gpd.read_file(output_path)
        elif result.stdout and result.stdout.strip():
            try:
                gdf = gpd.read_file(result.stdout)
            except Exception as e:
                logger.warning("Could not parse GeoJSON for %s: %s", data_type, e)

        # Clip to polygon if provided
        if clip_polygon is not None and not gdf.empty:
            try:
                mask = gpd.GeoDataFrame(geometry=[clip_polygon], crs=WGS84_CRS)
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
    # Validate and set default types
    if types is None:
        types = list(VALID_OVERTURE_TYPES)
    else:
        invalid_types = [t for t in types if t not in VALID_OVERTURE_TYPES]
        if invalid_types:
            raise ValueError(f"Invalid Overture Maps data type(s): {invalid_types}. "
                           f"Valid types are: {sorted(VALID_OVERTURE_TYPES)}")

    # Create output directory
    if save_to_file:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Download each data type
    result = {}
    for data_type in types:
        gdf = _download_overture_data(data_type, area, output_dir, prefix, save_to_file, return_data)
        if return_data and gdf is not None:
            result[data_type] = gdf

    return result


def _parse_level_rules(level_rules_str):
    """Parse level rules JSON string safely."""
    if not level_rules_str or not isinstance(level_rules_str, str):
        return []
    try:
        rules = json.loads(level_rules_str.replace("'", '"').replace("None", "null"))
        return rules if isinstance(rules, list) else [rules] if rules else []
    except (json.JSONDecodeError, TypeError):
        return []


def _extract_barrier_intervals(level_rules_str):
    """Extract barrier intervals from level rules where value != 0."""
    rules = _parse_level_rules(level_rules_str)
    barrier_intervals = []
    
    for rule in rules:
        if isinstance(rule, dict) and rule.get("value") != 0:
            between = rule.get("between")
            if between is None:
                return "full_barrier"  # Entire segment is a barrier
            if isinstance(between, list) and len(between) == 2:
                barrier_intervals.append((float(between[0]), float(between[1])))
    
    return sorted(barrier_intervals)


def _compute_passable_mask(barrier_intervals):
    """Compute passable (non-barrier) intervals from barrier intervals."""
    if barrier_intervals == "full_barrier":
        return []
    if not barrier_intervals:
        return [(0.0, 1.0)]
    
    # Compute complement of barrier intervals
    mask = []
    current = 0.0
    
    for start, end in barrier_intervals:
        if start > current:
            mask.append((current, start))
        current = max(current, end)
    
    if current < 1.0:
        mask.append((current, 1.0))
    
    return mask


def _create_barrier_geometry(geometry, passable_mask):
    """Create barrier geometry from passable mask."""
    if not passable_mask or not geometry or geometry.is_empty:
        return None
    
    if passable_mask == [(0.0, 1.0)]:
        return geometry
    
    def extract_line_parts(line):
        parts = []
        for start_pct, end_pct in passable_mask:
            try:
                if start_pct < 1e-9 and end_pct > 1 - 1e-9:
                    part = line
                else:
                    part = substring(line, start_pct, end_pct, normalized=True)
                if part and not part.is_empty:
                    parts.append(part)
            except Exception:
                continue
        return parts
    
    try:
        if isinstance(geometry, MultiLineString):
            all_parts = []
            for geom in geometry.geoms:
                all_parts.extend(extract_line_parts(geom))
        else:
            all_parts = extract_line_parts(geometry)
        
        if not all_parts:
            return None
        return all_parts[0] if len(all_parts) == 1 else MultiLineString(all_parts)
    except Exception:
        return None


def _split_segments_by_connectors(segments_gdf, connectors_gdf):
    """Split segments at connector positions."""
    if connectors_gdf is None or connectors_gdf.empty:
        return segments_gdf
    
    valid_connector_ids = set(connectors_gdf["id"])
    split_segments = []
    
    for _, segment in segments_gdf.iterrows():
        # Parse connectors from segment
        connectors_str = segment.get("connectors", "")
        connectors = _parse_level_rules(connectors_str) if connectors_str else []
        
        # Extract valid connector positions
        positions = []
        for connector in connectors:
            if (isinstance(connector, dict) and 
                connector.get("connector_id") in valid_connector_ids and
                "at" in connector):
                try:
                    positions.append(float(connector["at"]))
                except (ValueError, TypeError):
                    continue
        
        if not positions:
            # No valid connectors, keep original segment
            segment_copy = segment.copy()
            segment_copy["split_from"] = 0.0
            segment_copy["split_to"] = 1.0
            split_segments.append(segment_copy)
            continue
        
        # Split segment at connector positions
        positions = sorted(set([0.0] + positions + [1.0]))
        original_id = segment.get("id", segment.name)
        
        for i in range(len(positions) - 1):
            start_pct, end_pct = positions[i], positions[i + 1]
            if end_pct > start_pct:
                try:
                    if start_pct < 1e-9 and end_pct > 1 - 1e-9:
                        part_geom = segment.geometry
                    else:
                        part_geom = substring(segment.geometry, start_pct, end_pct, normalized=True)
                    
                    if part_geom and not part_geom.is_empty:
                        new_segment = segment.copy()
                        new_segment.geometry = part_geom
                        new_segment["split_from"] = start_pct
                        new_segment["split_to"] = end_pct
                        new_segment["id"] = f"{original_id}_{i+1}" if len(positions) > 2 else original_id
                        split_segments.append(new_segment)
                except Exception as e:
                    logger.warning("Error splitting segment: %s", e)
                    continue
    
    return gpd.GeoDataFrame(split_segments, crs=segments_gdf.crs).reset_index(drop=True)


def _adjust_segment_endpoints(segments_gdf, threshold):
    """Adjust segment endpoints by clustering nearby points."""
    if segments_gdf.empty:
        return segments_gdf
    
    # Extract all endpoints
    endpoints = []
    for idx, geom in segments_gdf.geometry.items():
        if isinstance(geom, LineString) and len(geom.coords) >= 2:
            coords = list(geom.coords)
            endpoints.extend([
                {"seg_id": idx, "pos": "start", "x": coords[0][0], "y": coords[0][1]},
                {"seg_id": idx, "pos": "end", "x": coords[-1][0], "y": coords[-1][1]}
            ])
    
    if not endpoints:
        return segments_gdf
    
    # Cluster endpoints and compute centroids
    endpoints_df = pd.DataFrame(endpoints)
    endpoints_df["bin_x"] = (endpoints_df["x"] / threshold).round().astype(int)
    endpoints_df["bin_y"] = (endpoints_df["y"] / threshold).round().astype(int)
    
    centroids = endpoints_df.groupby(["bin_x", "bin_y"])[["x", "y"]].mean()
    endpoints_df = endpoints_df.merge(centroids, on=["bin_x", "bin_y"], suffixes=("", "_new"))
    
    # Create lookup for new coordinates
    coord_lookup = {(row["seg_id"], row["pos"]): (row["x_new"], row["y_new"]) 
                   for _, row in endpoints_df.iterrows()}
    
    # Update geometries with adjusted endpoints
    def adjust_geometry(row):
        geom = row.geometry
        if not isinstance(geom, LineString) or len(geom.coords) < 2:
            return geom
        
        coords = list(geom.coords)
        start_coord = coord_lookup.get((row.name, "start"), coords[0])
        end_coord = coord_lookup.get((row.name, "end"), coords[-1])
        
        return LineString([start_coord] + coords[1:-1] + [end_coord])
    
    segments_gdf = segments_gdf.copy()
    segments_gdf.geometry = segments_gdf.apply(adjust_geometry, axis=1)
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
    if segments_gdf.empty:
        return segments_gdf
    
    # Ensure required columns exist
    if "level_rules" not in segments_gdf.columns:
        segments_gdf["level_rules"] = ""
    
    # Make a copy to avoid modifying the original
    result_gdf = segments_gdf.copy()
    
    # Split segments by connectors if provided
    if connectors_gdf is not None and not connectors_gdf.empty:
        result_gdf = _split_segments_by_connectors(result_gdf, connectors_gdf)
        result_gdf = _adjust_segment_endpoints(result_gdf, threshold)
    
    # Add length column
    result_gdf["length"] = result_gdf.geometry.length
    
    # Generate barrier geometries if requested
    if get_barriers:
        def compute_barrier_geometry(level_rules_str, geometry):
            barrier_intervals = _extract_barrier_intervals(level_rules_str)
            passable_mask = _compute_passable_mask(barrier_intervals)
            return _create_barrier_geometry(geometry, passable_mask)
        
        barrier_geometries = [
            compute_barrier_geometry(row.get("level_rules", ""), row.geometry)
            for _, row in result_gdf.iterrows()
        ]
        result_gdf["barrier_geometry"] = gpd.GeoSeries(barrier_geometries, crs=result_gdf.crs)
    
    return result_gdf
