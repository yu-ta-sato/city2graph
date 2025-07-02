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
    # Validate types and set defaults
    types = types or list(VALID_OVERTURE_TYPES)
    invalid_types = [t for t in types if t not in VALID_OVERTURE_TYPES]
    assert not invalid_types, f"Invalid types: {invalid_types}"

    # Convert area to bbox string and prepare clipping geometry
    bbox_str = (",".join(str(round(c, 10)) for c in area.to_crs(WGS84_CRS).bounds)
               if isinstance(area, Polygon) else ",".join(str(float(b)) for b in area))
    clip_geom = area.to_crs(WGS84_CRS) if isinstance(area, Polygon) and hasattr(area, "crs") and area.crs != WGS84_CRS else area if isinstance(area, Polygon) else None

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True) if save_to_file else None

    # Process each data type
    result = {}
    for data_type in types:
        output_path = Path(output_dir) / f"{prefix}{data_type}.geojson"
        cmd = ["overturemaps", "download", f"--bbox={bbox_str}", "-f", "geojson", f"--type={data_type}"]
        save_to_file and cmd.extend(["-o", str(output_path)])

        subprocess.run(cmd, check=True, capture_output=not save_to_file, text=True)

        # Load and clip data if requested
        gdf = gpd.read_file(output_path) if return_data and save_to_file and output_path.exists() else gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)
        gdf = gpd.clip(gdf, gpd.GeoDataFrame(geometry=[clip_geom], crs=WGS84_CRS).to_crs(gdf.crs)) if return_data and clip_geom is not None and not gdf.empty else gdf
        return_data and result.update({data_type: gdf})

    return result


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
    # Early return for empty input
    result_gdf = segments_gdf.copy() if not segments_gdf.empty else segments_gdf
    result_gdf["level_rules"] = result_gdf.get("level_rules", "").fillna("") if not segments_gdf.empty else None

    # Connector processing
    valid_connector_ids = set(connectors_gdf["id"]) if connectors_gdf is not None and not connectors_gdf.empty else set()
    split_segments = []

    # Process each segment for splitting
    for _, segment in result_gdf.iterrows():
        connectors_str = segment.get("connectors", "")
        connectors_data = json.loads(connectors_str.replace("'", '"').replace("None", "null")) if connectors_str else []
        connectors_data = connectors_data if isinstance(connectors_data, list) else [connectors_data] if connectors_data else []

        positions = [float(conn["at"]) for conn in connectors_data
                    if isinstance(conn, dict) and conn.get("connector_id") in valid_connector_ids and "at" in conn]
        positions = sorted({0.0, *positions, 1.0}) if positions else [0.0, 1.0]
        original_id = segment.get("id", segment.name)

        # Create segments from positions
        for i in range(len(positions) - 1):
            start_pct, end_pct = positions[i], positions[i + 1]
            part_geom = (segment.geometry if abs(start_pct) < 1e-9 and abs(end_pct - 1.0) < 1e-9
                        else substring(segment.geometry, start_pct, end_pct, normalized=True)) if end_pct > start_pct else None

            if part_geom and not part_geom.is_empty:
                new_segment = segment.copy()
                new_segment.geometry = part_geom
                new_segment["split_from"] = start_pct
                new_segment["split_to"] = end_pct
                new_segment["id"] = f"{original_id}_{i+1}" if len(positions) > 2 else original_id
                split_segments.append(new_segment)

    # Update result with split segments
    result_gdf = gpd.GeoDataFrame(split_segments, crs=result_gdf.crs).reset_index(drop=True) if split_segments else result_gdf

    # Endpoint adjustment
    endpoints_data = []
    for idx, geom in result_gdf.geometry.items():
        if isinstance(geom, LineString) and len(geom.coords) >= 2:
            coords = list(geom.coords)
            endpoints_data.extend([
                {"seg_id": idx, "pos": "start", "x": coords[0][0], "y": coords[0][1]},
                {"seg_id": idx, "pos": "end", "x": coords[-1][0], "y": coords[-1][1]},
            ])

    # Apply endpoint clustering
    if endpoints_data and valid_connector_ids:
        endpoints_df = pd.DataFrame(endpoints_data)
        endpoints_df["bin_x"] = (endpoints_df["x"] / threshold).round().astype(int)
        endpoints_df["bin_y"] = (endpoints_df["y"] / threshold).round().astype(int)
        centroids = endpoints_df.groupby(["bin_x", "bin_y"])[["x", "y"]].mean()
        endpoints_df = endpoints_df.merge(centroids, on=["bin_x", "bin_y"], suffixes=("", "_new"))
        coord_lookup = {(row["seg_id"], row["pos"]): (row["x_new"], row["y_new"]) for _, row in endpoints_df.iterrows()}

        # Update geometries with adjusted coordinates
        for idx, row in result_gdf.iterrows():
            geom = row.geometry
            if isinstance(geom, LineString) and len(geom.coords) >= 2:
                coords = list(geom.coords)
                start_coord = coord_lookup.get((idx, "start"), coords[0])
                end_coord = coord_lookup.get((idx, "end"), coords[-1])
                result_gdf.at[idx, "geometry"] = LineString([start_coord] + coords[1:-1] + [end_coord])

    # Add length column
    result_gdf["length"] = result_gdf.geometry.length

    # Barrier geometry generation
    barrier_geometries = []
    for _, row in result_gdf.iterrows():
        level_rules_str = row.get("level_rules", "")
        geometry = row.geometry

        # Parse level rules
        rules_data = json.loads(level_rules_str.replace("'", '"').replace("None", "null")) if level_rules_str else []
        rules_data = rules_data if isinstance(rules_data, list) else [rules_data] if rules_data else []

        # Extract barrier intervals
        barrier_intervals = []
        for rule in rules_data:
            if isinstance(rule, dict) and rule.get("value") != 0:
                between = rule.get("between")
                if between is None:
                    barrier_intervals = "full_barrier"
                    break
                if isinstance(between, list) and len(between) == 2:
                    barrier_intervals.append((float(between[0]), float(between[1])))

        # Compute passable mask
        passable_mask = ([] if barrier_intervals == "full_barrier" else
                        [(0.0, 1.0)] if not barrier_intervals else
                        [(current, start) for current in [0.0] for start, end in sorted(barrier_intervals) if start > current] +
                        [(max([end for _, end in sorted(barrier_intervals)], default=0.0), 1.0)]
                        if max([end for _, end in sorted(barrier_intervals)], default=0.0) < 1.0 else [])

        # Create barrier geometry
        barrier_geom = None
        if passable_mask and geometry and not geometry.is_empty:
            if passable_mask == [(0.0, 1.0)]:
                barrier_geom = geometry
            else:
                parts = []
                for start_pct, end_pct in passable_mask:
                    part = (geometry if abs(start_pct) < 1e-9 and abs(end_pct - 1.0) < 1e-9
                           else substring(geometry, start_pct, end_pct, normalized=True))
                    if part and not part.is_empty:
                        parts.append(part)
                barrier_geom = parts[0] if len(parts) == 1 else MultiLineString(parts) if parts else None

        barrier_geometries.append(barrier_geom)

    # Add barrier geometry column
    if get_barriers:
        result_gdf["barrier_geometry"] = gpd.GeoSeries(barrier_geometries, crs=result_gdf.crs)

    return result_gdf
