import os
import io
import zipfile
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

__all__ = ["load_gtfs", "get_od_pairs", "create_travel_summary_network"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_gtfs_df(gtfs_zip_path):
    """
    Load GTFS data from a zip file into a dictionary of pandas DataFrames.
    """
    gtfs_data = {}
    try:
        with zipfile.ZipFile(gtfs_zip_path, "r") as zip_ref:
            file_list = zip_ref.namelist()
            for file_name in file_list:
                # Skip directories and non-txt files
                if file_name.endswith("/") or not file_name.endswith(".txt"):
                    continue
                with zip_ref.open(file_name) as file:
                    try:
                        base_name = os.path.basename(file_name).replace(".txt", "")
                        gtfs_data[base_name] = pd.read_csv(
                            io.BytesIO(file.read()),
                            encoding="utf-8-sig",
                            dtype=str,
                        )
                    except Exception as e:
                        logger.error(f"Error loading {file_name}: {e}")
    except Exception as e:
        logger.error(f"Failed to read GTFS zip file: {e}")
    return gtfs_data


def _process_gtfs_df(gtfs_data):
    """
    Process GTFS DataFrames to apply appropriate data types and formats.
    """
    processed_data = gtfs_data.copy()

    # Process stops.txt - convert latitude and longitude columns to numeric
    stops_df = processed_data.get("stops")
    if stops_df is not None and all(col in stops_df.columns for col in ["stop_lat", "stop_lon"]):
        stops_df["stop_lat"] = pd.to_numeric(stops_df["stop_lat"], errors="coerce")
        stops_df["stop_lon"] = pd.to_numeric(stops_df["stop_lon"], errors="coerce")
        processed_data["stops"] = stops_df.dropna(subset=["stop_lat", "stop_lon"])

    # Process routes.txt - convert route_type to numeric
    routes_df = processed_data.get("routes")
    if routes_df is not None and "route_type" in routes_df.columns:
        processed_data["routes"]["route_type"] = pd.to_numeric(routes_df["route_type"], errors="coerce")

    # Process calendar.txt - convert service day columns to boolean
    calendar_df = processed_data.get("calendar")
    if calendar_df is not None:
        for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
            if day in calendar_df.columns:
                processed_data["calendar"][day] = calendar_df[day].astype(int).astype(bool)

    # Process stop_times.txt - no changes needed since time processing can be complex
    if processed_data.get("stop_times") is not None:
        pass

    return processed_data


def _get_stops_geometry(stops_df):
    """
    Create a GeoSeries of Points for stops based on latitude and longitude.
    
    Parameters
    ----------
    stops_df : pandas.DataFrame
        DataFrame containing stop information with stop_lat and stop_lon columns
        
    Returns
    -------
    geopandas.GeoSeries
        GeoSeries containing Point geometries indexed by stop_id
    """
    if stops_df is None or not all(col in stops_df.columns for col in ["stop_lon", "stop_lat", "stop_id"]):
        logger.warning("Missing required columns in stops DataFrame")
        return None
    
    # Convert lat/lon to numeric if not already
    stops_df = stops_df.copy()
    stops_df["stop_lat"] = pd.to_numeric(stops_df["stop_lat"], errors="coerce")
    stops_df["stop_lon"] = pd.to_numeric(stops_df["stop_lon"], errors="coerce")
    stops_df = stops_df.dropna(subset=["stop_lat", "stop_lon"])
    
    # Create Point geometries
    geometries = [Point(lon, lat) for lon, lat in zip(stops_df["stop_lon"], stops_df["stop_lat"])]
    
    # Create and return a GeoSeries indexed by stop_id
    return gpd.GeoSeries(geometries, crs="EPSG:4326")


def _get_shapes_geometry(shapes_df):
    """
    Create a GeoSeries of LineStrings for shapes by aggregating points in sequence.
    
    Parameters
    ----------
    shapes_df : pandas.DataFrame
        DataFrame containing shape information with shape_id, shape_pt_lat, 
        shape_pt_lon, and shape_pt_sequence columns
        
    Returns
    -------
    geopandas.GeoSeries
        GeoSeries containing LineString geometries indexed by shape_id
    """
    if shapes_df is None or not all(col in shapes_df.columns for col in ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]):
        logger.warning("Missing required columns in shapes DataFrame")
        return None

    # Convert columns to numeric
    shapes_df = shapes_df.copy()
    shapes_df["shape_pt_lat"] = pd.to_numeric(shapes_df["shape_pt_lat"], errors="coerce")
    shapes_df["shape_pt_lon"] = pd.to_numeric(shapes_df["shape_pt_lon"], errors="coerce")
    shapes_df["shape_pt_sequence"] = pd.to_numeric(shapes_df["shape_pt_sequence"], errors="coerce")
    shapes_df = shapes_df.dropna(subset=["shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"])
    shapes_df = shapes_df.sort_values(["shape_id", "shape_pt_sequence"])

    # Create LineStrings for each shape_id
    linestrings = {}
    for shape_id, group in shapes_df.groupby("shape_id"):
        points = [Point(lon, lat) for lon, lat in zip(group["shape_pt_lon"], group["shape_pt_lat"])]
        if len(points) >= 2:
            linestrings[shape_id] = LineString(points)
    
    # Return a GeoSeries if we have any valid shapes
    if linestrings:
        return gpd.GeoSeries(linestrings, crs="EPSG:4326")
    return None


def _create_route_trips_df(gtfs_data, shapes_geometry):
    """
    Create a GeoDataFrame linking routes and trips with shape geometries.
    
    Parameters
    ----------
    gtfs_data : dict
        Dictionary with GTFS file names as keys and pandas DataFrames as values
    shapes_geometry : geopandas.GeoSeries
        GeoSeries containing LineString geometries indexed by shape_id
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with route and trip information and LineString geometries
    """
    if not all(key in gtfs_data for key in ["routes", "trips"]) or shapes_geometry is None:
        logger.warning("Missing required data to create route trips GeoDataFrame")
        return None

    try:
        trips_df = gtfs_data["trips"]
        routes_df = gtfs_data["routes"]

        # Create a DataFrame from the shapes GeoSeries
        shapes_df = pd.DataFrame({"shape_id": shapes_geometry.index, "geometry": shapes_geometry.values})
        
        # Merge trips with shapes geometry
        trips_with_shapes = trips_df.merge(
            shapes_df, on="shape_id", how="inner"
        )
        
        # Merge with routes
        route_trips = trips_with_shapes.merge(
            routes_df, on="route_id", how="inner"
        )
        
        # Convert to GeoDataFrame
        return gpd.GeoDataFrame(route_trips, geometry="geometry", crs=shapes_geometry.crs)
    except Exception as e:
        logger.error(f"Error creating route trips GeoDataFrame: {e}")
    return None


def load_gtfs(gtfs_zip_path):
    """
    Complete workflow to process a GTFS zip file into DataFrames with geometry columns and summary statistics.
    
    Parameters
    ----------
    gtfs_zip_path : str or Path
        Path to the GTFS zip file
        
    Returns
    -------
    dict
        Dictionary with processed GTFS data including geometry objects
    """
    logger.info(f"Loading GTFS data from {gtfs_zip_path}...")
    gtfs_data = _get_gtfs_df(gtfs_zip_path)
    logger.info(f"Loaded {len(gtfs_data)} GTFS files")

    logger.info("Processing GTFS DataFrames...")
    gtfs_data = _process_gtfs_df(gtfs_data)

    if gtfs_data["stops"] is not None:
        logger.info("Creating stops geometry...")
        stops_geometry = _get_stops_geometry(gtfs_data["stops"])

        if stops_geometry is not None:
            # Add the stops GeoSeries to the dictionary
            gtfs_data["stops"] = gpd.GeoDataFrame(
                gtfs_data["stops"], geometry=stops_geometry, crs="EPSG:4326")
            
    if gtfs_data["shapes"] is not None:
        logger.info("Creating shapes geometry...")
        shapes_geometry = _get_shapes_geometry(gtfs_data["shapes"])

        if shapes_geometry is not None:
            # Add the shapes GeoSeries to the dictionary
            gtfs_data["shapes"] = gpd.GeoDataFrame(
                gtfs_data["shapes"], geometry=shapes_geometry, crs="EPSG:4326")

    logger.info("GTFS data processing complete")
    return gtfs_data


# Function to create origin-destination pairs from GTFS data
def _create_od_pairs(gtfs_data):
    """
    Create origin-destination pairs with timestamps from GTFS data.
    
    Parameters
    ----------
    gtfs_data : dict
        Dictionary with GTFS dataframes
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with trip_id, orig_stop_id, dest_stop_id, and timestamp columns
    """
    stop_times = gtfs_data.get('stop_times')
    trips = gtfs_data.get('trips')
    calendar = gtfs_data.get('calendar')
    calendar_dates = gtfs_data.get('calendar_dates')
    
    if stop_times is None or trips is None:
        print("Missing required GTFS data tables.")
        return None
    
    # Process stop_times to convert arrival_time and departure_time to datetime
    stop_times_copy = stop_times.copy()
    
    # Ensure stop_sequence is numeric
    stop_times_copy['stop_sequence'] = pd.to_numeric(stop_times_copy['stop_sequence'], errors='coerce')
    
    # Sort by trip_id and stop_sequence to ensure correct order
    stop_times_copy = stop_times_copy.sort_values(['trip_id', 'stop_sequence'])
    
    # Create a shifted dataframe to get the next stop in the sequence
    next_stops = stop_times_copy.copy()
    next_stops = next_stops.rename(columns={
        'stop_id': 'dest_stop_id',
        'arrival_time': 'dest_arrival_time',
        'departure_time': 'dest_departure_time',
        'stop_sequence': 'dest_stop_sequence'
    })
    
    # Create origin-destination pairs within the same trip
    od_pairs = []
    
    for trip_id, trip_stops in stop_times_copy.groupby('trip_id'):
        trip_stops = trip_stops.sort_values('stop_sequence')
        
        # Skip trips with only one stop
        if len(trip_stops) <= 1:
            continue
            
        # Iterate through consecutive stops
        for i in range(len(trip_stops) - 1):
            orig_stop = trip_stops.iloc[i]
            dest_stop = trip_stops.iloc[i + 1]
            
            od_pair = {
                'trip_id': trip_id,
                'orig_stop_id': orig_stop['stop_id'],
                'dest_stop_id': dest_stop['stop_id'],
                'departure_time': orig_stop['departure_time'],
                'arrival_time': dest_stop['arrival_time'],
                'orig_stop_sequence': orig_stop['stop_sequence'],
                'dest_stop_sequence': dest_stop['stop_sequence']
            }
            od_pairs.append(od_pair)
    
    # Create DataFrame from the list of OD pairs
    od_df = pd.DataFrame(od_pairs)
    
    # Merge with trips to get service_id
    od_df = od_df.merge(trips[['trip_id', 'service_id']], on='trip_id', how='left')
    
    return od_df


def _get_service_dates(gtfs_data, start_date=None, end_date=None):
    """
    Get the dates when each service_id is active based on calendar and calendar_dates.
    
    Parameters
    ----------
    gtfs_data : dict
        Dictionary with GTFS dataframes
    start_date : str, optional
        Start date in YYYYMMDD format, defaults to calendar's start_date if None
    end_date : str, optional
        End date in YYYYMMDD format, defaults to calendar's end_date if None
        
    Returns
    -------
    dict
        Dictionary mapping service_id to a list of dates (as datetime objects)
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    calendar = gtfs_data.get('calendar')
    calendar_dates = gtfs_data.get('calendar_dates')
    
    service_dates = {}
    
    # Process regular service from calendar.txt
    if calendar is not None:
        # If start_date and end_date are not provided, use min and max from calendar
        if start_date is None and 'start_date' in calendar.columns:
            min_start_date = calendar['start_date'].min()
            start_date = min_start_date if pd.notna(min_start_date) else None
        
        if end_date is None and 'end_date' in calendar.columns:
            max_end_date = calendar['end_date'].max()
            end_date = max_end_date if pd.notna(max_end_date) else None
                    
        # Convert dates to datetime objects
        start_dt = datetime.strptime(str(start_date), '%Y%m%d')
        end_dt = datetime.strptime(str(end_date), '%Y%m%d')
        
        # Map days of week to calendar columns
        day_mapping = {
            0: 'monday',
            1: 'tuesday',
            2: 'wednesday',
            3: 'thursday',
            4: 'friday',
            5: 'saturday',
            6: 'sunday'
        }
        
        # Create a set of dates for each service_id
        for _, row in calendar.iterrows():
            service_id = row['service_id']
            service_start = datetime.strptime(str(row['start_date']), '%Y%m%d')
            service_end = datetime.strptime(str(row['end_date']), '%Y%m%d')
            
            # Use the intersection of our date range and the service date range
            date_range_start = max(start_dt, service_start)
            date_range_end = min(end_dt, service_end)
            
            # Initialize empty list for this service_id
            if service_id not in service_dates:
                service_dates[service_id] = []
            
            # Add each applicable date
            current_date = date_range_start
            while current_date <= date_range_end:
                day_of_week = current_date.weekday()  # 0 = Monday, 6 = Sunday
                day_column = day_mapping.get(day_of_week)
                
                # If this service runs on this day of the week
                if day_column in row and row[day_column]:
                    service_dates[service_id].append(current_date)
                
                current_date += timedelta(days=1)
    
    # Process exceptions from calendar_dates.txt
    if calendar_dates is not None:
        for _, row in calendar_dates.iterrows():
            service_id = row['service_id']
            date_str = row['date']
            exception_type = int(row['exception_type']) if pd.notna(row['exception_type']) else 0
            
            try:
                exception_date = datetime.strptime(str(date_str), '%Y%m%d')
                
                # Initialize if this service_id isn't in the dict yet
                if service_id not in service_dates:
                    service_dates[service_id] = []
                
                # 1 = service added, 2 = service removed
                if exception_type == 1 and exception_date not in service_dates[service_id]:
                    service_dates[service_id].append(exception_date)
                elif exception_type == 2 and exception_date in service_dates[service_id]:
                    service_dates[service_id].remove(exception_date)
            except (ValueError, TypeError):
                print(f"Could not parse date {date_str} for service_id {service_id}")
    
    return service_dates


def _create_timestamp(time_str, date_obj):
    """Convert GTFS time string and date to a timestamp"""
    from datetime import datetime, timedelta
    
    if pd.isna(time_str):
        return None
        
    # GTFS times can be >24 hours, so we need to handle that
    try:
        hours, minutes, seconds = map(int, time_str.split(':'))
        days_offset = hours // 24
        hours = hours % 24
        
        # Create the timestamp
        timestamp = datetime(date_obj.year, date_obj.month, date_obj.day, hours, minutes, seconds)
        
        # Add any days offset for times >24 hours
        if days_offset > 0:
            timestamp += timedelta(days=days_offset)
            
        return timestamp
    except (ValueError, AttributeError):
        return None


def _expand_od_pairs_with_dates(od_pairs, service_dates):
    """
    Expand origin-destination pairs by combining with service dates.
    
    Parameters
    ----------
    od_pairs : pandas.DataFrame
        DataFrame with trip_id, orig_stop_id, dest_stop_id, etc.
    service_dates : dict
        Dictionary mapping service_id to list of dates
        
    Returns
    -------
    pandas.DataFrame
        Expanded DataFrame with complete timestamps
    """
    expanded_rows = []
    
    for _, row in od_pairs.iterrows():
        service_id = row['service_id']
        
        # Skip if no dates for this service_id
        if service_id not in service_dates or not service_dates[service_id]:
            continue
        
        for date in service_dates[service_id]:
            # Create departure and arrival timestamps
            departure_timestamp = _create_timestamp(row['departure_time'], date)
            arrival_timestamp = _create_timestamp(row['arrival_time'], date)
            
            if departure_timestamp and arrival_timestamp:
                expanded_row = {
                    'trip_id': row['trip_id'],
                    'service_id': service_id,
                    'orig_stop_id': row['orig_stop_id'],
                    'dest_stop_id': row['dest_stop_id'],
                    'departure_timestamp': departure_timestamp,
                    'arrival_timestamp': arrival_timestamp,
                    'travel_time_seconds': (arrival_timestamp - departure_timestamp).total_seconds(),
                    'date': date.strftime('%Y-%m-%d'),
                    'orig_stop_sequence': row['orig_stop_sequence'],
                    'dest_stop_sequence': row['dest_stop_sequence']
                }
                expanded_rows.append(expanded_row)
    
    return pd.DataFrame(expanded_rows)


# Create a function to export the OD pairs to a GeoDataFrame for visualization
def _create_od_gdf(expanded_od_pairs, gtfs_data):
    """
    Create a GeoDataFrame with LineString geometries for origin-destination pairs.
    
    Parameters
    ----------
    expanded_od_pairs : pandas.DataFrame
        DataFrame with origin-destination pair information
    gtfs_data : dict
        Dictionary with GTFS dataframes
        
    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with LineString geometries for OD pairs
    """
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    
    # Get stops dataframe with geometry
    stops = gtfs_data.get('stops')
    
    if stops is None or expanded_od_pairs is None or expanded_od_pairs.empty:
        print("Missing required data to create GeoDataFrame")
        return None
        
    # Create a dictionary mapping stop_id to Point geometry
    stop_geometries = {}
    for _, stop in stops.iterrows():
        if 'geometry' in stop:
            stop_geometries[stop['stop_id']] = stop['geometry']
        elif all(col in stop for col in ['stop_lon', 'stop_lat']):
            stop_geometries[stop['stop_id']] = Point(stop['stop_lon'], stop['stop_lat'])
    
    # Create LineString geometries for each OD pair
    geometries = []
    for _, row in expanded_od_pairs.iterrows():
        orig_geom = stop_geometries.get(row['orig_stop_id'])
        dest_geom = stop_geometries.get(row['dest_stop_id'])
        
        if orig_geom and dest_geom:
            geometries.append(LineString([orig_geom, dest_geom]))
        else:
            geometries.append(None)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        expanded_od_pairs,
        geometry=geometries,
        crs="EPSG:4326"
    )
    
    return gdf


def _expand_od_pairs_with_dates_gen(od_pairs, service_dates):
    """
    Generator version of expanding OD pairs with dates.
    Yields one expanded row (as a dict) at a time.
    """
    for _, row in od_pairs.iterrows():
        service_id = row['service_id']
        if service_id not in service_dates or not service_dates[service_id]:
            continue
        for date in service_dates[service_id]:
            departure_timestamp = _create_timestamp(row['departure_time'], date)
            arrival_timestamp = _create_timestamp(row['arrival_time'], date)
            if departure_timestamp and arrival_timestamp:
                yield {
                    'trip_id': row['trip_id'],
                    'service_id': service_id,
                    'orig_stop_id': row['orig_stop_id'],
                    'dest_stop_id': row['dest_stop_id'],
                    'departure_timestamp': departure_timestamp,
                    'arrival_timestamp': arrival_timestamp,
                    'travel_time_seconds': (arrival_timestamp - departure_timestamp).total_seconds(),
                    'date': date.strftime('%Y-%m-%d'),
                    'orig_stop_sequence': row['orig_stop_sequence'],
                    'dest_stop_sequence': row['dest_stop_sequence']
                }


def _get_od_pairs_generator(od_pairs, service_dates, gtfs_data, include_geometry, chunk_size):
    """
    Generator helper that yields chunks (as (Geo)DataFrames).
    """
    od_rows_gen = _expand_od_pairs_with_dates_gen(od_pairs, service_dates)
    chunk = []
    for idx, row in enumerate(od_rows_gen, start=1):
        chunk.append(row)
        if idx % chunk_size == 0:
            df_chunk = pd.DataFrame(chunk)
            if include_geometry:
                yield _create_od_gdf(df_chunk, gtfs_data)
            else:
                yield df_chunk
            chunk = []
    if chunk:
        df_chunk = pd.DataFrame(chunk)
        if include_geometry:
            yield _create_od_gdf(df_chunk, gtfs_data)
        else:
            yield df_chunk

def get_od_pairs(gtfs_data, start_date=None, end_date=None, include_geometry=True,
                 as_generator=False, chunk_size=10000):
    """
    Generate origin-destination pairs with timestamps from GTFS data.
    
    When as_generator is False this function returns a complete (Geo)DataFrame,
    and when True it returns a generator that yields chunks.
    
    Parameters
    ----------
    gtfs_data : dict
        Dictionary with GTFS dataframes from load_gtfs.
    start_date : str, optional
        Start date in YYYYMMDD format; if None, defaults from calendar.
    end_date : str, optional
        End date in YYYYMMDD format; if None, defaults from calendar.
    include_geometry : bool, default True
        Whether to include LineString geometries connecting the stops.
    as_generator : bool, default False
        If True, return a generator yielding GeoDataFrame chunks.
    chunk_size : int, default 10000
        Number of rows per chunk when using the generator.
        
    Returns
    -------
    geopandas.GeoDataFrame or pandas.DataFrame or generator
        If as_generator is False, returns a complete GeoDataFrame (or DataFrame if include_geometry=False).
        If as_generator is True, returns a generator yielding (Geo)DataFrame chunks.
    """
    logger.info("Creating origin-destination pairs from GTFS data...")
    
    # 1. Create basic OD pairs
    od_pairs = _create_od_pairs(gtfs_data)
    if od_pairs is None or od_pairs.empty:
        logger.error("Failed to create origin-destination pairs")
        return None
    
    # 2. Get service dates
    service_dates = _get_service_dates(gtfs_data, start_date, end_date)
    if not service_dates:
        logger.warning("No service dates found in calendar data")
        return od_pairs  # fallback: basic OD pairs
    
    if as_generator:
        return _get_od_pairs_generator(od_pairs, service_dates, gtfs_data, include_geometry, chunk_size)
    else:
        # Fully materialize the expansion
        expanded_od = _expand_od_pairs_with_dates(od_pairs, service_dates)
        if include_geometry:
            od_gdf = _create_od_gdf(expanded_od, gtfs_data)
            logger.info("Origin-destination pair generation complete")
            return od_gdf
        else:
            logger.info("Origin-destination pair generation complete")
            return expanded_od

def _time_to_seconds(time_str):
    if pd.isna(time_str):
        return np.nan
    if isinstance(time_str, (int, float)):
        return time_str
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + int(s)
    return np.nan

def _create_travel_summary_gdf(result, stops):
    # Merge coordinates from origin stops
    result = result.merge(
        stops[['stop_lat', 'stop_lon']],
        left_on='from_stop_id',
        right_index=True,
        suffixes=('', '_from')
    )
    
    # Merge coordinates from destination stops
    result = result.merge(
        stops[['stop_lat', 'stop_lon']],
        left_on='to_stop_id',
        right_index=True,
        suffixes=('_from', '_to')
    )
    
    # Create LineString geometries for each stop pair
    geometries = [
        LineString([(row['stop_lon_from'], row['stop_lat_from']),
                    (row['stop_lon_to'], row['stop_lat_to'])])
        for _, row in result.iterrows()
    ]
    
    gdf = gpd.GeoDataFrame(
        result[['from_stop_id', 'to_stop_id', 'travel_time', 'frequency']],
        geometry=geometries,
        crs="EPSG:4326"
    )
    return gdf


def _vectorized_time_to_seconds(time_series):
    """
    Efficiently convert a series of GTFS time strings to seconds.
    
    Parameters
    ----------
    time_series : pandas.Series
        Series containing time strings in HH:MM:SS format
        
    Returns
    -------
    pandas.Series
        Series containing time values converted to seconds
    """
    if time_series.dtype == 'object':
        # Only process string values
        mask = ~pd.isna(time_series)
        result = pd.Series(np.nan, index=time_series.index)
        
        if mask.any():
            # Process only non-NaN values
            time_parts = time_series[mask].str.split(':', expand=True).astype(int)
            result[mask] = time_parts[0] * 3600 + time_parts[1] * 60 + time_parts[2]
        
        return result
    else:
        # If already numeric, return as is
        return time_series


def create_travel_summary_network(gtfs_data, start_time=None, end_time=None, 
                                  calendar_start=None, calendar_end=None, as_gdf=True):
    """
    Create a network representing travel times and frequencies between stops.
    
    Parameters
    ----------
    gtfs_data : dict
        Dictionary with GTFS dataframes from load_gtfs
    start_time : str, optional
        Start time of day (HH:MM:SS) to filter trips
    end_time : str, optional
        End time of day (HH:MM:SS) to filter trips
    calendar_start : str, optional
        Start date in YYYYMMDD format to filter by service calendar
    calendar_end : str, optional
        End date in YYYYMMDD format to filter by service calendar
    as_gdf : bool, default True
        If True, return a GeoDataFrame; if False, return a dictionary
        
    Returns
    -------
    geopandas.GeoDataFrame or dict
        Network of stop connections with travel times and frequencies
    """    
    # Extract and preprocess the necessary dataframes
    stop_times = gtfs_data["stop_times"].copy()
    trips = gtfs_data["trips"][['trip_id', 'service_id']].copy()
    
    # Convert time columns to seconds (vectorized for speed)
    stop_times['arrival_time_sec'] = _vectorized_time_to_seconds(stop_times['arrival_time'])
    stop_times['departure_time_sec'] = _vectorized_time_to_seconds(stop_times['departure_time'])
    
    # Apply time-of-day filters efficiently
    if start_time is not None:
        start_time_sec = _time_to_seconds(str(start_time))
        stop_times = stop_times[stop_times['departure_time_sec'] >= start_time_sec]
    if end_time is not None:
        end_time_sec = _time_to_seconds(str(end_time))
        stop_times = stop_times[stop_times['arrival_time_sec'] <= end_time_sec]
    
    # Merge with trips to get service_id (using optimal merge strategy)
    stop_times = pd.merge(stop_times, trips, on='trip_id', how='inner')
    
    # Handle calendar filtering
    if calendar_start is not None or calendar_end is not None:
        # Get valid service dates and calculate service frequency
        service_dates = _get_service_dates(gtfs_data, calendar_start, calendar_end)
        
        # Create a mapping from service_id to service count for efficient lookup
        service_counts = {s_id: len(dates) for s_id, dates in service_dates.items()}
        
        # Add service counts using map (faster than apply)
        stop_times['service_count'] = stop_times['service_id'].map(service_counts).fillna(0)
        
        # Filter out trips with no valid service dates
        stop_times = stop_times[stop_times['service_count'] > 0]
    else:
        # Without calendar filtering, use uniform weight
        stop_times['service_count'] = 1
    
    # Create next stop info efficiently by sorting once then using shift
    stop_times = stop_times.sort_values(['trip_id', 'stop_sequence'])
    
    # Calculate next stop info within each trip
    stop_times['next_stop_id'] = stop_times.groupby('trip_id')['stop_id'].shift(-1)
    stop_times['next_arrival_time_sec'] = stop_times.groupby('trip_id')['arrival_time_sec'].shift(-1)
    
    # Calculate travel times vectorized
    valid_pairs = stop_times.dropna(subset=['next_stop_id', 'next_arrival_time_sec']).copy()
    valid_pairs['travel_time'] = valid_pairs['next_arrival_time_sec'] - valid_pairs['departure_time_sec']
    
    # Filter invalid pairs (all at once)
    valid_pairs = valid_pairs[valid_pairs['travel_time'] > 0]
    
    # Pre-calculate weights for aggregation
    valid_pairs['weighted_time'] = valid_pairs['travel_time'] * valid_pairs['service_count']
    
    # Efficient groupby aggregation with pre-calculated values
    result = valid_pairs.groupby(['stop_id', 'next_stop_id']).agg(
        weighted_time=('weighted_time', 'sum'),
        total_service_count=('service_count', 'sum')
    ).reset_index()
    
    # Calculate weighted average travel time
    result['travel_time'] = result['weighted_time'] / result['total_service_count']
    result['frequency'] = result['total_service_count']
    result = result.drop(['weighted_time', 'total_service_count'], axis=1)
    
    # Return dictionary if requested
    if not as_gdf:
        return {(row['stop_id'], row['next_stop_id']): (row['travel_time'], row['frequency']) 
                for _, row in result.iterrows()}
    
    # Prepare for GeoDataFrame creation
    result = result.rename(columns={'stop_id': 'from_stop_id', 'next_stop_id': 'to_stop_id'})
    stops = gtfs_data['stops'].set_index('stop_id')
    
    # Create GeoDataFrame
    gdf = _create_travel_summary_gdf(result, stops)
    
    return gdf