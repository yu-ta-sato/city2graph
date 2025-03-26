import pytest
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Polygon, Point, MultiLineString

@pytest.fixture
def simple_line():
    """Create a simple horizontal line from (0,0) to (10,0)"""
    return LineString([(0, 0), (10, 0)])

@pytest.fixture
def complex_line():
    """Create a more complex line with multiple segments"""
    return LineString([(0, 0), (2, 3), (5, 2), (8, 5), (10, 0)])

@pytest.fixture
def tunnel_road_flags():
    """Create sample road flags JSON with tunnel indicators"""
    # Full tunnel
    full_tunnel = '[{"values": {"is_tunnel": true}}]'
    
    # Partial tunnel (middle section)
    partial_tunnel = '[{"values": {"is_tunnel": true}, "between": [0.3, 0.7]}]'
    
    # Multiple tunnel sections
    multiple_tunnels = '''[
        {"values": {"is_tunnel": true}, "between": [0.2, 0.3]},
        {"values": {"is_tunnel": true}, "between": [0.6, 0.8]}
    ]'''
    
    # No tunnel
    no_tunnel = '[{"values": {"some_other_flag": true}}]'
    
    return {
        "full": full_tunnel,
        "partial": partial_tunnel,
        "multiple": multiple_tunnels,
        "none": no_tunnel
    }

@pytest.fixture
def grid_data():
    """Create a simple grid of buildings, roads, and tessellations"""
    # Buildings: 3x3 grid of 10x10 squares with 5-unit spacing
    buildings = []
    for i in range(3):
        for j in range(3):
            x = i * 15
            y = j * 15
            buildings.append(Polygon([(x, y), (x+10, y), (x+10, y+10), (x, y+10)]))
    
    buildings_gdf = gpd.GeoDataFrame(
        {'id': [f'b{i}' for i in range(len(buildings))],
         'geometry': buildings},
        crs="EPSG:27700"
    )
    
    # Roads: 4 horizontal and 4 vertical lines forming a grid
    h_roads = []
    for j in range(4):
        y = j * 15 - 2.5 if j > 0 else 0
        h_roads.append(LineString([(0, y), (40, y)]))
    
    v_roads = []
    for i in range(4):
        x = i * 15 - 2.5 if i > 0 else 0
        v_roads.append(LineString([(x, 0), (x, 40)]))
    
    roads = h_roads + v_roads
    roads_gdf = gpd.GeoDataFrame(
        {'id': [f'r{i}' for i in range(len(roads))],
         'subtype': 'road',
         'class': 'residential',
         'road_flags': [None] * len(roads),
         'geometry': roads},
        crs="EPSG:27700"
    )
    
    # Tessellations: slightly larger than buildings
    tessellations = []
    for i in range(3):
        for j in range(3):
            x = i * 15 - 1
            y = j * 15 - 1
            tessellations.append(Polygon([(x, y), (x+12, y), (x+12, y+12), (x, y+12)]))
    
    tessellation_gdf = gpd.GeoDataFrame(
        {'tess_id': [f't{i}' for i in range(len(tessellations))],
         'enclosure_index': [i//3 for i in range(len(tessellations))],
         'geometry': tessellations},
        crs="EPSG:27700"
    )
    
    return {
        "buildings": buildings_gdf,
        "roads": roads_gdf,
        "tessellations": tessellation_gdf
    }
