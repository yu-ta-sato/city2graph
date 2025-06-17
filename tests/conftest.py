import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import networkx as nx

# Try to import torch, skip tests if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

@pytest.fixture(scope="session")
def sample_crs():
    """Coordinate Reference System fixture."""
    return "EPSG:27700"

@pytest.fixture
def sample_nodes_gdf(sample_crs):
    """Fixture for a homogeneous nodes GeoDataFrame."""
    data = {
        "node_id": [1, 2, 3, 4],
        "feature1": [10.0, 20.0, 30.0, 40.0],
        "label1": [0, 1, 0, 1],
        "geometry": [Point(0, 0), Point(1, 1), Point(0, 1), Point(1, 0)],
    }
    gdf = gpd.GeoDataFrame(data, crs=sample_crs).set_index("node_id")
    return gdf

@pytest.fixture
def sample_edges_gdf():
    """Fixture for a homogeneous edges GeoDataFrame."""
    data = {
        "source_id": [1, 1, 2, 3],
        "target_id": [2, 3, 4, 4],
        "edge_feature1": [0.5, 0.8, 1.2, 2.5],
        "geometry": [
            LineString([(0,0), (1,1)]),
            LineString([(0,0), (0,1)]),
            LineString([(1,1), (1,0)]),
            LineString([(0,1), (1,0)]),
        ]
    }
    # Create a MultiIndex for source and target IDs
    multi_index = pd.MultiIndex.from_arrays([data['source_id'], data['target_id']], names=('source_id', 'target_id'))
    gdf = gpd.GeoDataFrame(data, index=multi_index, crs="EPSG:27700") # Use a CRS for edges too
    return gdf


@pytest.fixture
def sample_hetero_nodes_dict(sample_crs):
    """Fixture for a dictionary of heterogeneous nodes GeoDataFrames."""
    buildings_data = {
        "building_id": ["b1", "b2", "b3"],
        "b_feat1": [100.0, 150.0, 120.0],
        "b_label": [1, 0, 1],
        "geometry": [Point(10, 10), Point(11, 11), Point(10, 11)],
    }
    buildings_gdf = gpd.GeoDataFrame(buildings_data, crs=sample_crs).set_index("building_id")

    roads_data = {
        "road_id": ["r1", "r2"],
        "r_feat1": [5.5, 6.0],
        "r_label": [0, 0],
        "geometry": [Point(10, 12), Point(12, 12)],
    }
    roads_gdf = gpd.GeoDataFrame(roads_data, crs=sample_crs).set_index("road_id")

    return {"building": buildings_gdf, "road": roads_gdf}

@pytest.fixture
def sample_hetero_edges_dict(sample_crs):
    """Fixture for a dictionary of heterogeneous edges GeoDataFrames."""
    # Connects buildings to roads
    connections_data = {
        "building_id": ["b1", "b2", "b3"],
        "road_id": ["r1", "r1", "r2"],
        "conn_feat1": [1.0, 2.0, 3.0],
        "geometry": [
            LineString([(10,10), (10,12)]),
            LineString([(11,11), (10,12)]),
            LineString([(10,11), (12,12)]),
        ]
    }
    connections_multi_index = pd.MultiIndex.from_arrays(
        [connections_data['building_id'], connections_data['road_id']],
        names=('building_id', 'road_id')
    )
    connections_gdf = gpd.GeoDataFrame(connections_data, index=connections_multi_index, crs=sample_crs)

    # Roads connect to other roads (example of same-type connection)
    road_links_data = {
        "source_road_id": ["r1"],
        "target_road_id": ["r2"],
        "link_feat1": [0.7],
         "geometry": [LineString([(10,12), (12,12)])]
    }
    road_links_multi_index = pd.MultiIndex.from_arrays(
        [road_links_data['source_road_id'], road_links_data['target_road_id']],
        names=('source_road_id', 'target_road_id')
    )
    road_links_gdf = gpd.GeoDataFrame(road_links_data, index=road_links_multi_index, crs=sample_crs)


    return {
        ("building", "connects_to", "road"): connections_gdf,
        ("road", "links_to", "road"): road_links_gdf,
    }

@pytest.fixture
def sample_nx_graph():
    """Fixture for a NetworkX graph."""
    graph = nx.Graph()
    graph.add_node(1, feature1=10.0, label1=0, pos=(0,0), geometry=Point(0,0))
    graph.add_node(2, feature1=20.0, label1=1, pos=(1,1), geometry=Point(1,1))
    graph.add_node(3, feature1=30.0, label1=0, pos=(0,1), geometry=Point(0,1))
    graph.add_edge(1, 2, edge_feature1=0.5, geometry=LineString([(0,0),(1,1)]))
    graph.add_edge(1, 3, edge_feature1=0.8, geometry=LineString([(0,0),(0,1)]))
    graph.graph['crs'] = "EPSG:27700" # Add CRS to graph attributes
    return graph

# Pytest skipif marker for tests requiring torch
requires_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch or PyTorch Geometric is not available.")
