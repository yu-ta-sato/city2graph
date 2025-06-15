# tests/test_morphology.py
import pytest
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
import numpy as np
import math
import logging

# Assuming city2graph is installed or in PYTHONPATH
from city2graph.morphology import (
    morphological_graph,
    private_to_private_graph,
    private_to_public_graph,
    public_to_public_graph,
)

# It's good practice to have a consistent CRS for tests
TEST_CRS = "EPSG:32632" # Example projected CRS, ensure your data matches

# --- Fixtures ---

@pytest.fixture
def basic_buildings_gdf():
    """A simple GeoDataFrame with a few building polygons."""
    buildings_data = {
        'building_id': [1, 2, 3],
        'geometry': [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
            Polygon([(2, 0), (2, 1), (3, 1), (3, 0)]),
            Polygon([(0, 2), (0, 3), (1, 3), (1, 2)])
        ]
    }
    return gpd.GeoDataFrame(buildings_data, crs=TEST_CRS)

@pytest.fixture
def basic_segments_gdf():
    """A simple GeoDataFrame with a few street segments."""
    segments_data = {
        'segment_id': ['S1', 'S2', 'S3'],
        'geometry': [
            LineString([(0.5, -1), (0.5, 3.5)]),  # Vertical
            LineString([(-1, 0.5), (3.5, 0.5)]), # Horizontal
            LineString([(-1, 2.5), (3.5, 2.5)])  # Horizontal
        ]
    }
    segments = gpd.GeoDataFrame(segments_data, crs=TEST_CRS)
    # Add a default barrier geometry for tests that might use it
    segments["barrier_geometry"] = segments.geometry.buffer(0.1)
    return segments

@pytest.fixture
def center_point_gdf():
    """A sample center point for filtering."""
    return gpd.GeoSeries([Point(1.5, 1.5)], crs=TEST_CRS)

@pytest.fixture
def empty_buildings_gdf():
    return gpd.GeoDataFrame({'geometry': []}, crs=TEST_CRS)

@pytest.fixture
def empty_segments_gdf():
    return gpd.GeoDataFrame({'geometry': []}, crs=TEST_CRS)


# --- Tests for morphological_graph ---

def test_morphological_graph_basic_run(basic_buildings_gdf, basic_segments_gdf):
    """Test morphological_graph with basic inputs."""
    nodes, edges = morphological_graph(basic_buildings_gdf, basic_segments_gdf)

    assert isinstance(nodes, dict)
    assert isinstance(edges, dict)
    assert "private" in nodes
    assert "public" in nodes
    assert isinstance(nodes["private"], gpd.GeoDataFrame)
    assert isinstance(nodes["public"], gpd.GeoDataFrame)

    assert ("private", "touched_to", "private") in edges
    assert ("public", "connected_to", "public") in edges
    assert ("private", "faced_to", "public") in edges
    assert isinstance(edges[("private", "touched_to", "private")], gpd.GeoDataFrame)
    assert isinstance(edges[("public", "connected_to", "public")], gpd.GeoDataFrame)
    assert isinstance(edges[("private", "faced_to", "public")], gpd.GeoDataFrame)

    # Check CRS
    assert nodes["private"].crs == TEST_CRS
    assert nodes["public"].crs == TEST_CRS
    assert edges[("private", "touched_to", "private")].crs == TEST_CRS
    # ... and so on for other edge GDFs if they have geometry

    # Check for ID columns (default names)
    if not nodes["private"].empty:
        assert "private_id" in nodes["private"].index.name or "private_id" in nodes["private"].columns
    if not nodes["public"].empty:
         assert "public_id" in nodes["public"].index.name or "public_id" in nodes["public"].columns


@pytest.mark.parametrize("clipping_buffer_val, distance_val", [
    (10.0, 50.0),
    (math.inf, 50.0),
    (0.0, 50.0)
])
def test_morphological_graph_filtering(basic_buildings_gdf, basic_segments_gdf, center_point_gdf, clipping_buffer_val, distance_val):
    """Test with center_point, distance, and clipping_buffer."""
    nodes, edges = morphological_graph(
        basic_buildings_gdf,
        basic_segments_gdf,
        center_point=center_point_gdf,
        distance=distance_val,
        clipping_buffer=clipping_buffer_val
    )
    assert isinstance(nodes, dict) # Basic check, more detailed checks on filtered results would be needed
    # For example, check if the number of public nodes is reduced as expected.
    # This requires knowing the network structure and distances.
    # For now, we just check if it runs.
    if not basic_segments_gdf.empty: # only check if there were segments to begin with
         # A very basic check, assuming filtering might reduce nodes.
         # This is not a robust check for filtering correctness.
        assert len(nodes["public"]) <= len(basic_segments_gdf)


def test_morphological_graph_keep_buildings(basic_buildings_gdf, basic_segments_gdf):
    """Test keep_buildings parameter."""
    nodes, _ = morphological_graph(
        basic_buildings_gdf, basic_segments_gdf, keep_buildings=True
    )
    if not nodes["private"].empty and not basic_buildings_gdf.empty:
        # Assuming 'building_id' was in the original buildings_gdf
        # and _add_building_info merges it.
        assert "building_id" in nodes["private"].columns # or whatever columns were in original buildings

def test_morphological_graph_custom_ids(basic_buildings_gdf, basic_segments_gdf):
    """Test with custom ID column names."""
    b_gdf = basic_buildings_gdf.copy()
    s_gdf = basic_segments_gdf.copy()
    b_gdf["custom_b_id"] = b_gdf.index.astype(str) + "_b"
    s_gdf["custom_s_id"] = s_gdf.index.astype(str) + "_s"

    nodes, edges = morphological_graph(
        b_gdf, s_gdf, private_id_col="custom_b_id", public_id_col="custom_s_id"
    )
    if not nodes["private"].empty:
        assert nodes["private"].index.name == "custom_b_id"
    if not nodes["public"].empty:
        assert nodes["public"].index.name == "custom_s_id"

    # Check edge GDFs for corresponding ID columns
    # e.g. for priv_pub edges
    priv_pub_edges = edges[("private", "faced_to", "public")]
    if not priv_pub_edges.empty:
        assert "custom_b_id" in priv_pub_edges.columns
        assert "custom_s_id" in priv_pub_edges.columns


def test_morphological_graph_error_negative_clipping_buffer(basic_buildings_gdf, basic_segments_gdf):
    """Test ValueError for negative clipping_buffer."""
    with pytest.raises(ValueError, match="clipping_buffer cannot be negative"):
        morphological_graph(basic_buildings_gdf, basic_segments_gdf, clipping_buffer=-10.0)

def test_morphological_graph_empty_inputs(empty_buildings_gdf, empty_segments_gdf):
    """Test with empty GeoDataFrames."""
    nodes, edges = morphological_graph(empty_buildings_gdf, empty_segments_gdf)
    assert nodes["private"].empty
    assert nodes["public"].empty
    assert edges[("private", "touched_to", "private")].empty
    assert edges[("public", "connected_to", "public")].empty
    assert edges[("private", "faced_to", "public")].empty

@pytest.mark.parametrize("contiguity_val", ["queen", "rook"])
def test_morphological_graph_contiguity(basic_buildings_gdf, basic_segments_gdf, contiguity_val):
    """Test different contiguity options."""
    # This primarily affects private-to-private graph, check if it runs
    nodes, edges = morphological_graph(
        basic_buildings_gdf, basic_segments_gdf, contiguity=contiguity_val
    )
    assert isinstance(nodes, dict) # Basic check

def test_morphological_graph_no_priv_pub_connection_warning(caplog, basic_buildings_gdf):
    """Test warning when no private-public connections are found."""
    # Create segments far away from buildings
    segments_far_away_data = {
        'segment_id': ['S_far'],
        'geometry': [LineString([(100, 100), (101, 101)])]
    }
    segments_far_away_gdf = gpd.GeoDataFrame(segments_far_away_data, crs=TEST_CRS)

    with caplog.at_level(logging.WARNING):
        morphological_graph(basic_buildings_gdf, segments_far_away_gdf)
    
    if not basic_buildings_gdf.empty and not segments_far_away_gdf.empty:
         # This warning depends on the actual outcome of private_to_public_graph
         # which might not be empty if tessellation cells extend far.
         # A more robust test would ensure priv_pub is indeed empty.
        pass # For now, just check if it runs. A more specific check on caplog.text might be needed.


# --- Tests for private_to_private_graph ---

@pytest.mark.parametrize("contiguity_val", ["queen", "rook"])
def test_private_to_private_graph_basic(basic_buildings_gdf, contiguity_val):
    """Test private_to_private_graph with queen and rook contiguity."""
    # Note: private_to_private_graph expects tessellation, not raw buildings.
    # For a simple test, we can use buildings if they are like tessellation cells.
    # A proper test would use output from create_tessellation.
    # Here, basic_buildings_gdf might act as a proxy for simple tessellation.
    
    # Create a dummy tessellation GDF from buildings for testing purposes
    # In real scenario, this would come from create_tessellation
    tess_gdf = basic_buildings_gdf.copy()
    tess_gdf["private_id"] = tess_gdf.index.astype(str)
    tess_gdf = tess_gdf.set_index("private_id")


    # The function expects an ID column, let's ensure 'private_id' exists and is used
    # The fixture basic_buildings_gdf doesn't have 'private_id' by default.
    # Let's create a simple tessellation-like structure for this test.
    
    # Building 1: (0,0)-(1,1)
    # Building 3: (0,2)-(1,3)
    # These are not adjacent with the current basic_buildings_gdf.
    # Let's make a fixture where some are adjacent.
    
    adjacent_buildings_data = {
        'private_id': [1, 2, 3], # ensure this column exists
        'geometry': [
            Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]), # B1
            Polygon([(1, 0), (1, 1), (2, 1), (2, 0)]), # B2, adjacent to B1 (rook & queen)
            Polygon([(0, 1), (0, 2), (1, 2), (1, 1)])  # B3, adjacent to B1 (rook & queen)
        ],
        'enclosure_index': [0, 0, 0] # For group_col testing
    }
    private_gdf = gpd.GeoDataFrame(adjacent_buildings_data, crs=TEST_CRS)
    
    edges_gdf = private_to_private_graph(
        private_gdf, private_id_col="private_id", contiguity=contiguity_val, group_col="enclosure_index"
    )

    assert isinstance(edges_gdf, gpd.GeoDataFrame)
    assert "from_private_id" in edges_gdf.columns
    assert "to_private_id" in edges_gdf.columns
    assert "geometry" in edges_gdf.columns
    if not private_gdf.empty and len(private_gdf) >=2 :
        # With the adjacent_buildings_data, we expect some edges
        # B1-B2, B1-B3. B2 and B3 are not directly adjacent in this setup.
        # So, 2 unique pairs, meaning 4 rows if edges are bidirectional or 2 if one-way stored.
        # The function should return unique pairs, typically as LineStrings.
        # The number of edges depends on implementation details (e.g., if (1,2) and (2,1) are both stored)
        # Based on typical adjacency graphs, we'd expect edges for B1-B2 and B1-B3.
        # The function _extract_adjacency_relationships seems to create (i, neighbor) pairs.
        # If B1 is 0, B2 is 1, B3 is 2: (0,1), (1,0), (0,2), (2,0) might be generated before unique edges.
        # The current implementation of private_to_private_graph seems to create LineStrings between centroids.
        # For B1-B2 and B1-B3, we expect 2 edges if they are unique undirected edges.
        # The example data has IDs 1,2,3. So (1,2) and (1,3) are expected.
        if contiguity_val == "queen" or contiguity_val == "rook": # for these simple box adjacencies
             assert len(edges_gdf) >= 2 # Expecting at least B1-B2 and B1-B3 connections
    else:
        assert edges_gdf.empty


def test_private_to_private_graph_empty_or_single(empty_buildings_gdf):
    private_gdf_empty = empty_buildings_gdf
    private_gdf_single = gpd.GeoDataFrame({'geometry': [Polygon([(0,0),(1,1),(1,0)])]}, crs=TEST_CRS)
    private_gdf_single["pid"] = ["p1"]

    edges_empty = private_to_private_graph(private_gdf_empty, private_id_col="pid")
    assert edges_empty.empty
    assert "from_private_id" in edges_empty.columns # Check schema even if empty

    edges_single = private_to_private_graph(private_gdf_single, private_id_col="pid")
    assert edges_single.empty


# --- Tests for private_to_public_graph ---

def test_private_to_public_graph_basic(basic_buildings_gdf, basic_segments_gdf):
    """Test private_to_public_graph with basic inputs."""
    # Again, this expects tessellation. Using buildings as a proxy.
    # Ensure ID columns are present
    tess_gdf = basic_buildings_gdf.copy()
    tess_gdf["private_id"] = ["p1", "p2", "p3"]
    
    seg_gdf = basic_segments_gdf.copy()
    seg_gdf["public_id"] = ["s1", "s2", "s3"]
    
    # Use a small tolerance for interface detection
    interfaces_gdf = private_to_public_graph(
        tess_gdf, seg_gdf, private_id_col="private_id", public_id_col="public_id", tolerance=0.1
    )

    assert isinstance(interfaces_gdf, gpd.GeoDataFrame)
    assert "private_id" in interfaces_gdf.columns
    assert "public_id" in interfaces_gdf.columns
    assert "geometry" in interfaces_gdf.columns # Should contain LineStrings of interfaces

    # With basic_buildings_gdf and basic_segments_gdf, some interfaces should be found.
    # E.g., Building 1 (0,0)-(1,1) should interface with Segment S1 (x=0.5) and S2 (y=0.5)
    # Building 3 (0,2)-(1,3) should interface with S1 (x=0.5) and S3 (y=2.5)
    if not tess_gdf.empty and not seg_gdf.empty:
        assert not interfaces_gdf.empty 
        # A more specific check would be on the number of interfaces or their lengths.


def test_private_to_public_no_interface(basic_buildings_gdf, basic_segments_gdf):
    tess_gdf = basic_buildings_gdf.copy()
    tess_gdf["private_id"] = ["p1", "p2", "p3"]
    
    # Create segments far away
    segments_far_data = {'public_id': ['s_far'], 'geometry': [LineString([(100,100), (101,100)])]}
    segments_far_gdf = gpd.GeoDataFrame(segments_far_data, crs=TEST_CRS)
    segments_far_gdf["barrier_geometry"] = segments_far_gdf.geometry.buffer(0.1)


    interfaces_gdf = private_to_public_graph(
        tess_gdf, segments_far_gdf, private_id_col="private_id", public_id_col="public_id"
    )
    assert interfaces_gdf.empty


# --- Tests for public_to_public_graph ---

def test_public_to_public_graph_basic(basic_segments_gdf):
    """Test public_to_public_graph for connected segments."""
    seg_gdf = basic_segments_gdf.copy()
    seg_gdf["public_id"] = seg_gdf["segment_id"] # Use existing 'segment_id'

    # S1 (0.5,-1)-(0.5,3.5)
    # S2 (-1,0.5)-(3.5,0.5)
    # S3 (-1,2.5)-(3.5,2.5)
    # S1 intersects S2 at (0.5, 0.5)
    # S1 intersects S3 at (0.5, 2.5)
    # S2 and S3 are parallel, do not intersect.
    # Expected connections: S1-S2, S1-S3 (and reverse if stored)
    
    connections_gdf = public_to_public_graph(seg_gdf)#, public_id_col="public_id")

    assert isinstance(connections_gdf, gpd.GeoDataFrame)
    assert "from_public_id" in connections_gdf.columns
    assert "to_public_id" in connections_gdf.columns
    assert "geometry" in connections_gdf.columns # Should contain Points of connection

    if not seg_gdf.empty:
        assert not connections_gdf.empty
        # We expect 2 unique connection points, so 2 rows in the GDF.
        # (S1,S2) and (S1,S3)
        assert len(connections_gdf) == 2 
        
        connected_pairs = set()
        for _, row in connections_gdf.iterrows():
            pair = tuple(sorted((row["from_public_id"], row["to_public_id"])))
            connected_pairs.add(pair)
        
        assert ("S1", "S2") in connected_pairs or ("S2", "S1") in connected_pairs
        assert ("S1", "S3") in connected_pairs or ("S3", "S1") in connected_pairs


def test_public_to_public_graph_disconnected():
    """Test with segments that do not connect."""
    disconnected_segments_data = {
        'public_id': ['DS1', 'DS2'],
        'geometry': [
            LineString([(0,0), (1,1)]),
            LineString([(2,2), (3,3)])
        ]
    }
    disconnected_gdf = gpd.GeoDataFrame(disconnected_segments_data, crs=TEST_CRS)
    connections_gdf = public_to_public_graph(disconnected_gdf, public_id_col="public_id")
    assert connections_gdf.empty
