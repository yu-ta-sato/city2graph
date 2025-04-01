import pytest
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point, Polygon, MultiLineString
from city2graph.morphology import (
    convert_gdf_to_dual,
    create_private_to_private,
    create_private_to_public,
    create_public_to_public,
    _get_adjacent_publics,
    _extract_dual_graph_nodes,
    _extract_node_connections,
    _find_additional_connections,
    _create_connecting_lines,
    _prep_contiguity_graph
)

class TestConvertGdfToDual:
    def test_convert_gdf_to_dual_basic(self, grid_data):
        """Test basic dual graph conversion with grid data"""
        roads_gdf = grid_data["roads"]
        dual_nodes, connections = convert_gdf_to_dual(roads_gdf, id_col='id')
        
        # Check that dual nodes were created
        assert not dual_nodes.empty
        assert len(dual_nodes) > 0
        
        # Check that connections dictionary exists with correct structure
        assert connections
        assert isinstance(connections, dict)
        
        # Each road should have connections
        for road_id in roads_gdf['id']:
            assert road_id in connections
            # For grid data, each road should connect to at least one other road
            assert len(connections[road_id]) > 0
    
    def test_convert_gdf_to_dual_empty(self):
        """Test conversion with empty GeoDataFrame"""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")
        dual_nodes, connections = convert_gdf_to_dual(empty_gdf)
        
        # Should return empty results
        assert dual_nodes.empty
        assert connections == {}
    
    def test_convert_gdf_to_dual_invalid_geometries(self):
        """Test conversion with invalid geometry types"""
        invalid_gdf = gpd.GeoDataFrame(
            {'id': ['p1', 'p2'], 'geometry': [Point(0, 0), Point(1, 1)]},
            crs="EPSG:27700"
        )
        
        # Should handle invalid geometries gracefully
        with pytest.warns(RuntimeWarning):
            dual_nodes, connections = convert_gdf_to_dual(invalid_gdf, id_col='id')
        
        assert dual_nodes.empty
        assert connections == {}
    
    def test_convert_gdf_to_dual_tolerance(self, grid_data):
        """Test effect of tolerance parameter on connections"""
        roads_gdf = grid_data["roads"].copy()
        
        # Create a disconnected line that's close to another line
        roads_gdf.loc[len(roads_gdf)] = {
            'id': 'close_line',
            'subtype': 'road',
            'class': 'residential',
            'road_flags': None,
            'geometry': LineString([(40.1, 0), (45, 0)])  # Close to the end of first horizontal road
        }
        
        # Test with small tolerance - should not connect
        dual_nodes_small, connections_small = convert_gdf_to_dual(
            roads_gdf, id_col='id', tolerance=0.01
        )
        
        # Test with large tolerance - should connect
        dual_nodes_large, connections_large = convert_gdf_to_dual(
            roads_gdf, id_col='id', tolerance=0.5
        )
        
        # Larger tolerance should find more connections
        assert 'close_line' in connections_large
        assert len(connections_large['close_line']) > 0
        
        # If 'close_line' exists in small connections, it shouldn't connect to r0
        if 'close_line' in connections_small and len(connections_small['close_line']) > 0:
            assert 'r0' not in connections_small['close_line']
    
    def test_convert_gdf_to_dual_id_generation(self):
        """Test that IDs are generated correctly when not provided"""
        lines = [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (2, 0)])
        ]
        gdf = gpd.GeoDataFrame({'geometry': lines}, crs="EPSG:27700")
        
        # Without explicit ID column
        dual_nodes, connections = convert_gdf_to_dual(gdf)
        
        # Should create default IDs (0, 1)
        assert not dual_nodes.empty
        assert 0 in connections
        assert 1 in connections


class TestGetAdjacentPublics:
    def test_get_adjacent_publics_basic(self, grid_data):
        """Test basic functionality to identify adjacent public spaces"""
        tessellations = grid_data["tessellations"]
        roads = grid_data["roads"]
        
        adjacent = _get_adjacent_publics(
            tessellations, 
            roads, 
            public_id_col="id", 
            private_id_col="tess_id",
            buffer=1
        )
        
        # Should return a dictionary
        assert isinstance(adjacent, dict)
        assert len(adjacent) > 0
        
        # Each tessellation should have adjacent roads
        for tess_id in tessellations["tess_id"]:
            assert tess_id in adjacent
            assert len(adjacent[tess_id]) > 0
    
    def test_get_adjacent_publics_empty(self):
        """Test with empty inputs"""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")
        
        # Both empty
        adjacent = _get_adjacent_publics(empty_gdf, empty_gdf)
        assert adjacent == {}
        
        # One empty
        buildings = gpd.GeoDataFrame(
            {'id': ['b1'], 'geometry': [Polygon([(0,0), (1,0), (1,1), (0,1)])]},
            crs="EPSG:27700"
        )
        adjacent = _get_adjacent_publics(buildings, empty_gdf)
        assert adjacent == {}
        
        adjacent = _get_adjacent_publics(empty_gdf, buildings)
        assert adjacent == {}
    
    def test_get_adjacent_publics_buffer(self, grid_data):
        """Test effect of buffer parameter on adjacency"""
        tessellations = grid_data["tessellations"]
        roads = grid_data["roads"]
        
        # Small buffer - fewer adjacencies
        small_adjacent = _get_adjacent_publics(
            tessellations, roads, public_id_col="id", private_id_col="tess_id", buffer=0.1
        )
        
        # Large buffer - more adjacencies
        large_adjacent = _get_adjacent_publics(
            tessellations, roads, public_id_col="id", private_id_col="tess_id", buffer=5
        )
        
        # Count total adjacencies
        small_count = sum(len(roads) for roads in small_adjacent.values())
        large_count = sum(len(roads) for roads in large_adjacent.values())
        
        # Large buffer should find more adjacencies
        assert large_count >= small_count
    
    def test_get_adjacent_publics_alternative_geometry(self, grid_data):
        """Test using alternative geometry column"""
        tessellations = grid_data["tessellations"]
        roads = grid_data["roads"].copy()
        
        # Create buffered geometry column
        roads["buffered_geom"] = roads.geometry.buffer(2)
        
        # Get adjacencies with regular geometry
        regular = _get_adjacent_publics(
            tessellations, roads, public_id_col="id", private_id_col="tess_id", buffer=1
        )
        
        # Get adjacencies with buffered geometry
        buffered = _get_adjacent_publics(
            tessellations, roads, public_id_col="id", private_id_col="tess_id", 
            public_geom_col="buffered_geom", buffer=1
        )
        
        # Buffered geometry should find more adjacencies
        regular_count = sum(len(roads) for roads in regular.values())
        buffered_count = sum(len(roads) for roads in buffered.values())
        assert buffered_count >= regular_count


class TestPrivateToPrivate:
    def test_create_private_to_private_basic(self, grid_data):
        """Test basic private-to-private connections"""
        tessellations = grid_data["tessellations"]
        
        connections = create_private_to_private(
            tessellations,
            private_id_col="tess_id",
            contiguity="queen"
        )
        
        # Check result structure
        assert not connections.empty
        assert "from_private_id" in connections.columns
        assert "to_private_id" in connections.columns
        assert "geometry" in connections.columns
        
        # Expected connection counts in a 3x3 grid with queen contiguity:
        # - Corner cells (4): 3 connections each
        # - Edge cells (4): 5 connections each
        # - Center cell (1): 8 connections
        # Total: 4*3 + 4*5 + 1*8 = 12 + 20 + 8 = 40
        # But connections are bi-directional, so divide by 2: 20 unique connections
        
        # Count connections for specific cells
        connections_from = connections["from_private_id"].value_counts().to_dict()
        
        # Corner cells
        assert connections_from.get("t0", 0) == 3  # Top-left
        assert connections_from.get("t2", 0) == 3  # Top-right
        assert connections_from.get("t6", 0) == 3  # Bottom-left
        assert connections_from.get("t8", 0) == 3  # Bottom-right
        
        # Center cell
        assert connections_from.get("t4", 0) == 8
    
    def test_create_private_to_private_with_grouping(self, grid_data):
        """Test private-to-private connections with grouping"""
        tessellations = grid_data["tessellations"]
        
        connections = create_private_to_private(
            tessellations,
            private_id_col="tess_id",
            group_col="enclosure_index",
            contiguity="queen"
        )
        
        # Check group column exists
        assert "enclosure_index" in connections.columns
        
        # Check connections only exist within same group
        for _, row in connections.iterrows():
            from_id = int(row["from_private_id"].split("t")[1])
            to_id = int(row["to_private_id"].split("t")[1])
            
            # In our test data, enclosure_index is row number (0, 1, 2)
            # Each row has 3 cells, so within same row: from_id // 3 == to_id // 3
            assert from_id // 3 == to_id // 3
            assert row["enclosure_index"] == from_id // 3
    
    def test_create_private_to_private_rook_vs_queen(self, grid_data):
        """Test difference between rook and queen contiguity"""
        tessellations = grid_data["tessellations"]
        
        # Queen contiguity (includes diagonals)
        queen_connections = create_private_to_private(
            tessellations,
            private_id_col="tess_id",
            contiguity="queen"
        )
        
        # Rook contiguity (only shares edges, no diagonals)
        rook_connections = create_private_to_private(
            tessellations,
            private_id_col="tess_id",
            contiguity="rook"
        )
        
        # Queen should have more connections than rook
        assert len(queen_connections) > len(rook_connections)
        
        # Check center cell (t4):
        # - Queen: should connect to all 8 surrounding cells
        # - Rook: should only connect to 4 cells (left, right, top, bottom)
        t4_queen = queen_connections[queen_connections["from_private_id"] == "t4"]
        t4_rook = rook_connections[rook_connections["from_private_id"] == "t4"]
        
        assert len(t4_queen) == 8
        assert len(t4_rook) == 4
        
        # Rook connections should be subset of queen connections
        queen_pairs = set(zip(queen_connections["from_private_id"], queen_connections["to_private_id"]))
        rook_pairs = set(zip(rook_connections["from_private_id"], rook_connections["to_private_id"]))
        assert rook_pairs.issubset(queen_pairs)
    
    def test_create_private_to_private_invalid_contiguity(self, grid_data):
        """Test with invalid contiguity parameter"""
        tessellations = grid_data["tessellations"]
        
        with pytest.raises(ValueError, match="contiguity must be 'queen' or 'rook'"):
            create_private_to_private(
                tessellations,
                private_id_col="tess_id",
                contiguity="invalid"
            )


class TestPrivateToPublic:
    def test_create_private_to_public_basic(self, grid_data):
        """Test basic private-to-public connections"""
        tessellations = grid_data["tessellations"]
        roads = grid_data["roads"]
        
        connections = create_private_to_public(
            tessellations,
            roads,
            private_id_col="tess_id",
            public_id_col="id",
            tolerance=1
        )
        
        # Check result structure
        assert not connections.empty
        assert "private_id" in connections.columns
        assert "public_id" in connections.columns
        assert "geometry" in connections.columns
        
        # Each tessellation should connect to at least one road
        connected_tessellations = set(connections["private_id"])
        all_tessellations = set(tessellations["tess_id"])
        assert connected_tessellations == all_tessellations
        
        # All connections should be LineStrings
        assert all(isinstance(geom, LineString) for geom in connections.geometry)
    
    def test_create_private_to_public_tolerance(self, grid_data):
        """Test effect of tolerance parameter on connections"""
        tessellations = grid_data["tessellations"]
        roads = grid_data["roads"]
        
        # With small tolerance
        small_connections = create_private_to_public(
            tessellations,
            roads,
            private_id_col="tess_id",
            public_id_col="id",
            tolerance=0.1
        )
        
        # With large tolerance
        large_connections = create_private_to_public(
            tessellations,
            roads,
            private_id_col="tess_id",
            public_id_col="id",
            tolerance=5
        )
        
        # Larger tolerance should find more connections
        assert len(large_connections) >= len(small_connections)
    
    def test_create_private_to_public_empty(self):
        """Test with empty inputs"""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")
        
        # Both empty
        connections = create_private_to_public(empty_gdf, empty_gdf)
        assert connections.empty
        
        # One empty
        buildings = gpd.GeoDataFrame(
            {'id': ['b1'], 'geometry': [Polygon([(0,0), (1,0), (1,1), (0,1)])]},
            crs="EPSG:27700"
        )
        roads = gpd.GeoDataFrame(
            {'id': ['r1'], 'geometry': [LineString([(0,0), (1,0)])]},
            crs="EPSG:27700"
        )
        
        connections = create_private_to_public(buildings, empty_gdf)
        assert connections.empty
        
        connections = create_private_to_public(empty_gdf, roads)
        assert connections.empty


class TestPublicToPublic:
    def test_create_public_to_public_basic(self, grid_data):
        """Test basic public-to-public connections"""
        roads = grid_data["roads"]
        
        connections = create_public_to_public(roads, public_id_col="id")
        
        # Check result structure
        assert not connections.empty
        assert "from_public_id" in connections.columns
        assert "to_public_id" in connections.columns
        assert "geometry" in connections.columns
        
        # All connections should be LineStrings
        assert all(isinstance(geom, LineString) for geom in connections.geometry)
        
        # For our grid data, each road should connect to others at intersections
        connection_counts = connections["from_public_id"].value_counts()
        
        # Horizontal roads (r0-r3) should connect to all 4 vertical roads
        for i in range(4):
            assert connection_counts.get(f"r{i}", 0) >= 4
            
        # Vertical roads (r4-r7) should connect to all 4 horizontal roads
        for i in range(4, 8):
            assert connection_counts.get(f"r{i}", 0) >= 4
    
    def test_create_public_to_public_no_duplicates(self, grid_data):
        """Test that connections aren't duplicated"""
        roads = grid_data["roads"]
        
        connections = create_public_to_public(roads, public_id_col="id")
        
        # Create a set of unique connections (regardless of direction)
        unique_connections = set()
        for _, row in connections.iterrows():
            conn = tuple(sorted([row["from_public_id"], row["to_public_id"]]))
            unique_connections.add(conn)
            
        # Number of unique connections should equal length of dataframe
        # (no duplicates in opposite directions)
        assert len(unique_connections) == len(connections)
    
    def test_create_public_to_public_empty(self):
        """Test with empty input"""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")
        
        connections = create_public_to_public(empty_gdf)
        assert connections.empty


class TestHelperFunctions:
    def test_extract_dual_graph_nodes(self):
        """Test extraction of nodes from dual graph"""
        # Create a simple dual graph
        G = nx.Graph()
        G.add_node((0, 0), id="r1")
        G.add_node((1, 1), id="r2")
        G.add_edge((0, 0), (1, 1), weight=1)
        
        # Extract nodes
        nodes_gdf = _extract_dual_graph_nodes(G, "id", "EPSG:27700")
        
        # Check structure
        assert len(nodes_gdf) == 2
        assert set(nodes_gdf.index) == {"r1", "r2"}
        assert all(isinstance(geom, Point) for geom in nodes_gdf.geometry)
    
    def test_extract_node_connections(self):
        """Test extraction of connections from dual graph"""
        # Create a simple dual graph
        G = nx.Graph()
        G.add_node((0, 0), id="r1")
        G.add_node((1, 1), id="r2")
        G.add_node((2, 0), id="r3")
        G.add_edge((0, 0), (1, 1))
        G.add_edge((1, 1), (2, 0))
        
        # Extract connections
        connections = _extract_node_connections(G, "id")
        
        # Check structure
        assert len(connections) == 3
        assert connections["r1"] == ["r2"]
        assert set(connections["r2"]) == {"r1", "r3"}
        assert connections["r3"] == ["r2"]
    
    def test_find_additional_connections(self):
        """Test finding additional connections based on proximity"""
        # Create two LineStrings that have endpoints close to each other
        line1 = LineString([(0, 0), (10, 0)])
        line2 = LineString([(10.05, 0), (20, 0)])  # Gap of 0.05 units
        
        gdf = gpd.GeoDataFrame(
            {"id": ["l1", "l2"], "geometry": [line1, line2]},
            crs="EPSG:27700"
        )
        
        # With small tolerance - no connections
        small_connections = _find_additional_connections(gdf, "id", 0.01)
        assert not small_connections.get("l1", [])
        assert not small_connections.get("l2", [])
        
        # With larger tolerance - should connect
        large_connections = _find_additional_connections(gdf, "id", 0.1)
        assert "l2" in large_connections.get("l1", [])
        assert "l1" in large_connections.get("l2", [])
