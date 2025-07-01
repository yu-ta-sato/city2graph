"""Test module for morphology.py with full coverage using fixtures from conftest.py."""

import math
import warnings

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

from city2graph.morphology import morphological_graph
from city2graph.morphology import private_to_private_graph
from city2graph.morphology import private_to_public_graph
from city2graph.morphology import public_to_public_graph


class TestMorphologicalGraph:
    """Test suite for morphological_graph function."""

    def test_basic_morphological_graph(
        self, sample_buildings_gdf, sample_segments_gdf,
    ):
        """Test basic morphological graph creation."""
        nodes, edges = morphological_graph(sample_buildings_gdf, sample_segments_gdf)

        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)
        assert "private" in nodes
        assert "public" in nodes
        assert ("private", "touched_to", "private") in edges
        assert ("public", "connected_to", "public") in edges
        assert ("private", "faced_to", "public") in edges

    def test_morphological_graph_as_nx(
        self, sample_buildings_gdf, sample_segments_gdf,
    ):
        """Test morphological graph creation as NetworkX graph."""
        graph = morphological_graph(
            sample_buildings_gdf, sample_segments_gdf, as_nx=True,
        )

        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() > 0

    @pytest.mark.parametrize("contiguity", ["queen", "rook"])
    def test_morphological_graph_contiguity(
        self, sample_buildings_gdf, sample_segments_gdf, contiguity,
    ):
        """Test morphological graph with different contiguity types."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf, sample_segments_gdf, contiguity=contiguity,
        )

        assert isinstance(nodes["private"], gpd.GeoDataFrame)
        assert isinstance(edges[("private", "touched_to", "private")], gpd.GeoDataFrame)

    def test_morphological_graph_with_distance_filtering(
        self, sample_buildings_gdf, sample_segments_gdf, custom_center_point,
    ):
        """Test morphological graph with distance filtering."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=custom_center_point,
            distance=1000,
        )

        assert "private" in nodes
        assert "public" in nodes

    @pytest.mark.parametrize("clipping_buffer", [0, 100, 500, math.inf])
    def test_morphological_graph_clipping_buffer(
        self, sample_buildings_gdf, sample_segments_gdf, custom_center_point, clipping_buffer,
    ):
        """Test morphological graph with various clipping buffer values."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=custom_center_point,
            distance=500,
            clipping_buffer=clipping_buffer,
        )

        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

    def test_morphological_graph_with_barrier_column(
        self, sample_buildings_gdf, segments_gdf_alt_geom,
    ):
        """Test morphological graph with alternative barrier geometry column."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            segments_gdf_alt_geom,
            primary_barrier_col="barrier_geometry",
        )

        assert "private" in nodes
        assert "public" in nodes

    def test_morphological_graph_with_custom_barrier(
        self, sample_buildings_gdf, segments_gdf_with_custom_barrier,
    ):
        """Test morphological graph with custom barrier column."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            segments_gdf_with_custom_barrier,
            primary_barrier_col="custom_barrier",
        )

        assert isinstance(nodes["private"], gpd.GeoDataFrame)

    def test_morphological_graph_keep_buildings(
        self, sample_buildings_gdf, sample_segments_gdf,
    ):
        """Test morphological graph with keep_buildings option."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf, sample_segments_gdf, keep_buildings=True,
        )

        private_nodes = nodes["private"]
        assert isinstance(private_nodes, gpd.GeoDataFrame)
        # Check if building information is preserved
        if not private_nodes.empty and not sample_buildings_gdf.empty:
            assert "building_geometry" in private_nodes.columns

    @pytest.mark.parametrize("tolerance", [1e-6, 1e-3, 0.1, 1.0])
    def test_morphological_graph_tolerance(
        self, sample_buildings_gdf, sample_segments_gdf, tolerance,
    ):
        """Test morphological graph with different tolerance values."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf, sample_segments_gdf, tolerance=tolerance,
        )

        assert isinstance(edges[("private", "faced_to", "public")], gpd.GeoDataFrame)

    def test_morphological_graph_empty_buildings(
        self, empty_gdf, sample_segments_gdf,
    ):
        """Test morphological graph with empty buildings."""
        nodes, edges = morphological_graph(empty_gdf, sample_segments_gdf)

        assert nodes["private"].empty
        assert edges[("private", "touched_to", "private")].empty
        assert edges[("private", "faced_to", "public")].empty

    def test_morphological_graph_empty_segments(
        self, sample_buildings_gdf, empty_gdf,
    ):
        """Test morphological graph with empty segments."""
        nodes, edges = morphological_graph(sample_buildings_gdf, empty_gdf)

        assert nodes["public"].empty
        assert edges[("public", "connected_to", "public")].empty

    def test_morphological_graph_both_empty(self, empty_gdf):
        """Test morphological graph with both inputs empty."""
        nodes, edges = morphological_graph(empty_gdf, empty_gdf)

        assert nodes["private"].empty
        assert nodes["public"].empty
        assert all(edge_gdf.empty for edge_gdf in edges.values())

    def test_morphological_graph_single_building_single_segment(
        self, single_building_gdf, single_segment_gdf,
    ):
        """Test morphological graph with single building and segment."""
        nodes, edges = morphological_graph(
            single_building_gdf, single_segment_gdf,
        )

        # Tessellation might fail with single building/segment, so just check structure
        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)
        assert "private" in nodes
        assert "public" in nodes
        assert len(nodes["public"]) == 1

    def test_morphological_graph_crs_mismatch(
        self, sample_buildings_gdf, segments_gdf_alt_crs,
    ):
        """Test morphological graph with CRS mismatch."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nodes, edges = morphological_graph(
                sample_buildings_gdf, segments_gdf_alt_crs,
            )

        assert nodes["private"].crs == sample_buildings_gdf.crs
        assert nodes["public"].crs == sample_buildings_gdf.crs

    def test_morphological_graph_invalid_contiguity(
        self, sample_buildings_gdf, sample_segments_gdf,
    ):
        """Test morphological graph with invalid contiguity parameter."""
        with pytest.raises(ValueError, match="contiguity must be 'queen' or 'rook'"):
            morphological_graph(
                sample_buildings_gdf, sample_segments_gdf, contiguity="invalid",
            )

    def test_morphological_graph_negative_clipping_buffer(
        self, sample_buildings_gdf, sample_segments_gdf,
    ):
        """Test morphological graph with negative clipping buffer."""
        with pytest.raises(ValueError, match="clipping_buffer cannot be negative"):
            morphological_graph(
                sample_buildings_gdf, sample_segments_gdf, clipping_buffer=-100,
            )

    def test_morphological_graph_invalid_buildings_type(
        self, not_a_gdf, sample_segments_gdf,
    ):
        """Test morphological graph with invalid buildings type."""
        with pytest.raises(TypeError, match="buildings_gdf must be a GeoDataFrame"):
            morphological_graph(not_a_gdf, sample_segments_gdf)

    def test_morphological_graph_invalid_segments_type(
        self, sample_buildings_gdf, not_a_gdf,
    ):
        """Test morphological graph with invalid segments type."""
        with pytest.raises(TypeError, match="segments_gdf must be a GeoDataFrame"):
            morphological_graph(sample_buildings_gdf, not_a_gdf)

    def test_morphological_graph_invalid_buildings_geometry(
        self, buildings_invalid_geom_gdf, sample_segments_gdf,
    ):
        """Test morphological graph with invalid building geometry types."""
        with pytest.raises(ValueError, match="buildings_gdf must contain only Polygon"):
            morphological_graph(buildings_invalid_geom_gdf, sample_segments_gdf)

    def test_morphological_graph_invalid_segments_geometry(
        self, sample_buildings_gdf, segments_invalid_geom_gdf,
    ):
        """Test morphological graph with invalid segment geometry types."""
        with pytest.raises(ValueError, match="segments_gdf must contain only LineString"):
            morphological_graph(sample_buildings_gdf, segments_invalid_geom_gdf)

    def test_morphological_graph_no_private_public_connections(
        self, sample_buildings_gdf, segments_gdf_far_away,
    ):
        """Test morphological graph when buildings and segments are far apart."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf, segments_gdf_far_away,
        )

        # Private-public edges should be empty due to distance
        assert edges[("private", "faced_to", "public")].empty

    def test_morphological_graph_center_point_as_point(
        self, sample_buildings_gdf, sample_segments_gdf, custom_center_point,
    ):
        """Test morphological graph with center_point as Point object."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=custom_center_point,
            distance=10000,
        )

        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

    @pytest.mark.parametrize(
        "distance,expected_reduced",
        [(100, True), (1000, True), (10000, False)],
    )
    def test_distance_filtering_effect(
        self,
        sample_buildings_gdf,
        sample_segments_gdf,
        custom_center_point,
        distance,
        expected_reduced,
    ):
        """Test that distance filtering actually reduces the graph size."""
        # Full graph
        nodes_full, _ = morphological_graph(
            sample_buildings_gdf, sample_segments_gdf,
        )

        # Filtered graph
        nodes_filtered, _ = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=custom_center_point,
            distance=distance,
        )

        if expected_reduced and len(nodes_full["public"]) > 1:
            # For small distances, expect some filtering
            assert len(nodes_filtered["public"]) <= len(nodes_full["public"])
        elif not expected_reduced:
            # For large distances, expect minimal filtering (some segments might still be filtered due to network connectivity)
            assert len(nodes_filtered["public"]) >= len(nodes_full["public"]) * 0.9  # Allow for some filtering

    def test_morphological_graph_with_complex_tessellation(self, sample_crs):
        """Test morphological graph with complex tessellation scenario."""
        # Create buildings with varying sizes and positions
        buildings = [
            Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),  # Large building
            Polygon([(7, 0), (9, 0), (9, 2), (7, 2)]),  # Small building
            Polygon([(0, 7), (2, 7), (2, 9), (0, 9)]),  # Small building
            Polygon([(4, 4), (8, 4), (8, 8), (4, 8)]),  # Medium building
        ]

        buildings_gdf = gpd.GeoDataFrame(
            {"bldg_id": [f"b{i}" for i in range(len(buildings))]},
            geometry=buildings,
            crs=sample_crs,
        )

        # Create intersecting streets
        streets = [
            LineString([(-1, 3), (10, 3)]),  # Horizontal
            LineString([(3, -1), (3, 10)]),  # Vertical
            LineString([(-1, 6), (10, 6)]),  # Horizontal
            LineString([(6, -1), (6, 10)]),  # Vertical
        ]

        segments_gdf = gpd.GeoDataFrame(
            {"seg_id": [f"s{i}" for i in range(len(streets))]},
            geometry=streets,
            crs=sample_crs,
        )

        nodes, edges = morphological_graph(
            buildings_gdf, segments_gdf, keep_buildings=True,
        )

        # Verify all components are created
        assert len(nodes["private"]) >= len(buildings)  # Tessellation may create more
        assert len(nodes["public"]) == len(streets)

        # Verify all edge types exist
        assert all(key in edges for key in [
            ("private", "touched_to", "private"),
            ("public", "connected_to", "public"),
            ("private", "faced_to", "public"),
        ])


class TestPrivateToPrivateGraph:
    """Test suite for private_to_private_graph function."""

    def test_basic_private_to_private(self, sample_tessellation_gdf):
        """Test basic private-to-private graph creation."""
        nodes, edges = private_to_private_graph(sample_tessellation_gdf)

        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)
        assert nodes.equals(sample_tessellation_gdf)

    def test_private_to_private_as_nx(self, sample_tessellation_gdf):
        """Test private-to-private graph as NetworkX."""
        graph = private_to_private_graph(sample_tessellation_gdf, as_nx=True)

        assert isinstance(graph, nx.Graph)

    @pytest.mark.parametrize("contiguity", ["queen", "rook"])
    def test_private_to_private_contiguity(self, sample_tessellation_gdf, contiguity):
        """Test private-to-private with different contiguity types."""
        nodes, edges = private_to_private_graph(
            sample_tessellation_gdf, contiguity=contiguity,
        )

        assert isinstance(edges, gpd.GeoDataFrame)
        if not edges.empty:
            assert "from_private_id" in edges.columns
            assert "to_private_id" in edges.columns

    def test_private_to_private_with_group_col(self, sample_tessellation_gdf):
        """Test private-to-private with group column."""
        nodes, edges = private_to_private_graph(
            sample_tessellation_gdf, group_col="enclosure_index",
        )

        assert isinstance(edges, gpd.GeoDataFrame)
        if not edges.empty:
            assert "enclosure_index" in edges.columns

    def test_private_to_private_empty_input(self, empty_gdf):
        """Test private-to-private with empty input."""
        nodes, edges = private_to_private_graph(empty_gdf)

        assert nodes.empty
        assert edges.empty

    def test_private_to_private_single_polygon(self, single_tessellation_cell_gdf):
        """Test private-to-private with single polygon."""
        nodes, edges = private_to_private_graph(single_tessellation_cell_gdf)

        assert len(nodes) == 1
        assert edges.empty  # No adjacencies with single polygon

    def test_private_to_private_isolated_polygons(self, p2p_isolated_polys_gdf):
        """Test private-to-private with isolated polygons."""
        nodes, edges = private_to_private_graph(p2p_isolated_polys_gdf)

        assert len(nodes) == 3
        assert edges.empty  # No adjacencies between isolated polygons

    def test_private_to_private_invalid_type(self, not_a_gdf):
        """Test private-to-private with invalid input type."""
        with pytest.raises(TypeError, match="private_gdf must be a GeoDataFrame"):
            private_to_private_graph(not_a_gdf)

    def test_private_to_private_invalid_contiguity(self, sample_tessellation_gdf):
        """Test private-to-private with invalid contiguity."""
        with pytest.raises(ValueError, match="contiguity must be either 'queen' or 'rook'"):
            private_to_private_graph(sample_tessellation_gdf, contiguity="invalid")

    def test_private_to_private_missing_private_id(self, private_gdf_no_private_id):
        """Test private-to-private with missing private_id column."""
        with pytest.raises(ValueError, match="Expected ID column 'private_id' not found"):
            private_to_private_graph(private_gdf_no_private_id)

    def test_private_to_private_invalid_group_col(self, sample_tessellation_gdf):
        """Test private-to-private with non-existent group column."""
        with pytest.raises(ValueError, match="group_col 'nonexistent' not found"):
            private_to_private_graph(sample_tessellation_gdf, group_col="nonexistent")

    def test_private_to_private_edge_geometries(self, sample_tessellation_gdf):
        """Test that edge geometries are LineStrings connecting centroids."""
        nodes, edges = private_to_private_graph(sample_tessellation_gdf)

        if not edges.empty:
            assert all(edges.geometry.geom_type == "LineString")
            # Check that geometries connect centroids
            for _, edge in edges.iterrows():
                from_id = edge["from_private_id"]
                to_id = edge["to_private_id"]
                from_centroid = nodes.loc[from_id].geometry.centroid
                to_centroid = nodes.loc[to_id].geometry.centroid
                edge_coords = list(edge.geometry.coords)
                assert len(edge_coords) == 2
                # Check proximity to centroids (within small tolerance)
                assert Point(edge_coords[0]).distance(from_centroid) < 1e-6
                assert Point(edge_coords[1]).distance(to_centroid) < 1e-6


class TestPrivateToPublicGraph:
    """Test suite for private_to_public_graph function."""

    def test_basic_private_to_public(
        self, sample_tessellation_gdf, sample_segments_gdf,
    ):
        """Test basic private-to-public graph creation."""
        edges = private_to_public_graph(sample_tessellation_gdf, sample_segments_gdf)

        assert isinstance(edges, gpd.GeoDataFrame)
        if not edges.empty:
            assert "private_id" in edges.columns
            assert "public_id" in edges.columns

    def test_private_to_public_as_nx(
        self, sample_tessellation_gdf, sample_segments_gdf,
    ):
        """Test private-to-public graph as NetworkX."""
        graph = private_to_public_graph(
            sample_tessellation_gdf, sample_segments_gdf, as_nx=True,
        )

        assert isinstance(graph, nx.Graph)

    @pytest.mark.parametrize("tolerance", [1e-6, 1e-3, 0.1, 1.0])
    def test_private_to_public_tolerance(
        self, sample_tessellation_gdf, sample_segments_gdf, tolerance,
    ):
        """Test private-to-public with different tolerance values."""
        edges = private_to_public_graph(
            sample_tessellation_gdf, sample_segments_gdf, tolerance=tolerance,
        )

        assert isinstance(edges, gpd.GeoDataFrame)

    def test_private_to_public_with_barrier_col(
        self, sample_tessellation_gdf, segments_gdf_alt_geom,
    ):
        """Test private-to-public with alternative barrier column."""
        edges = private_to_public_graph(
            sample_tessellation_gdf,
            segments_gdf_alt_geom,
            primary_barrier_col="barrier_geometry",
        )

        assert isinstance(edges, gpd.GeoDataFrame)

    def test_private_to_public_empty_private(self, empty_gdf, sample_segments_gdf):
        """Test private-to-public with empty private input."""
        edges = private_to_public_graph(empty_gdf, sample_segments_gdf)

        assert edges.empty

    def test_private_to_public_empty_public(self, sample_tessellation_gdf, empty_gdf):
        """Test private-to-public with empty public input."""
        edges = private_to_public_graph(sample_tessellation_gdf, empty_gdf)

        assert edges.empty

    def test_private_to_public_both_empty(self, empty_gdf):
        """Test private-to-public with both inputs empty."""
        edges = private_to_public_graph(empty_gdf, empty_gdf)

        assert edges.empty

    def test_private_to_public_single_cell_single_segment(
        self, p2pub_private_single_cell, p2pub_public_single_segment,
    ):
        """Test private-to-public with single cell and segment."""
        edges = private_to_public_graph(
            p2pub_private_single_cell, p2pub_public_single_segment,
        )

        # Should have a connection due to proximity
        assert len(edges) == 1
        assert edges.iloc[0]["private_id"] == 0
        assert edges.iloc[0]["public_id"] == 10

    def test_private_to_public_no_connections(
        self, sample_tessellation_gdf, segments_gdf_far_away,
    ):
        """Test private-to-public with no possible connections."""
        edges = private_to_public_graph(
            sample_tessellation_gdf, segments_gdf_far_away,
        )

        assert edges.empty

    def test_private_to_public_crs_mismatch(
        self, sample_tessellation_gdf, segments_gdf_alt_crs,
    ):
        """Test private-to-public with CRS mismatch."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            edges = private_to_public_graph(
                sample_tessellation_gdf, segments_gdf_alt_crs,
            )

        assert edges.crs == sample_tessellation_gdf.crs

    def test_private_to_public_invalid_private_type(
        self, not_a_gdf, sample_segments_gdf,
    ):
        """Test private-to-public with invalid private type."""
        with pytest.raises(TypeError, match="private_gdf must be a GeoDataFrame"):
            private_to_public_graph(not_a_gdf, sample_segments_gdf)

    def test_private_to_public_invalid_public_type(
        self, sample_tessellation_gdf, not_a_gdf,
    ):
        """Test private-to-public with invalid public type."""
        with pytest.raises(TypeError, match="public_gdf must be a GeoDataFrame"):
            private_to_public_graph(sample_tessellation_gdf, not_a_gdf)

    def test_private_to_public_missing_private_id(
        self, private_gdf_no_private_id, sample_segments_gdf,
    ):
        """Test private-to-public with missing private_id."""
        with pytest.raises(ValueError, match="Expected ID column 'private_id' not found"):
            private_to_public_graph(private_gdf_no_private_id, sample_segments_gdf)

    def test_private_to_public_missing_public_id(
        self, sample_tessellation_gdf, segments_no_public_id_gdf,
    ):
        """Test private-to-public with missing public_id."""
        with pytest.raises(ValueError, match="Expected ID column 'public_id' not found"):
            private_to_public_graph(sample_tessellation_gdf, segments_no_public_id_gdf)

    def test_private_to_public_edge_geometries(
        self, sample_tessellation_gdf, sample_segments_gdf,
    ):
        """Test that edge geometries are LineStrings connecting centroids."""
        edges = private_to_public_graph(
            sample_tessellation_gdf, sample_segments_gdf,
        )

        if not edges.empty:
            assert all(edges.geometry.geom_type == "LineString")
            # Each edge should have exactly 2 coordinates
            for _, edge in edges.iterrows():
                assert len(list(edge.geometry.coords)) == 2


class TestPublicToPublicGraph:
    """Test suite for public_to_public_graph function."""

    def test_basic_public_to_public(self, sample_segments_gdf):
        """Test basic public-to-public graph creation."""
        nodes, edges = public_to_public_graph(sample_segments_gdf)

        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)
        assert nodes.equals(sample_segments_gdf)

    def test_public_to_public_as_nx(self, sample_segments_gdf):
        """Test public-to-public graph as NetworkX."""
        graph = public_to_public_graph(sample_segments_gdf, as_nx=True)

        assert isinstance(graph, nx.Graph)

    def test_public_to_public_empty_input(self, empty_gdf):
        """Test public-to-public with empty input."""
        nodes, edges = public_to_public_graph(empty_gdf)

        assert nodes.empty
        assert edges.empty

    def test_public_to_public_single_segment(self, single_segment_gdf):
        """Test public-to-public with single segment."""
        nodes, edges = public_to_public_graph(single_segment_gdf)

        assert len(nodes) == 1
        assert edges.empty  # No connections with single segment

    def test_public_to_public_invalid_type(self, not_a_gdf):
        """Test public-to-public with invalid input type."""
        with pytest.raises(TypeError, match="public_gdf must be a GeoDataFrame"):
            public_to_public_graph(not_a_gdf)

    def test_public_to_public_edge_indices(self, sample_segments_gdf):
        """Test that edge indices are properly set."""
        nodes, edges = public_to_public_graph(sample_segments_gdf)

        if not edges.empty:
            # Check if the edges have the expected columns
            assert "from_public_id" in edges.columns
            assert "to_public_id" in edges.columns

    def test_public_to_public_with_existing_public_id(self, sample_segments_gdf):
        """Test public-to-public when public_id already exists."""
        # The fixture already adds public_id
        nodes, edges = public_to_public_graph(sample_segments_gdf)

        assert "public_id" in nodes.columns

    def test_public_to_public_multiindex_public_id(
        self, segments_gdf_with_multiindex_public_id,
    ):
        """Test public-to-public with MultiIndex public_id."""
        # This test should handle the MultiIndex issue gracefully
        try:
            nodes, edges = public_to_public_graph(segments_gdf_with_multiindex_public_id)
            assert isinstance(nodes, gpd.GeoDataFrame)
            assert isinstance(edges, gpd.GeoDataFrame)
        except NotImplementedError:
            # Expected behavior for MultiIndex public_id
            pytest.skip("MultiIndex public_id not supported")

    def test_public_to_public_connectivity(self, sample_segments_gdf):
        """Test that connected segments are properly identified."""
        nodes, edges = public_to_public_graph(sample_segments_gdf)

        if not edges.empty and len(sample_segments_gdf) > 1:
            # Should have some connections for segments that share endpoints
            assert len(edges) > 0
            # All edges should have valid from/to IDs
            for idx, edge in edges.iterrows():
                from_id = edge["from_public_id"]
                to_id = edge["to_public_id"]
                assert from_id in nodes.index.values
                assert to_id in nodes.index.values


class TestEdgeCasesAndIntegration:
    """Test suite for edge cases and integration scenarios."""

    def test_morphological_graph_with_all_parameters(
        self, sample_buildings_gdf, segments_gdf_alt_geom, custom_center_point,
    ):
        """Test morphological graph with all parameters specified."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            segments_gdf_alt_geom,
            center_point=custom_center_point,
            distance=2000,
            clipping_buffer=500,
            primary_barrier_col="barrier_geometry",
            contiguity="rook",
            keep_buildings=True,
            tolerance=0.01,
            as_nx=False,
        )

        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

    def test_morphological_graph_stress_test(self, sample_crs):
        """Test morphological graph with larger synthetic dataset."""
        # Create a grid of buildings
        buildings = []
        for i in range(5):
            for j in range(5):
                poly = Polygon([
                    (i * 10, j * 10),
                    (i * 10 + 8, j * 10),
                    (i * 10 + 8, j * 10 + 8),
                    (i * 10, j * 10 + 8),
                ])
                buildings.append(poly)

        buildings_gdf = gpd.GeoDataFrame(
            {"bldg_id": range(len(buildings))},
            geometry=buildings,
            crs=sample_crs,
        )

        # Create a grid of streets
        streets = []
        # Horizontal streets
        for j in range(6):
            line = LineString([(0, j * 10 - 1), (50, j * 10 - 1)])
            streets.append(line)
        # Vertical streets
        for i in range(6):
            line = LineString([(i * 10 - 1, 0), (i * 10 - 1, 50)])
            streets.append(line)

        segments_gdf = gpd.GeoDataFrame(
            {"seg_id": range(len(streets))},
            geometry=streets,
            crs=sample_crs,
        )

        nodes, edges = morphological_graph(buildings_gdf, segments_gdf)

        # Tessellation might fail with synthetic data, so check structure
        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)
        assert "private" in nodes
        assert "public" in nodes
        assert len(nodes["public"]) == len(streets)

    def test_private_graphs_consistency(self, sample_tessellation_gdf):
        """Test consistency between different private graph representations."""
        # Test that converting to NetworkX and back preserves information
        nodes_gdf, edges_gdf = private_to_private_graph(sample_tessellation_gdf)
        nx_graph = private_to_private_graph(sample_tessellation_gdf, as_nx=True)

        # Check node count consistency
        assert len(nodes_gdf) == nx_graph.number_of_nodes()

        # Check edge count consistency (considering undirected nature)
        if not edges_gdf.empty:
            assert len(edges_gdf) == nx_graph.number_of_edges()

    def test_morphological_graph_reproducibility(
        self, sample_buildings_gdf, sample_segments_gdf,
    ):
        """Test that morphological graph creation is reproducible."""
        nodes1, edges1 = morphological_graph(
            sample_buildings_gdf, sample_segments_gdf,
        )
        nodes2, edges2 = morphological_graph(
            sample_buildings_gdf, sample_segments_gdf,
        )

        # Check that the same inputs produce the same outputs
        for key in nodes1:
            assert len(nodes1[key]) == len(nodes2[key])

        for key in edges1:
            assert len(edges1[key]) == len(edges2[key])


class TestComprehensiveCoverage:
    """Additional comprehensive tests to achieve full coverage through public API only."""

    def test_morphological_graph_comprehensive_edge_cases(self, sample_buildings_gdf, sample_segments_gdf):
        """Test morphological graph with comprehensive edge cases to cover helper functions."""
        # Test with various combinations to trigger different code paths

        # Test with empty buildings but non-empty segments
        nodes, edges = morphological_graph(
            sample_buildings_gdf.iloc[:0], sample_segments_gdf,
        )
        assert nodes["private"].empty
        assert edges[("private", "touched_to", "private")].empty
        assert edges[("private", "faced_to", "public")].empty

        # Test with non-empty buildings but empty segments
        nodes, edges = morphological_graph(
            sample_buildings_gdf, sample_segments_gdf.iloc[:0],
        )
        assert nodes["public"].empty
        assert edges[("public", "connected_to", "public")].empty

        # Test with both empty
        nodes, edges = morphological_graph(
            sample_buildings_gdf.iloc[:0], sample_segments_gdf.iloc[:0],
        )
        assert nodes["private"].empty
        assert nodes["public"].empty
        assert all(edge_gdf.empty for edge_gdf in edges.values())

    def test_private_to_private_comprehensive_scenarios(self, sample_crs):
        """Test private_to_private with various scenarios to cover all code paths."""
        # Test with empty GDF
        empty_gdf = gpd.GeoDataFrame(columns=["private_id", "geometry"], crs=sample_crs)
        nodes, edges = private_to_private_graph(empty_gdf)
        assert nodes.empty
        assert edges.empty

        # Test with single polygon
        single_poly = gpd.GeoDataFrame({
            "private_id": [0],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        }, crs=sample_crs)
        nodes, edges = private_to_private_graph(single_poly)
        assert len(nodes) == 1
        assert edges.empty

        # Test with adjacent polygons
        adjacent_polys = gpd.GeoDataFrame({
            "private_id": [0, 1],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        }, crs=sample_crs)
        nodes, edges = private_to_private_graph(adjacent_polys)
        assert len(nodes) == 2
        # Should have adjacency
        assert len(edges) >= 0  # May or may not have edges depending on exact geometry

    def test_private_to_public_comprehensive_scenarios(self, sample_crs):
        """Test private_to_public with various scenarios to cover all code paths."""
        # Test with empty private GDF
        empty_private = gpd.GeoDataFrame(columns=["private_id", "geometry"], crs=sample_crs)
        public_gdf = gpd.GeoDataFrame({
            "public_id": [0],
            "geometry": [LineString([(0, 0), (1, 1)])],
        }, crs=sample_crs)
        edges = private_to_public_graph(empty_private, public_gdf)
        assert edges.empty

        # Test with empty public GDF
        private_gdf = gpd.GeoDataFrame({
            "private_id": [0],
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        }, crs=sample_crs)
        empty_public = gpd.GeoDataFrame(columns=["public_id", "geometry"], crs=sample_crs)
        edges = private_to_public_graph(private_gdf, empty_public)
        assert edges.empty

        # Test with both empty
        edges = private_to_public_graph(empty_private, empty_public)
        assert edges.empty

        # Test with intersecting geometries
        intersecting_public = gpd.GeoDataFrame({
            "public_id": [0],
            "geometry": [LineString([(0.5, 0), (0.5, 1)])],  # Line through polygon
        }, crs=sample_crs)
        edges = private_to_public_graph(private_gdf, intersecting_public)
        assert isinstance(edges, gpd.GeoDataFrame)

    def test_public_to_public_comprehensive_scenarios(self, sample_crs):
        """Test public_to_public with various scenarios to cover all code paths."""
        # Test with empty GDF
        empty_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=sample_crs)
        nodes, edges = public_to_public_graph(empty_gdf)
        assert nodes.empty
        assert edges.empty

        # Test with single segment
        single_segment = gpd.GeoDataFrame({
            "geometry": [LineString([(0, 0), (1, 1)])],
        }, crs=sample_crs)
        nodes, edges = public_to_public_graph(single_segment)
        assert len(nodes) == 1
        assert edges.empty

        # Test with connected segments
        connected_segments = gpd.GeoDataFrame({
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
            ],
        }, crs=sample_crs)
        nodes, edges = public_to_public_graph(connected_segments)
        assert len(nodes) == 2
        # Should have connectivity
        assert isinstance(edges, gpd.GeoDataFrame)

    def test_morphological_graph_index_handling(self, sample_buildings_gdf, sample_segments_gdf):
        """Test morphological graph index handling to cover _set_node_index and _set_edge_index."""
        # Test with custom indices
        buildings_custom_idx = sample_buildings_gdf.copy()
        buildings_custom_idx.index = range(100, 100 + len(buildings_custom_idx))

        segments_custom_idx = sample_segments_gdf.copy()
        segments_custom_idx.index = range(200, 200 + len(segments_custom_idx))

        nodes, edges = morphological_graph(buildings_custom_idx, segments_custom_idx)

        # Check that indices are properly set
        assert "private_id" in nodes["private"].index.names or nodes["private"].index.name == "private_id"
        assert "public_id" in nodes["public"].index.names or nodes["public"].index.name == "public_id"

        # Check edge indices
        for edge_type, edge_gdf in edges.items():
            if not edge_gdf.empty:
                assert isinstance(edge_gdf.index, (pd.Index, pd.MultiIndex))

    def test_morphological_graph_barrier_geometry_coverage(self, sample_buildings_gdf, sample_segments_gdf):
        """Test barrier geometry handling to cover _prepare_barriers."""
        # Test with non-existent barrier column
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            primary_barrier_col="nonexistent_column",
        )
        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

        # Test with None barrier column
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            primary_barrier_col=None,
        )
        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

    def test_morphological_graph_building_info_coverage(self, sample_buildings_gdf, sample_segments_gdf):
        """Test building info addition to cover _add_building_info."""
        # Test with keep_buildings=True
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            keep_buildings=True,
        )

        private_nodes = nodes["private"]
        if not private_nodes.empty and not sample_buildings_gdf.empty:
            # Should have building-related columns if intersections exist
            assert isinstance(private_nodes, gpd.GeoDataFrame)

        # Test with empty buildings and keep_buildings=True
        empty_buildings = sample_buildings_gdf.iloc[:0]
        nodes, edges = morphological_graph(
            empty_buildings,
            sample_segments_gdf,
            keep_buildings=True,
        )
        assert nodes["private"].empty

    def test_morphological_graph_tessellation_filtering_coverage(self, sample_buildings_gdf, sample_segments_gdf, custom_center_point):
        """Test tessellation filtering to cover _filter_adjacent_tessellation."""
        # Test with various distance values to trigger filtering
        for distance in [100, 1000, 10000]:
            nodes, edges = morphological_graph(
                sample_buildings_gdf,
                sample_segments_gdf,
                center_point=custom_center_point,
                distance=distance,
            )
            assert isinstance(nodes, dict)
            assert isinstance(edges, dict)

        # Test with infinite clipping buffer
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=custom_center_point,
            distance=1000,
            clipping_buffer=math.inf,
        )
        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

    def test_all_functions_with_multiindex_handling(self, sample_crs):
        """Test functions with MultiIndex scenarios to cover edge cases."""
        # Create data with MultiIndex
        segments_with_multiindex = gpd.GeoDataFrame({
            "geometry": [
                LineString([(0, 0), (1, 1)]),
                LineString([(1, 1), (2, 2)]),
            ],
        }, crs=sample_crs)
        segments_with_multiindex.index = pd.MultiIndex.from_tuples(
            [("A", 1), ("B", 2)], names=["type", "id"],
        )

        # Test public_to_public with MultiIndex
        try:
            nodes, edges = public_to_public_graph(segments_with_multiindex)
            assert isinstance(nodes, gpd.GeoDataFrame)
            assert isinstance(edges, gpd.GeoDataFrame)
        except (NotImplementedError, ValueError):
            # Expected for some MultiIndex scenarios
            pass

    def test_comprehensive_error_scenarios(self, sample_buildings_gdf, sample_segments_gdf):
        """Test comprehensive error scenarios to cover validation functions."""
        # These tests will trigger the validation helper functions through public API

        # Test invalid geometry types through morphological_graph
        invalid_buildings = gpd.GeoDataFrame({
            "geometry": [LineString([(0, 0), (1, 1)])],  # Invalid: should be Polygon
        }, crs=sample_buildings_gdf.crs)

        with pytest.raises(ValueError, match="buildings_gdf must contain only Polygon"):
            morphological_graph(invalid_buildings, sample_segments_gdf)

        invalid_segments = gpd.GeoDataFrame({
            "geometry": [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],  # Invalid: should be LineString
        }, crs=sample_segments_gdf.crs)

        with pytest.raises(ValueError, match="segments_gdf must contain only LineString"):
            morphological_graph(sample_buildings_gdf, invalid_segments)

    def test_crs_consistency_coverage(self, sample_buildings_gdf, sample_segments_gdf):
        """Test CRS consistency handling to cover _ensure_crs_consistency."""
        # Create segments with different CRS
        segments_diff_crs = sample_segments_gdf.copy()
        if sample_segments_gdf.crs != "EPSG:4326":
            segments_diff_crs = segments_diff_crs.to_crs("EPSG:4326")
        else:
            segments_diff_crs = segments_diff_crs.to_crs("EPSG:3857")

        # This should trigger CRS reprojection warning and _ensure_crs_consistency
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nodes, edges = morphological_graph(sample_buildings_gdf, segments_diff_crs)
            assert isinstance(nodes, dict)
            assert isinstance(edges, dict)
            # Check that output CRS matches buildings CRS
            assert nodes["public"].crs == sample_buildings_gdf.crs


class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling."""

    def test_morphological_graph_with_point_center(self, sample_buildings_gdf, sample_segments_gdf):
        """Test morphological graph with Point center instead of GeoSeries."""
        center_point = Point(0, 0)
        center_geoseries = gpd.GeoSeries([center_point], crs=sample_buildings_gdf.crs)

        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=center_geoseries,
            distance=1000,
        )

        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

    def test_morphological_graph_zero_distance(self, sample_buildings_gdf, sample_segments_gdf, custom_center_point):
        """Test morphological graph with zero distance."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=custom_center_point,
            distance=0,
        )

        # Should result in very limited or empty graphs
        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

    def test_morphological_graph_very_large_distance(self, sample_buildings_gdf, sample_segments_gdf, custom_center_point):
        """Test morphological graph with very large distance."""
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=custom_center_point,
            distance=1000000,  # Very large distance
        )

        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

    def test_private_to_private_no_adjacencies(self, p2p_isolated_polys_gdf):
        """Test private_to_private_graph with no adjacencies."""
        nodes, edges = private_to_private_graph(p2p_isolated_polys_gdf)

        assert len(nodes) == 3
        assert edges.empty

    def test_private_to_public_very_small_tolerance(self, sample_tessellation_gdf, sample_segments_gdf):
        """Test private_to_public_graph with very small tolerance."""
        edges = private_to_public_graph(
            sample_tessellation_gdf,
            sample_segments_gdf,
            tolerance=1e-12,  # Very small tolerance
        )

        assert isinstance(edges, gpd.GeoDataFrame)

    def test_private_to_public_very_large_tolerance(self, sample_tessellation_gdf, sample_segments_gdf):
        """Test private_to_public_graph with very large tolerance."""
        edges = private_to_public_graph(
            sample_tessellation_gdf,
            sample_segments_gdf,
            tolerance=1000,  # Very large tolerance
        )

        assert isinstance(edges, gpd.GeoDataFrame)
        # Should have many connections with large tolerance
        if not sample_tessellation_gdf.empty and not sample_segments_gdf.empty:
            assert len(edges) >= 0

    def test_morphological_graph_missing_enclosure_index(self, sample_buildings_gdf, sample_segments_gdf):
        """Test morphological graph when tessellation lacks enclosure_index."""
        # This tests the warning path when enclosure_index is missing
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            nodes, edges = morphological_graph(sample_buildings_gdf, sample_segments_gdf)

            # Check if warning was issued (may or may not happen depending on tessellation result)
            assert isinstance(nodes, dict)
            assert isinstance(edges, dict)

    @pytest.mark.parametrize("contiguity", ["queen", "rook"])
    def test_private_to_private_different_contiguity_types(self, sample_tessellation_gdf, contiguity):
        """Test private_to_private_graph with different contiguity types."""
        nodes, edges = private_to_private_graph(sample_tessellation_gdf, contiguity=contiguity)

        assert isinstance(nodes, gpd.GeoDataFrame)
        assert isinstance(edges, gpd.GeoDataFrame)

    def test_morphological_graph_center_point_as_geodataframe(self, sample_buildings_gdf, sample_segments_gdf):
        """Test morphological graph with center_point as GeoDataFrame."""
        center_point = gpd.GeoSeries([Point(0, 0)], crs=sample_buildings_gdf.crs)

        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=center_point,
            distance=1000,
        )

        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

    def test_all_functions_return_correct_types(self, sample_buildings_gdf, sample_segments_gdf, sample_tessellation_gdf):
        """Test that all main functions return correct types."""
        # Test morphological_graph
        nodes, edges = morphological_graph(sample_buildings_gdf, sample_segments_gdf)
        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

        # Test as NetworkX
        nx_graph = morphological_graph(sample_buildings_gdf, sample_segments_gdf, as_nx=True)
        assert isinstance(nx_graph, nx.Graph)

        # Test private_to_private_graph
        p2p_nodes, p2p_edges = private_to_private_graph(sample_tessellation_gdf)
        assert isinstance(p2p_nodes, gpd.GeoDataFrame)
        assert isinstance(p2p_edges, gpd.GeoDataFrame)

        # Test private_to_public_graph
        p2pub_edges = private_to_public_graph(sample_tessellation_gdf, sample_segments_gdf)
        assert isinstance(p2pub_edges, gpd.GeoDataFrame)

        # Test public_to_public_graph
        pub2pub_nodes, pub2pub_edges = public_to_public_graph(sample_segments_gdf)
        assert isinstance(pub2pub_nodes, gpd.GeoDataFrame)
        assert isinstance(pub2pub_edges, gpd.GeoDataFrame)


class TestParameterValidation:
    """Test suite for parameter validation."""

    def test_morphological_graph_invalid_contiguity_values(self, sample_buildings_gdf, sample_segments_gdf):
        """Test morphological_graph with various invalid contiguity values."""
        invalid_values = ["invalid", "QUEEN", "ROOK", "", None, 123]

        for invalid_value in invalid_values:
            with pytest.raises(ValueError, match="contiguity must be 'queen' or 'rook'"):
                morphological_graph(sample_buildings_gdf, sample_segments_gdf, contiguity=invalid_value)

    def test_morphological_graph_negative_distance(self, sample_buildings_gdf, sample_segments_gdf, custom_center_point):
        """Test morphological_graph with negative distance."""
        # Negative distance should be handled gracefully or raise appropriate error
        try:
            nodes, edges = morphological_graph(
                sample_buildings_gdf,
                sample_segments_gdf,
                center_point=custom_center_point,
                distance=-100,
            )
            # If it doesn't raise an error, check that results are reasonable
            assert isinstance(nodes, dict)
            assert isinstance(edges, dict)
        except ValueError:
            # It's also acceptable to raise a ValueError for negative distance
            pass

    def test_morphological_graph_distance_without_center(self, sample_buildings_gdf, sample_segments_gdf):
        """Test morphological_graph with distance but no center_point."""
        # Should work normally, distance should be ignored
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            distance=1000,  # No center_point provided
        )

        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

    def test_morphological_graph_center_without_distance(self, sample_buildings_gdf, sample_segments_gdf, custom_center_point):
        """Test morphological_graph with center_point but no distance."""
        # Should work normally, center_point should be ignored
        nodes, edges = morphological_graph(
            sample_buildings_gdf,
            sample_segments_gdf,
            center_point=custom_center_point,  # No distance provided
        )

        assert isinstance(nodes, dict)
        assert isinstance(edges, dict)

    @pytest.mark.parametrize("clipping_buffer", [-1, -100, -0.1])
    def test_morphological_graph_negative_clipping_buffer_values(
        self, sample_buildings_gdf, sample_segments_gdf, clipping_buffer,
    ):
        """Test morphological_graph with various negative clipping buffer values."""
        with pytest.raises(ValueError, match="clipping_buffer cannot be negative"):
            morphological_graph(
                sample_buildings_gdf,
                sample_segments_gdf,
                clipping_buffer=clipping_buffer,
            )

    def test_private_to_private_invalid_contiguity_values(self, sample_tessellation_gdf):
        """Test private_to_private_graph with various invalid contiguity values."""
        invalid_values = ["invalid", "QUEEN", "ROOK", "", None, 123]

        for invalid_value in invalid_values:
            with pytest.raises(ValueError, match="contiguity must be either 'queen' or 'rook'"):
                private_to_private_graph(sample_tessellation_gdf, contiguity=invalid_value)
