import pytest
import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import dgl
import torch
from shapely.geometry import LineString, Polygon, Point, MultiLineString
from city2graph.graph import (
    create_homogeneous_graph,
    create_heterogeneous_graph,
    create_morphological_graph
)

@pytest.fixture
def graph_test_data():
    """Create test data for graph creation tests"""
    # Create private spaces (buildings/tessellations)
    private_polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        Polygon([(0, 2), (1, 2), (1, 3), (0, 3)]),
        Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    ]
    
    private_gdf = gpd.GeoDataFrame(
        {
            'tess_id': ['t1', 't2', 't3', 't4'],
            'test1': [0.1, 0.2, 0.3, 0.4],
            'test2': [0.5, 0.6, 0.7, 0.8],
            'test3': [0.9, 1.0, 1.1, 1.2],
            'enclosure_index': [0, 0, 1, 1],
            'geometry': private_polygons
        },
        crs="EPSG:27700"
    )
    
    # Create public spaces (roads)
    public_lines = [
        LineString([(0, 1.5), (3, 1.5)]),  # Horizontal road
        LineString([(1.5, 0), (1.5, 3)])   # Vertical road
    ]
    
    public_gdf = gpd.GeoDataFrame(
        {
            'id': ['r1', 'r2'],
            'test1': [0.1, 0.2],
            'test2': [0.3, 0.4],
            'subtype': ['road', 'road'],
            'class': ['residential', 'residential'],
            'road_flags': [None, None],
            'geometry': public_lines,
            'barrier_geometry': public_lines  # Same as geometry for simplicity
        },
        crs="EPSG:27700"
    )
    
    # Create private-private connections
    priv_priv_lines = [
        LineString([(0.5, 1), (2.5, 1)]),   # t1 to t2
        LineString([(0.5, 2.5), (2.5, 2.5)]) # t3 to t4
    ]
    
    private_to_private_gdf = gpd.GeoDataFrame(
        {
            'from_private_id': ['t1', 't3'],
            'to_private_id': ['t2', 't4'],
            'enclosure_index': [0, 1],
            'geometry': priv_priv_lines
        },
        crs="EPSG:27700"
    )
    
    # Create public-public connections
    pub_pub_lines = [
        LineString([(1.5, 1.5), (1.5, 1.5)])  # r1 to r2 (intersection point)
    ]
    
    public_to_public_gdf = gpd.GeoDataFrame(
        {
            'from_public_id': ['r1'],
            'to_public_id': ['r2'],
            'geometry': pub_pub_lines
        },
        crs="EPSG:27700"
    )
    
    # Create private-public connections
    priv_pub_lines = [
        LineString([(0.5, 0.5), (0.5, 1.5)]),  # t1 to r1
        LineString([(2.5, 0.5), (2.5, 1.5)]),  # t2 to r1
        LineString([(0.5, 2.5), (0.5, 1.5)]),  # t3 to r1
        LineString([(2.5, 2.5), (2.5, 1.5)])   # t4 to r1
    ]
    
    private_to_public_gdf = gpd.GeoDataFrame(
        {
            'private_id': ['t1', 't2', 't3', 't4'],
            'public_id': ['r1', 'r1', 'r1', 'r1'],
            'geometry': priv_pub_lines
        },
        crs="EPSG:27700"
    )
    
    return {
        "private": private_gdf,
        "public": public_gdf,
        "private_to_private": private_to_private_gdf,
        "public_to_public": public_to_public_gdf,
        "private_to_public": private_to_public_gdf
    }


class TestHomogeneousGraph:
    def test_create_homogeneous_public_graph(self, graph_test_data):
        """Test creating a homogeneous graph of public spaces"""
        public_gdf = graph_test_data["public"]
        public_to_public_gdf = graph_test_data["public_to_public"]
        
        # Create graph from public spaces and their connections
        public_graph = create_homogeneous_graph(
            nodes_gdf=public_gdf,
            edges_gdf=public_to_public_gdf,
            id_col="id",
            attribute_cols=["test1", "test2"],
            source_col="from_public_id",
            target_col="to_public_id"
        )
        
        # Check graph structure
        assert isinstance(public_graph, dgl.DGLGraph)
        assert public_graph.num_nodes() == 2  # r1 and r2
        assert public_graph.num_edges() == 1  # One connection between r1 and r2
        
        # Check node features
        assert public_graph.ndata["test1"].shape == (2, 1)
        assert public_graph.ndata["test2"].shape == (2, 1)
        
        # Check that feature values were transferred correctly
        node_ids = public_gdf['id'].tolist()
        for i, node_id in enumerate(node_ids):
            node_idx = i  # In this simple case, indices match positions
            assert float(public_graph.ndata["test1"][node_idx]) == public_gdf.loc[public_gdf['id'] == node_id, "test1"].values[0]
            assert float(public_graph.ndata["test2"][node_idx]) == public_gdf.loc[public_gdf['id'] == node_id, "test2"].values[0]
    
    def test_create_homogeneous_private_graph(self, graph_test_data):
        """Test creating a homogeneous graph of private spaces"""
        private_gdf = graph_test_data["private"]
        private_to_private_gdf = graph_test_data["private_to_private"]
        
        # Create graph from private spaces and their connections
        private_graph = create_homogeneous_graph(
            nodes_gdf=private_gdf,
            edges_gdf=private_to_private_gdf,
            id_col="tess_id",
            attribute_cols=["test1", "test2", "test3"],
            source_col="from_private_id",
            target_col="to_private_id"
        )
        
        # Check graph structure
        assert isinstance(private_graph, dgl.DGLGraph)
        assert private_graph.num_nodes() == 4  # t1, t2, t3, t4
        assert private_graph.num_edges() == 2  # Two connections: t1-t2, t3-t4
        
        # Check node features
        assert private_graph.ndata["test1"].shape == (4, 1)
        assert private_graph.ndata["test2"].shape == (4, 1)
        assert private_graph.ndata["test3"].shape == (4, 1)
        
        # Check edge connectivity
        src, dst = private_graph.edges()
        src_ids = [private_gdf.iloc[i.item()]["tess_id"] for i in src]
        dst_ids = [private_gdf.iloc[i.item()]["tess_id"] for i in dst]
        edge_pairs = list(zip(src_ids, dst_ids))
        
        # Check expected connections
        assert ('t1', 't2') in edge_pairs or ('t2', 't1') in edge_pairs
        assert ('t3', 't4') in edge_pairs or ('t4', 't3') in edge_pairs
    
    def test_create_homogeneous_graph_with_no_source_target(self, graph_test_data):
        """Test creating a homogeneous graph with automatic source/target detection"""
        private_gdf = graph_test_data["private"]
        private_to_private_gdf = graph_test_data["private_to_private"]
        
        # Create graph without specifying source_col and target_col
        private_graph = create_homogeneous_graph(
            nodes_gdf=private_gdf,
            edges_gdf=private_to_private_gdf,
            id_col="tess_id",
            attribute_cols=["test1", "test2"]
        )
        
        # Check that the graph was created successfully
        assert isinstance(private_graph, dgl.DGLGraph)
        assert private_graph.num_nodes() == 4
        assert private_graph.num_edges() == 2
    
    def test_create_homogeneous_graph_empty_inputs(self):
        """Test creating a homogeneous graph with empty inputs"""
        empty_nodes_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")
        empty_edges_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")
        
        # Empty nodes should raise ValueError
        with pytest.raises(ValueError, match="nodes_gdf is empty"):
            create_homogeneous_graph(
                nodes_gdf=empty_nodes_gdf,
                edges_gdf=empty_edges_gdf,
                id_col="id",
                attribute_cols=["attr1"]
            )
        
        # Create some nodes but empty edges
        nodes_gdf = gpd.GeoDataFrame(
            {
                'id': ['n1', 'n2'],
                'attr1': [0.1, 0.2],
                'geometry': [Point(0, 0), Point(1, 1)]
            },
            crs="EPSG:27700"
        )
        
        # Should create graph with no edges
        graph = create_homogeneous_graph(
            nodes_gdf=nodes_gdf,
            edges_gdf=empty_edges_gdf,
            id_col="id",
            attribute_cols=["attr1"]
        )
        
        assert isinstance(graph, dgl.DGLGraph)
        assert graph.num_nodes() == 2
        assert graph.num_edges() == 0
    
    def test_create_homogeneous_graph_attribute_handling(self, graph_test_data):
        """Test attribute handling in homogeneous graph creation"""
        private_gdf = graph_test_data["private"]
        private_to_private_gdf = graph_test_data["private_to_private"]
        
        # Test with no attributes
        graph_no_attrs = create_homogeneous_graph(
            nodes_gdf=private_gdf,
            edges_gdf=private_to_private_gdf,
            id_col="tess_id",
            attribute_cols=[]
        )
        
        assert isinstance(graph_no_attrs, dgl.DGLGraph)
        assert graph_no_attrs.num_nodes() == 4
        assert not graph_no_attrs.ndata  # No node attributes
        
        # Test with non-existent attributes (should warn but not fail)
        with pytest.warns(RuntimeWarning):
            graph_bad_attrs = create_homogeneous_graph(
                nodes_gdf=private_gdf,
                edges_gdf=private_to_private_gdf,
                id_col="tess_id",
                attribute_cols=["nonexistent_attribute"]
            )
            
        assert isinstance(graph_bad_attrs, dgl.DGLGraph)
        assert graph_bad_attrs.num_nodes() == 4
        assert not graph_bad_attrs.ndata  # No valid node attributes
        
        # Test with mixed valid and invalid attributes
        with pytest.warns(RuntimeWarning):
            graph_mixed_attrs = create_homogeneous_graph(
                nodes_gdf=private_gdf,
                edges_gdf=private_to_private_gdf,
                id_col="tess_id",
                attribute_cols=["test1", "nonexistent_attribute"]
            )
            
        assert isinstance(graph_mixed_attrs, dgl.DGLGraph)
        assert graph_mixed_attrs.num_nodes() == 4
        assert "test1" in graph_mixed_attrs.ndata
        assert "nonexistent_attribute" not in graph_mixed_attrs.ndata


class TestHeterogeneousGraph:
    def test_create_heterogeneous_graph_basic(self, graph_test_data):
        """Test basic heterogeneous graph creation"""
        # Get test data
        private_gdf = graph_test_data["private"]
        public_gdf = graph_test_data["public"]
        private_to_private_gdf = graph_test_data["private_to_private"]
        public_to_public_gdf = graph_test_data["public_to_public"]
        private_to_public_gdf = graph_test_data["private_to_public"]
        
        # Create heterogeneous graph
        het_graph = create_heterogeneous_graph(
            nodes_dict={
                "private": private_gdf,
                "public": public_gdf
            },
            edges_dict={
                ("private", "connects_to", "private"): private_to_private_gdf,
                ("private", "adjacent_to", "public"): private_to_public_gdf,
                ("public", "connects_to", "public"): public_to_public_gdf
            },
            node_id_cols={
                "private": "tess_id",
                "public": "id"
            },
            node_attribute_cols={
                "private": ["test1", "test2", "test3"],
                "public": ["test1", "test2"]
            },
            edge_source_cols={
                ("private", "connects_to", "private"): "from_private_id",
                ("private", "adjacent_to", "public"): "private_id",
                ("public", "connects_to", "public"): "from_public_id"
            },
            edge_target_cols={
                ("private", "connects_to", "private"): "to_private_id",
                ("private", "adjacent_to", "public"): "public_id",
                ("public", "connects_to", "public"): "to_public_id"
            }
        )
        
        # Check graph structure
        assert isinstance(het_graph, dgl.DGLGraph)
        assert het_graph.ntypes == ["private", "public"]
        assert set(het_graph.canonical_etypes) == {
            ("private", "connects_to", "private"),
            ("private", "adjacent_to", "public"),
            ("public", "connects_to", "public")
        }
        
        # Check node counts
        assert het_graph.num_nodes("private") == 4
        assert het_graph.num_nodes("public") == 2
        
        # Check edge counts
        assert het_graph.num_edges(("private", "connects_to", "private")) == 2
        assert het_graph.num_edges(("private", "adjacent_to", "public")) == 4
        assert het_graph.num_edges(("public", "connects_to", "public")) == 1
        
        # Check node features
        assert "test1" in het_graph.nodes["private"].data
        assert "test2" in het_graph.nodes["private"].data
        assert "test3" in het_graph.nodes["private"].data
        assert "test1" in het_graph.nodes["public"].data
        assert "test2" in het_graph.nodes["public"].data
    
    def test_create_heterogeneous_graph_empty_inputs(self):
        """Test heterogeneous graph creation with empty inputs"""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")
        
        # Empty nodes dict should raise ValueError
        with pytest.raises(ValueError, match="nodes_dict is empty"):
            create_heterogeneous_graph(
                nodes_dict={},
                edges_dict={},
                node_id_cols={},
                node_attribute_cols={}
            )
        
        # Empty node type should raise ValueError
        with pytest.raises(ValueError, match="Node type 'private' has empty GeoDataFrame"):
            create_heterogeneous_graph(
                nodes_dict={"private": empty_gdf},
                edges_dict={},
                node_id_cols={"private": "id"},
                node_attribute_cols={"private": []}
            )
        
        # Valid nodes but empty edges should work
        nodes_gdf = gpd.GeoDataFrame(
            {
                'id': ['n1', 'n2'],
                'attr1': [0.1, 0.2],
                'geometry': [Point(0, 0), Point(1, 1)]
            },
            crs="EPSG:27700"
        )
        
        # Should create graph with no edges
        het_graph = create_heterogeneous_graph(
            nodes_dict={"node_type": nodes_gdf},
            edges_dict={},
            node_id_cols={"node_type": "id"},
            node_attribute_cols={"node_type": ["attr1"]}
        )
        
        assert isinstance(het_graph, dgl.DGLGraph)
        assert het_graph.ntypes == ["node_type"]
        assert het_graph.num_nodes("node_type") == 2
        assert not het_graph.canonical_etypes  # No edge types
    
    def test_create_heterogeneous_graph_attribute_handling(self, graph_test_data):
        """Test attribute handling in heterogeneous graph creation"""
        private_gdf = graph_test_data["private"]
        public_gdf = graph_test_data["public"]
        private_to_private_gdf = graph_test_data["private_to_private"]
        
        # Test with no attributes
        het_graph_no_attrs = create_heterogeneous_graph(
            nodes_dict={
                "private": private_gdf,
                "public": public_gdf
            },
            edges_dict={
                ("private", "connects_to", "private"): private_to_private_gdf
            },
            node_id_cols={
                "private": "tess_id",
                "public": "id"
            },
            node_attribute_cols={
                "private": [],
                "public": []
            },
            edge_source_cols={
                ("private", "connects_to", "private"): "from_private_id"
            },
            edge_target_cols={
                ("private", "connects_to", "private"): "to_private_id"
            }
        )
        
        assert isinstance(het_graph_no_attrs, dgl.DGLGraph)
        assert not het_graph_no_attrs.nodes["private"].data  # No node attributes
        assert not het_graph_no_attrs.nodes["public"].data  # No node attributes
        
        # Test with non-existent attributes (should warn but not fail)
        with pytest.warns(RuntimeWarning):
            het_graph_bad_attrs = create_heterogeneous_graph(
                nodes_dict={
                    "private": private_gdf,
                    "public": public_gdf
                },
                edges_dict={
                    ("private", "connects_to", "private"): private_to_private_gdf
                },
                node_id_cols={
                    "private": "tess_id",
                    "public": "id"
                },
                node_attribute_cols={
                    "private": ["nonexistent_attribute"],
                    "public": ["nonexistent_attribute"]
                },
                edge_source_cols={
                    ("private", "connects_to", "private"): "from_private_id"
                },
                edge_target_cols={
                    ("private", "connects_to", "private"): "to_private_id"
                }
            )
        
        assert isinstance(het_graph_bad_attrs, dgl.DGLGraph)
        assert not het_graph_bad_attrs.nodes["private"].data  # No valid attributes
        assert not het_graph_bad_attrs.nodes["public"].data  # No valid attributes


class TestMorphologicalGraph:
    def test_create_morphological_graph_basic(self, graph_test_data):
        """Test creating a morphological graph with basic parameters"""
        private_gdf = graph_test_data["private"]
        public_gdf = graph_test_data["public"]
        
        # Create morphological graph
        morpho_graph = create_morphological_graph(
            private_gdf=private_gdf,
            public_gdf=public_gdf,
            private_id_col="tess_id",
            public_id_col="id",
            private_attribute_cols=['test1', 'test2', 'test3'],
            public_attribute_cols=['test1', 'test2'],
            private_group_col="enclosure_index",
            public_geom_col="barrier_geometry"
        )
        
        # Check that it's a heterogeneous graph with proper node and edge types
        assert isinstance(morpho_graph, dgl.DGLGraph)
        assert set(morpho_graph.ntypes) == {"private", "public"}
        assert set(morpho_graph.canonical_etypes) == {
            ("private", "connects_to", "private"),
            ("private", "adjacent_to", "public"),
            ("public", "connects_to", "public")
        }
        
        # Check node counts
        assert morpho_graph.num_nodes("private") == 4
        assert morpho_graph.num_nodes("public") == 2
        
        # Check that node features were transferred
        assert "test1" in morpho_graph.nodes["private"].data
        assert "test2" in morpho_graph.nodes["private"].data
        assert "test3" in morpho_graph.nodes["private"].data
        assert "test1" in morpho_graph.nodes["public"].data
        assert "test2" in morpho_graph.nodes["public"].data
    
    def test_create_morphological_graph_different_contiguity(self, graph_test_data):
        """Test morphological graph creation with different contiguity settings"""
        private_gdf = graph_test_data["private"]
        public_gdf = graph_test_data["public"]
        
        # Create with queen contiguity
        morpho_graph_queen = create_morphological_graph(
            private_gdf=private_gdf,
            public_gdf=public_gdf,
            private_id_col="tess_id",
            public_id_col="id",
            private_attribute_cols=['test1'],
            public_attribute_cols=['test1'],
            contiguity="queen"
        )
        
        # Create with rook contiguity
        morpho_graph_rook = create_morphological_graph(
            private_gdf=private_gdf,
            public_gdf=public_gdf,
            private_id_col="tess_id",
            public_id_col="id",
            private_attribute_cols=['test1'],
            public_attribute_cols=['test1'],
            contiguity="rook"
        )
        
        # Queen should potentially have more private-to-private connections
        queen_edges = morpho_graph_queen.num_edges(("private", "connects_to", "private"))
        rook_edges = morpho_graph_rook.num_edges(("private", "connects_to", "private"))
        
        # In our test data, they might be the same due to the simple grid layout
        # but queen should never have fewer edges than rook
        assert queen_edges >= rook_edges
    
    def test_create_morphological_graph_different_tolerance(self, graph_test_data):
        """Test morphological graph creation with different tolerance values"""
        private_gdf = graph_test_data["private"]
        public_gdf = graph_test_data["public"]
        
        # Create with small tolerance
        morpho_graph_small_tol = create_morphological_graph(
            private_gdf=private_gdf,
            public_gdf=public_gdf,
            private_id_col="tess_id",
            public_id_col="id",
            private_attribute_cols=['test1'],
            public_attribute_cols=['test1'],
            tolerance=0.01
        )
        
        # Create with large tolerance
        morpho_graph_large_tol = create_morphological_graph(
            private_gdf=private_gdf,
            public_gdf=public_gdf,
            private_id_col="tess_id",
            public_id_col="id",
            private_attribute_cols=['test1'],
            public_attribute_cols=['test1'],
            tolerance=5.0
        )
        
        # Larger tolerance should potentially find more connections
        small_tol_public_edges = morpho_graph_small_tol.num_edges(("public", "connects_to", "public"))
        large_tol_public_edges = morpho_graph_large_tol.num_edges(("public", "connects_to", "public"))
        
        # This might be the same for our test data, but larger tolerance should never find fewer
        assert large_tol_public_edges >= small_tol_public_edges
        
        small_tol_priv_pub_edges = morpho_graph_small_tol.num_edges(("private", "adjacent_to", "public"))
        large_tol_priv_pub_edges = morpho_graph_large_tol.num_edges(("private", "adjacent_to", "public"))
        
        assert large_tol_priv_pub_edges >= small_tol_priv_pub_edges
    
    def test_create_morphological_graph_empty_inputs(self):
        """Test morphological graph creation with empty inputs"""
        empty_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:27700")
        valid_gdf = gpd.GeoDataFrame(
            {
                'id': ['n1', 'n2'],
                'attr1': [0.1, 0.2],
                'geometry': [Point(0, 0), Point(1, 1)]
            },
            crs="EPSG:27700"
        )
        
        # Empty private GeoDataFrame should raise ValueError
        with pytest.raises(ValueError):
            create_morphological_graph(
                private_gdf=empty_gdf,
                public_gdf=valid_gdf,
                private_id_col="id",
                public_id_col="id"
            )
        
        # Empty public GeoDataFrame should raise ValueError
        with pytest.raises(ValueError):
            create_morphological_graph(
                private_gdf=valid_gdf,
                public_gdf=empty_gdf,
                private_id_col="id",
                public_id_col="id"
            )
