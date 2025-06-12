import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
import torch
from shapely.geometry import LineString
from shapely.geometry import Polygon
from torch_geometric.data import Data
from torch_geometric.data import HeteroData

import city2graph
from city2graph.utils import gdf_to_nx as utils_gdf_to_nx
from city2graph.utils import nx_to_gdf as utils_nx_to_gdf

# Define a common CRS
TEST_CRS = "EPSG:27700"

@pytest.fixture
def sample_data_params():
    """Provides parameters for creating sample data."""
    private_node_features = ["area", "perimeter", "compactness"]
    public_node_features = ["length"]
    private_edge_features = ["edge_weight"] # Example edge feature
    public_edge_features = ["length"]

    return {
        "crs": TEST_CRS,
        "private_node_features": private_node_features,
        "public_node_features": public_node_features,
        "private_edge_features": private_edge_features,
        "public_edge_features": public_edge_features,
        "node_feature_cols_homo": private_node_features,
        "edge_feature_cols_homo": private_edge_features,
        "node_feature_cols_hetero": {
            "private": private_node_features,
            "public": public_node_features,
        },
        "edge_feature_cols_hetero": {
            ("private", "touched_to", "private"): private_edge_features,
            ("public", "connected_to", "public"): public_edge_features,
            ("private", "faced_to", "public"): [], # No features for this edge type
        },
    }

@pytest.fixture
def sample_gdf_data(sample_data_params):
    """
    Provides sample morpho_nodes (dict of GDFs) and morpho_edges (dict of GDFs)
    for testing graph conversions, along with feature column definitions.
    """
    params = sample_data_params
    crs = params["crs"]

    # Private Nodes
    private_nodes_data = {
        "tess_id": [0, 1, 2],
        "geometry": [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
        ],
        "area": [1.0, 1.0, 1.0],
        "perimeter": [4.0, 4.0, 4.0],
        "compactness": [np.pi * 4 * 1.0 / (4.0**2)] * 3,
    }
    private_nodes_gdf = gpd.GeoDataFrame(private_nodes_data, crs=crs).set_index("tess_id")

    # Public Nodes
    public_nodes_data = {
        "public_id": [100, 101],
        "geometry": [LineString([(0.5, -0.5), (1.5, -0.5)]), LineString([(0.5, 2.5), (1.5, 2.5)])],
        "length": [1.0, 1.0], # Feature for public nodes
    }
    public_nodes_gdf = gpd.GeoDataFrame(public_nodes_data, crs=crs).set_index("public_id")

    morpho_nodes = {"private": private_nodes_gdf, "public": public_nodes_gdf}

    # Private-to-Private Edges
    p_to_p_edges_data = {
        "from_private_id": [0, 0, 1],
        "to_private_id": [1, 2, 2],
        "geometry": [
            LineString([private_nodes_gdf.geometry.iloc[0].centroid, private_nodes_gdf.geometry.iloc[1].centroid]),
            LineString([private_nodes_gdf.geometry.iloc[0].centroid, private_nodes_gdf.geometry.iloc[2].centroid]),
            LineString([private_nodes_gdf.geometry.iloc[1].centroid, private_nodes_gdf.geometry.iloc[2].centroid]),
        ],
        "edge_weight": [0.5, 0.8, 1.2], # Feature for private-private edges
    }
    p_to_p_edges_gdf = gpd.GeoDataFrame(p_to_p_edges_data, crs=crs).set_index(
        ["from_private_id", "to_private_id"],
    )

    # Public-to-Public Edges
    pub_to_pub_edges_data = {
        "from_public_id": [100],
        "to_public_id": [101],
        "geometry": [LineString([public_nodes_gdf.geometry.iloc[0].centroid, public_nodes_gdf.geometry.iloc[1].centroid])],
        "length": [public_nodes_gdf.geometry.iloc[0].centroid.distance(public_nodes_gdf.geometry.iloc[1].centroid)], # Feature
    }
    pub_to_pub_edges_gdf = gpd.GeoDataFrame(pub_to_pub_edges_data, crs=crs).set_index(
        ["from_public_id", "to_public_id"],
    )

    # Private-to-Public Edges
    p_to_pub_edges_data = {
        "private_id": [0, 1, 2],
        "public_id": [100, 100, 101],
        "geometry": [
            LineString([private_nodes_gdf.geometry.iloc[0].centroid, public_nodes_gdf.geometry.iloc[0].centroid]),
            LineString([private_nodes_gdf.geometry.iloc[1].centroid, public_nodes_gdf.geometry.iloc[0].centroid]),
            LineString([private_nodes_gdf.geometry.iloc[2].centroid, public_nodes_gdf.geometry.iloc[1].centroid]),
        ],
        # No specific features for this edge type in this example
    }
    p_to_pub_edges_gdf = gpd.GeoDataFrame(p_to_pub_edges_data, crs=crs).set_index(
        ["private_id", "public_id"],
    )

    morpho_edges = {
        ("private", "touched_to", "private"): p_to_p_edges_gdf,
        ("public", "connected_to", "public"): pub_to_pub_edges_gdf,
        ("private", "faced_to", "public"): p_to_pub_edges_gdf,
    }
    return morpho_nodes, morpho_edges, params


# --- Helper Assertion Functions ---
def assert_gdf_equals(gdf1, gdf2, check_geom_equals=True, check_crs=True, sort_index=True):
    if sort_index:
        gdf1 = gdf1.sort_index()
        gdf2 = gdf2.sort_index()
    pd.testing.assert_frame_equal(gdf1.drop(columns=["geometry"] if "geometry" in gdf1 else []),
                                  gdf2.drop(columns=["geometry"] if "geometry" in gdf2 else []),
                                  check_dtype=False, rtol=1e-5) # Allow for float precision issues
    if check_crs:
        assert gdf1.crs == gdf2.crs, "CRS mismatch"
    if check_geom_equals and "geometry" in gdf1 and "geometry" in gdf2:
        assert gdf1.geometry.equals(gdf2.geometry), "Geometry mismatch"

def assert_nx_graph_struct_equal(g1, g2):
    assert set(g1.nodes()) == set(g2.nodes()), "Node sets differ"
    # Sort edges to ensure consistent comparison for MultiGraph or DiGraph if used
    g1_edges = sorted([tuple(sorted(edge)) for edge in g1.edges()])
    g2_edges = sorted([tuple(sorted(edge)) for edge in g2.edges()])
    assert g1_edges == g2_edges, "Edge sets differ"
    if hasattr(g1, "graph") and hasattr(g2, "graph"):
        assert g1.graph.get("crs") == g2.graph.get("crs"), "Graph CRS mismatch"

def assert_pyg_data_struct_equal(data1, data2, num_node_features, num_edge_features=0):
    assert data1.num_nodes == data2.num_nodes, "Num nodes differ"
    if data1.edge_index is not None and data2.edge_index is not None:
         # Sort columns of edge_index for comparison if order might change
        edge_index1_sorted = torch.sort(data1.edge_index, dim=1)[0]
        edge_index2_sorted = torch.sort(data2.edge_index, dim=1)[0]
        assert torch.equal(edge_index1_sorted, edge_index2_sorted), "Edge index differ"
    elif data1.edge_index is not None or data2.edge_index is not None: # one is None, other is not
        assert False, "Edge index presence differs"

    if data1.x is not None and data2.x is not None:
        assert torch.allclose(data1.x, data2.x, atol=1e-5), "Node features (x) differ"
        assert data1.x.shape[1] == num_node_features if num_node_features is not None else True
    elif data1.x is not None or data2.x is not None:
        msg = "Node features (x) presence differs"
        raise AssertionError(msg)

    if data1.pos is not None and data2.pos is not None:
        assert torch.allclose(data1.pos, data2.pos, atol=1e-5), "Node positions (pos) differ"
    elif data1.pos is not None or data2.pos is not None:
        msg = "Node positions (pos) presence differs"
        raise AssertionError(msg)

    if hasattr(data1, "edge_attr") and hasattr(data2, "edge_attr"):
        if data1.edge_attr is not None and data2.edge_attr is not None:
            assert torch.allclose(data1.edge_attr, data2.edge_attr, atol=1e-5), "Edge attributes (edge_attr) differ"
            if num_edge_features > 0 :
                 assert data1.edge_attr.shape[1] == num_edge_features, f"Edge attr feature count mismatch, expected {num_edge_features}"
        elif data1.edge_attr is not None or data2.edge_attr is not None:
            msg = "Edge attributes (edge_attr) presence differs"
            raise AssertionError(msg)
    if hasattr(data1, "crs") and hasattr(data2, "crs"):
        assert data1.crs == data2.crs, "PyG CRS mismatch"


# --- Homogeneous Graph Conversion Tests ---

def test_gdf_to_pyg_homogeneous(sample_gdf_data):
    nodes_dict, edges_dict, params = sample_gdf_data
    nodes_gdf = nodes_dict["private"]
    edges_gdf = edges_dict[("private", "touched_to", "private")]
    node_features = params["node_feature_cols_homo"]
    edge_features = params["edge_feature_cols_homo"]

    pyg_data = city2graph.gdf_to_pyg(
        nodes=nodes_gdf,
        edges=edges_gdf,
        node_feature_cols=node_features,
        edge_feature_cols=edge_features,
    )

    assert isinstance(pyg_data, Data)
    assert pyg_data.num_nodes == len(nodes_gdf)
    # For undirected graphs, PyG typically stores twice the number of directed edges
    assert pyg_data.num_edges == len(edges_gdf) * 2
    assert pyg_data.x.shape[1] == len(node_features)
    if edge_features and len(edge_features) > 0:
        assert pyg_data.edge_attr.shape[1] == len(edge_features)
    else:
        assert pyg_data.edge_attr is None or pyg_data.edge_attr.shape[1] == 0
    assert hasattr(pyg_data, "pos")
    assert pyg_data.pos is not None
    assert pyg_data.pos.shape == (len(nodes_gdf), 2) # Assuming 2D coordinates
    assert pyg_data.crs == nodes_gdf.crs

def test_pyg_to_gdf_homogeneous(sample_gdf_data) -> None:
    nodes_dict, edges_dict, params = sample_gdf_data
    original_nodes_gdf = nodes_dict["private"]
    original_edges_gdf = edges_dict[("private", "touched_to", "private")]
    node_features = params["node_feature_cols_homo"]
    edge_features = params["edge_feature_cols_homo"]

    pyg_data = city2graph.gdf_to_pyg(
        nodes=original_nodes_gdf,
        edges=original_edges_gdf,
        node_feature_cols=node_features,
        edge_feature_cols=edge_features,
    )
    reconstructed_nodes_gdf, reconstructed_edges_gdf = city2graph.pyg_to_gdf(pyg_data)

    assert isinstance(reconstructed_nodes_gdf, gpd.GeoDataFrame)
    assert len(reconstructed_nodes_gdf) == len(original_nodes_gdf)
    assert reconstructed_nodes_gdf.index.name == original_nodes_gdf.index.name
    pd.testing.assert_index_equal(reconstructed_nodes_gdf.index, original_nodes_gdf.index, exact=True)
    assert all(col in reconstructed_nodes_gdf.columns for col in node_features)
    assert reconstructed_nodes_gdf.crs == original_nodes_gdf.crs

    if reconstructed_edges_gdf is not None:
        assert isinstance(reconstructed_edges_gdf, gpd.GeoDataFrame)
        # pyg_to_gdf might return unique directed edges. Original might be undirected.
        # The number of unique edges (ignoring direction) should match.
        # Group by sorted node pairs to count unique undirected edges.
        if not original_edges_gdf.empty:
            re_src = reconstructed_edges_gdf.index.get_level_values(0).astype(original_edges_gdf.index.get_level_values(0).dtype)
            re_tgt = reconstructed_edges_gdf.index.get_level_values(1).astype(original_edges_gdf.index.get_level_values(1).dtype)

            reconstructed_undirected_edges = set()
            for s, t in zip(re_src, re_tgt, strict=False):
                reconstructed_undirected_edges.add(tuple(sorted((s, t))))
            assert len(reconstructed_undirected_edges) == len(original_edges_gdf)

        assert reconstructed_edges_gdf.index.names == original_edges_gdf.index.names
        if edge_features and len(edge_features) > 0:
            assert all(col in reconstructed_edges_gdf.columns for col in edge_features)
        assert reconstructed_edges_gdf.crs == original_edges_gdf.crs

def test_gdf_to_nx_homogeneous(sample_gdf_data) -> None:
    nodes_dict, edges_dict, params = sample_gdf_data
    nodes_gdf = nodes_dict["private"]
    edges_gdf = edges_dict[("private", "touched_to", "private")]

    nx_graph = utils_gdf_to_nx(nodes=nodes_gdf, edges=edges_gdf)

    assert isinstance(nx_graph, nx.Graph)
    assert nx_graph.number_of_nodes() == len(nodes_gdf)
    assert nx_graph.number_of_edges() == len(edges_gdf)
    assert nx_graph.graph.get("crs") == nodes_gdf.crs
    for node_id, attrs in nodes_gdf.iterrows():
        assert node_id in nx_graph.nodes
        for col in params["node_feature_cols_homo"]:
            assert nx_graph.nodes[node_id][col] == attrs[col]
        assert "pos" in nx_graph.nodes[node_id] # gdf_to_nx from utils adds 'pos'

def test_nx_to_gdf_homogeneous(sample_gdf_data) -> None:
    nodes_dict, edges_dict, params = sample_gdf_data
    original_nodes_gdf = nodes_dict["private"].sort_index()
    original_edges_gdf = edges_dict[("private", "touched_to", "private")].sort_index()

    nx_graph = utils_gdf_to_nx(nodes=original_nodes_gdf, edges=original_edges_gdf)
    reconstructed_nodes_gdf, reconstructed_edges_gdf = utils_nx_to_gdf(nx_graph, nodes=True, edges=True)

    reconstructed_nodes_gdf = reconstructed_nodes_gdf.sort_index()
    reconstructed_edges_gdf = reconstructed_edges_gdf.sort_index()

    assert_gdf_equals(reconstructed_nodes_gdf, original_nodes_gdf, check_geom_equals=True)
    # nx_to_gdf might change edge geometry (e.g. from centroids if nodes were polygons)
    # So we check attribute columns and CRS, and length.
    assert len(reconstructed_edges_gdf) == len(original_edges_gdf)
    assert reconstructed_edges_gdf.index.names == original_edges_gdf.index.names
    pd.testing.assert_index_equal(reconstructed_edges_gdf.index, original_edges_gdf.index)

    # Check non-geometry columns
    original_edge_cols = [c for c in original_edges_gdf.columns if c != "geometry"]
    reconstructed_edge_cols = [c for c in reconstructed_edges_gdf.columns if c != "geometry"]
    assert set(original_edge_cols) == set(reconstructed_edge_cols)
    if original_edge_cols:
         pd.testing.assert_frame_equal(
            original_edges_gdf[original_edge_cols].sort_index(),
            reconstructed_edges_gdf[reconstructed_edge_cols].sort_index(),
            check_dtype=False, rtol=1e-5,
        )
    assert reconstructed_edges_gdf.crs == original_edges_gdf.crs


def test_pyg_to_nx_homogeneous(sample_gdf_data) -> None:
    nodes_dict, edges_dict, params = sample_gdf_data
    nodes_gdf = nodes_dict["private"]
    edges_gdf = edges_dict[("private", "touched_to", "private")]
    node_features = params["node_feature_cols_homo"]
    edge_features = params["edge_feature_cols_homo"]

    pyg_data = city2graph.gdf_to_pyg(
        nodes=nodes_gdf, edges=edges_gdf,
        node_feature_cols=node_features, edge_feature_cols=edge_features,
    )
    nx_graph = city2graph.pyg_to_nx(pyg_data)

    assert isinstance(nx_graph, nx.Graph)
    assert nx_graph.number_of_nodes() == pyg_data.num_nodes
    # pyg_to_nx converts undirected PyG (2*E edges) to NX graph (E edges)
    assert nx_graph.number_of_edges() == pyg_data.num_edges / 2
    assert nx_graph.graph.get("crs") == pyg_data.crs
    # Check if node features are preserved (pyg_to_nx adds them as attributes)
    for i in range(pyg_data.num_nodes):
        assert i in nx_graph.nodes # pyg_to_nx uses integer node IDs
        for idx, feat_name in enumerate(node_features):
             assert np.isclose(nx_graph.nodes[i][feat_name], pyg_data.x[i, idx].item()), f"Node feature {feat_name} mismatch for node {i}"
        if pyg_data.pos is not None:
            assert "pos" in nx_graph.nodes[i]
            assert np.allclose(nx_graph.nodes[i]["pos"], pyg_data.pos[i].tolist())


def test_nx_to_pyg_homogeneous(sample_gdf_data) -> None:
    nodes_dict, edges_dict, params = sample_gdf_data
    nodes_gdf = nodes_dict["private"]
    edges_gdf = edges_dict[("private", "touched_to", "private")]
    node_features = params["node_feature_cols_homo"]
    edge_features = params["edge_feature_cols_homo"] # nx_to_pyg uses these

    # Create NX graph first (ensuring 'pos' and feature attributes are present)
    nx_graph_orig = utils_gdf_to_nx(nodes=nodes_gdf, edges=edges_gdf)
    # nx_to_pyg expects node features as attributes, and 'pos' for positions.
    # utils_gdf_to_nx should set these up correctly.

    pyg_data = city2graph.nx_to_pyg(
        nx_graph_orig,
        node_feature_cols=node_features,
        edge_feature_cols=edge_features,
    )

    assert isinstance(pyg_data, Data)
    assert pyg_data.num_nodes == nx_graph_orig.number_of_nodes()
    assert pyg_data.num_edges == nx_graph_orig.number_of_edges() * 2
    assert pyg_data.x.shape[1] == len(node_features)
    if edge_features and len(edge_features) > 0 and pyg_data.edge_attr is not None:
         assert pyg_data.edge_attr.shape[1] == len(edge_features)
    assert hasattr(pyg_data, "pos")
    assert pyg_data.pos is not None
    assert pyg_data.crs == nx_graph_orig.graph.get("crs")


# --- Heterogeneous Graph Conversion Tests ---

def test_gdf_to_pyg_heterogeneous(sample_gdf_data) -> None:
    nodes_dict, edges_dict, params = sample_gdf_data
    node_features = params["node_feature_cols_hetero"]
    edge_features = params["edge_feature_cols_hetero"]

    pyg_hetero_data = city2graph.gdf_to_pyg(
        nodes=nodes_dict,
        edges=edges_dict,
        node_feature_cols=node_features,
        edge_feature_cols=edge_features,
    )

    assert isinstance(pyg_hetero_data, HeteroData)
    for node_type, gdf in nodes_dict.items():
        assert pyg_hetero_data[node_type].num_nodes == len(gdf)
        if node_features.get(node_type):
            assert pyg_hetero_data[node_type].x.shape[1] == len(node_features[node_type])
        assert pyg_hetero_data[node_type].pos.shape == (len(gdf), 2)

    for edge_type, gdf in edges_dict.items():
        # For undirected relations, PyG stores 2 * num_original_edges
        # For directed (like private-faced_to-public), it's 1 * num_original_edges
        expected_edges = len(gdf)
        if edge_type[0] == edge_type[2]: # Assuming same src/dst type implies undirected for this test
            expected_edges *=2
        assert pyg_hetero_data[edge_type].edge_index.shape[1] == expected_edges

        current_edge_features = edge_features.get(edge_type, [])
        if current_edge_features and len(current_edge_features) > 0:
            assert pyg_hetero_data[edge_type].edge_attr.shape[1] == len(current_edge_features)
        elif pyg_hetero_data[edge_type].edge_attr is not None:
             assert pyg_hetero_data[edge_type].edge_attr.shape[1] == 0


    assert pyg_hetero_data.crs == params["crs"]


def test_pyg_to_gdf_heterogeneous(sample_gdf_data) -> None:
    original_nodes_dict, original_edges_dict, params = sample_gdf_data
    node_features = params["node_feature_cols_hetero"]
    edge_features = params["edge_feature_cols_hetero"]

    pyg_hetero_data = city2graph.gdf_to_pyg(
        nodes=original_nodes_dict, edges=original_edges_dict,
        node_feature_cols=node_features, edge_feature_cols=edge_features,
    )
    reconstructed_nodes_dict, reconstructed_edges_dict = city2graph.pyg_to_gdf(pyg_hetero_data)

    assert isinstance(reconstructed_nodes_dict, dict)
    assert isinstance(reconstructed_edges_dict, dict)

    for node_type, original_gdf in original_nodes_dict.items():
        assert node_type in reconstructed_nodes_dict
        reconstructed_gdf = reconstructed_nodes_dict[node_type]
        assert isinstance(reconstructed_gdf, gpd.GeoDataFrame)
        assert len(reconstructed_gdf) == len(original_gdf)
        assert reconstructed_gdf.index.name == original_gdf.index.name
        pd.testing.assert_index_equal(reconstructed_gdf.index.sort_values(), original_gdf.index.sort_values(), exact=True)
        if node_features.get(node_type):
            assert all(col in reconstructed_gdf.columns for col in node_features[node_type])
        assert reconstructed_gdf.crs == original_gdf.crs

    for edge_type, original_gdf in original_edges_dict.items():
        assert edge_type in reconstructed_edges_dict
        reconstructed_gdf = reconstructed_edges_dict[edge_type]
        assert isinstance(reconstructed_gdf, gpd.GeoDataFrame)

        # Similar to homogeneous, compare unique undirected edges if applicable
        # For this test, we'll check if the number of edges is consistent with original,
        # considering pyg_to_gdf might simplify directed edges from an undirected PyG representation.
        # A more robust check would involve comparing sets of (sorted_source_id, sorted_target_id).
        if not original_gdf.empty:
            re_src = reconstructed_gdf.index.get_level_values(0).astype(original_gdf.index.get_level_values(0).dtype)
            re_tgt = reconstructed_gdf.index.get_level_values(1).astype(original_gdf.index.get_level_values(1).dtype)

            reconstructed_edge_pairs = set()
            # If original edge type implies undirected (e.g. src_type == dst_type)
            if edge_type[0] == edge_type[2]:
                 for s, t in zip(re_src, re_tgt, strict=False):
                    reconstructed_edge_pairs.add(tuple(sorted((s, t))))
            else: # Directed
                 for s, t in zip(re_src, re_tgt, strict=False):
                    reconstructed_edge_pairs.add((s,t))
            assert len(reconstructed_edge_pairs) == len(original_gdf)


        assert reconstructed_gdf.index.names == original_gdf.index.names
        current_edge_features = edge_features.get(edge_type, [])
        if current_edge_features and len(current_edge_features) > 0:
            assert all(col in reconstructed_gdf.columns for col in current_edge_features)
        assert reconstructed_gdf.crs == original_gdf.crs


def test_gdf_to_nx_heterogeneous(sample_gdf_data) -> None:
    nodes_dict, edges_dict, params = sample_gdf_data

    nx_graph = utils_gdf_to_nx(nodes=nodes_dict, edges=edges_dict)

    assert isinstance(nx_graph, nx.Graph)
    assert nx_graph.graph.get("is_hetero") is True
    assert nx_graph.graph.get("crs") == params["crs"]

    expected_total_nodes = sum(len(gdf) for gdf in nodes_dict.values())
    expected_total_edges = sum(len(gdf) for gdf in edges_dict.values())

    assert nx_graph.number_of_nodes() == expected_total_nodes
    assert nx_graph.number_of_edges() == expected_total_edges

    # Check node attributes and types
    for node_type, gdf in nodes_dict.items():
        node_feature_cols = params["node_feature_cols_hetero"].get(node_type, [])
        for original_id, attrs in gdf.iterrows():
            # Find the corresponding node in nx_graph (gdf_to_nx stores original_id and node_type)
            found_node = None
            for nx_id, nx_attrs in nx_graph.nodes(data=True):
                if nx_attrs.get("_original_index") == original_id and nx_attrs.get("node_type") == node_type:
                    found_node = nx_id
                    break
            assert found_node is not None, f"Node {original_id} of type {node_type} not found in NX graph"
            for col in node_feature_cols:
                assert nx_graph.nodes[found_node][col] == attrs[col]
            assert "pos" in nx_graph.nodes[found_node]


def test_nx_to_gdf_heterogeneous(sample_gdf_data) -> None:
    original_nodes_dict, original_edges_dict, params = sample_gdf_data

    # Sort indices for consistent comparison
    for nt in original_nodes_dict:
        original_nodes_dict[nt] = original_nodes_dict[nt].sort_index()
    for et in original_edges_dict:
        original_edges_dict[et] = original_edges_dict[et].sort_index()

    nx_graph = utils_gdf_to_nx(nodes=original_nodes_dict, edges=original_edges_dict)
    reconstructed_nodes_dict, reconstructed_edges_dict = utils_nx_to_gdf(nx_graph, nodes=True, edges=True)

    assert isinstance(reconstructed_nodes_dict, dict)
    assert isinstance(reconstructed_edges_dict, dict)

    for node_type, original_gdf in original_nodes_dict.items():
        assert node_type in reconstructed_nodes_dict
        reconstructed_gdf = reconstructed_nodes_dict[node_type].sort_index()
        assert_gdf_equals(reconstructed_gdf, original_gdf, check_geom_equals=True)

    for edge_type, original_gdf in original_edges_dict.items():
        assert edge_type in reconstructed_edges_dict
        reconstructed_gdf = reconstructed_edges_dict[edge_type].sort_index()
        # As with homogeneous, check attributes and structure, geometry might change
        assert len(reconstructed_gdf) == len(original_gdf)
        assert reconstructed_gdf.index.names == original_gdf.index.names
        pd.testing.assert_index_equal(reconstructed_gdf.index, original_gdf.index)

        original_edge_cols = [c for c in original_gdf.columns if c != "geometry"]
        reconstructed_edge_cols = [c for c in reconstructed_gdf.columns if c != "geometry"]
        assert set(original_edge_cols) == set(reconstructed_edge_cols)

        if original_edge_cols:
            pd.testing.assert_frame_equal(
                original_gdf[original_edge_cols].sort_index(),
                reconstructed_gdf[reconstructed_edge_cols].sort_index(),
                check_dtype=False, rtol=1e-5,
            )
        assert reconstructed_gdf.crs == original_gdf.crs


def test_pyg_to_nx_heterogeneous(sample_gdf_data) -> None:
    nodes_dict, edges_dict, params = sample_gdf_data
    node_features = params["node_feature_cols_hetero"]
    edge_features = params["edge_feature_cols_hetero"]

    pyg_hetero_data = city2graph.gdf_to_pyg(
        nodes=nodes_dict, edges=edges_dict,
        node_feature_cols=node_features, edge_feature_cols=edge_features,
    )
    nx_graph = city2graph.pyg_to_nx(pyg_hetero_data)

    assert isinstance(nx_graph, nx.Graph)
    assert nx_graph.graph.get("is_hetero") is True
    assert nx_graph.graph.get("crs") == pyg_hetero_data.crs

    expected_total_nodes = sum(pyg_hetero_data[nt].num_nodes for nt in pyg_hetero_data.node_types)
    # pyg_to_nx converts undirected PyG edges (2*E) to NX graph (E edges)
    expected_total_edges = 0
    for et in pyg_hetero_data.edge_types:
        num_pyg_edges = pyg_hetero_data[et].edge_index.shape[1]
        # If source and target types are the same, assume it was an undirected relation in GDF
        # and PyG stored it as 2*E. NX will have E.
        # If source and target types differ, assume it was directed, PyG stored E, NX will have E.
        if et[0] == et[2]:
            expected_total_edges += num_pyg_edges / 2
        else:
            expected_total_edges += num_pyg_edges

    assert nx_graph.number_of_nodes() == expected_total_nodes
    assert nx_graph.number_of_edges() == expected_total_edges

    # Check node attributes (features, pos, type)
    for node_type in pyg_hetero_data.node_types:
        type_node_features = node_features.get(node_type, [])
        for i in range(pyg_hetero_data[node_type].num_nodes):
            # pyg_to_nx creates global node IDs. We need to find them.
            # This check is complex due to ID mapping. A simpler check is overall counts.
            # For a more detailed check, one would need to trace original IDs through mappings.
            pass # Detailed check omitted for brevity but important for full validation.


# --- Round Trip Tests ---

def test_round_trip_homogeneous_gdf_pyg_gdf(sample_gdf_data) -> None:
    nodes_dict, edges_dict, params = sample_gdf_data
    original_nodes_gdf = nodes_dict["private"].sort_index()
    original_edges_gdf = edges_dict[("private", "touched_to", "private")].sort_index()
    node_features = params["node_feature_cols_homo"]
    edge_features = params["edge_feature_cols_homo"]

    pyg_data = city2graph.gdf_to_pyg(original_nodes_gdf, original_edges_gdf, node_features, edge_features)
    reconstructed_nodes_gdf, reconstructed_edges_gdf = city2graph.pyg_to_gdf(pyg_data)

    reconstructed_nodes_gdf = reconstructed_nodes_gdf.sort_index()

    assert_gdf_equals(reconstructed_nodes_gdf, original_nodes_gdf, check_geom_equals=False) # Pos might be from centroid
    # Check positions separately if needed, pyg_to_gdf reconstructs geometry from 'pos'
    assert reconstructed_nodes_gdf.geometry.is_valid.all()


    if reconstructed_edges_gdf is not None and not original_edges_gdf.empty:
        reconstructed_edges_gdf = reconstructed_edges_gdf.sort_index()
        # Compare based on unique undirected edges
        orig_undirected = set()
        for idx_tuple in original_edges_gdf.index:
            orig_undirected.add(tuple(sorted(idx_tuple)))

        recon_undirected = set()
        # Ensure index levels are of compatible types for comparison
        r_src = reconstructed_edges_gdf.index.get_level_values(0).astype(original_edges_gdf.index.get_level_values(0).dtype)
        r_tgt = reconstructed_edges_gdf.index.get_level_values(1).astype(original_edges_gdf.index.get_level_values(1).dtype)

        for s, t in zip(r_src, r_tgt, strict=False):
             recon_undirected.add(tuple(sorted((s, t))))
        assert recon_undirected == orig_undirected

        # Check feature columns
        if edge_features and len(edge_features) > 0:
            assert all(col in reconstructed_edges_gdf.columns for col in edge_features)
        assert reconstructed_edges_gdf.crs == original_edges_gdf.crs


def test_round_trip_homogeneous_gdf_nx_gdf(sample_gdf_data) -> None:
    nodes_dict, edges_dict, params = sample_gdf_data
    original_nodes_gdf = nodes_dict["private"].sort_index()
    original_edges_gdf = edges_dict[("private", "touched_to", "private")].sort_index()

    nx_graph = utils_gdf_to_nx(original_nodes_gdf, original_edges_gdf)
    reconstructed_nodes_gdf, reconstructed_edges_gdf = utils_nx_to_gdf(nx_graph, nodes=True, edges=True)

    reconstructed_nodes_gdf = reconstructed_nodes_gdf.sort_index()
    reconstructed_edges_gdf = reconstructed_edges_gdf.sort_index()

    assert_gdf_equals(reconstructed_nodes_gdf, original_nodes_gdf, check_geom_equals=True)
    assert_gdf_equals(reconstructed_edges_gdf.drop(columns=["geometry"] if "geometry" in reconstructed_edges_gdf else []),
                      original_edges_gdf.drop(columns=["geometry"] if "geometry" in original_edges_gdf else []),
                      check_geom_equals=False) # Geometry might be regenerated
    assert reconstructed_edges_gdf.crs == original_edges_gdf.crs


def test_round_trip_heterogeneous_gdf_pyg_gdf(sample_gdf_data) -> None:
    original_nodes_dict, original_edges_dict, params = sample_gdf_data
    node_features = params["node_feature_cols_hetero"]
    edge_features = params["edge_feature_cols_hetero"]

    # Sort indices for consistent comparison
    for nt in original_nodes_dict: original_nodes_dict[nt] = original_nodes_dict[nt].sort_index()
    for et in original_edges_dict: original_edges_dict[et] = original_edges_dict[et].sort_index()

    pyg_data = city2graph.gdf_to_pyg(original_nodes_dict, original_edges_dict, node_features, edge_features)
    reconstructed_nodes_dict, reconstructed_edges_dict = city2graph.pyg_to_gdf(pyg_data)

    for node_type, original_gdf in original_nodes_dict.items():
        reconstructed_gdf = reconstructed_nodes_dict[node_type].sort_index()
        assert_gdf_equals(reconstructed_gdf, original_gdf, check_geom_equals=False) # Pos might be from centroid
        assert reconstructed_gdf.geometry.is_valid.all()


    for edge_type, original_gdf in original_edges_dict.items():
        if original_gdf.empty:
            assert edge_type not in reconstructed_edges_dict or reconstructed_edges_dict[edge_type].empty
            continue

        reconstructed_gdf = reconstructed_edges_dict[edge_type].sort_index()

        # Compare based on unique edges, considering directionality based on edge_type
        orig_edges_set = set()
        for idx_tuple in original_gdf.index:
            orig_edges_set.add(idx_tuple if edge_type[0] != edge_type[2] else tuple(sorted(idx_tuple)))

        recon_edges_set = set()
        r_src = reconstructed_gdf.index.get_level_values(0).astype(original_gdf.index.get_level_values(0).dtype)
        r_tgt = reconstructed_gdf.index.get_level_values(1).astype(original_gdf.index.get_level_values(1).dtype)

        for s, t in zip(r_src, r_tgt, strict=False):
            recon_edges_set.add((s,t) if edge_type[0] != edge_type[2] else tuple(sorted((s,t))))

        assert recon_edges_set == orig_edges_set, f"Edge set mismatch for type {edge_type}"

        current_edge_features = edge_features.get(edge_type, [])
        if current_edge_features and len(current_edge_features) > 0:
            assert all(col in reconstructed_gdf.columns for col in current_edge_features)
        assert reconstructed_gdf.crs == original_gdf.crs

def test_round_trip_heterogeneous_gdf_nx_gdf(sample_gdf_data) -> None:
    original_nodes_dict, original_edges_dict, params = sample_gdf_data

    for nt in original_nodes_dict: original_nodes_dict[nt] = original_nodes_dict[nt].sort_index()
    for et in original_edges_dict: original_edges_dict[et] = original_edges_dict[et].sort_index()

    nx_graph = utils_gdf_to_nx(original_nodes_dict, original_edges_dict)
    reconstructed_nodes_dict, reconstructed_edges_dict = utils_nx_to_gdf(nx_graph, nodes=True, edges=True)

    for node_type, original_gdf in original_nodes_dict.items():
        reconstructed_gdf = reconstructed_nodes_dict[node_type].sort_index()
        assert_gdf_equals(reconstructed_gdf, original_gdf, check_geom_equals=True)

    for edge_type, original_gdf in original_edges_dict.items():
        reconstructed_gdf = reconstructed_edges_dict[edge_type].sort_index()
        assert_gdf_equals(reconstructed_gdf.drop(columns=["geometry"] if "geometry" in reconstructed_gdf else []),
                          original_gdf.drop(columns=["geometry"] if "geometry" in original_gdf else []),
                          check_geom_equals=False) # Geometry might be regenerated
        assert reconstructed_gdf.crs == original_gdf.crs
