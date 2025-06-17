import pytest
import geopandas as gpd
from geopandas import testing as gpd_testing # Import for assert_geodataframe_equal
import pandas as pd
from shapely.geometry import Point, LineString
import networkx as nx
import numpy as np

# Attempt to import torch and torch_geometric
try:
    import torch
    from torch_geometric.data import Data, HeteroData
    torch_geometric_available = True
except ImportError:
    torch_geometric_available = False
    # Define dummy classes if torch_geometric is not available for type hinting
    class Data:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    class HeteroData:
        def __init__(self):
            self._node_types = []
            self._edge_types = []

        def __getitem__(self, key):
            if isinstance(key, str): # node type
                if key not in self._node_types:
                    self._node_types.append(key)
                if not hasattr(self, key):
                    setattr(self, key, Data())
                return getattr(self, key)
            elif isinstance(key, tuple): # edge type
                if key not in self._edge_types:
                    self._edge_types.append(key)
                if not hasattr(self, "_".join(key)):
                     setattr(self, "_".join(key), Data())
                return getattr(self, "_".join(key))
            raise TypeError("Invalid key type")

        @property
        def node_types(self):
            return self._node_types

        @property
        def edge_types(self):
            return self._edge_types

    class PyTorchDevice:
        """Dummy PyTorchDevice class."""
        ...
    class PyTorchDtype:
        """Dummy PyTorchDtype class."""
        ...
    torch = None # type: ignore[assignment]

from city2graph import graph as c2g_graph # Import the module to allow mocking its globals

# Constants
TEST_CRS = "EPSG:27700"

# Pytest marker for skipping tests if torch or torch_geometric is not available
skip_if_no_torch = pytest.mark.skipif(not torch_geometric_available, reason="PyTorch or PyTorch Geometric is not available")

# Helper for comparing GDFs
def assert_gdf_equals(gdf1, gdf2, check_like=False, **kwargs):
    if check_like: # only check columns and dtypes
        pd.testing.assert_frame_equal(gdf1.drop(columns='geometry'), gdf2.drop(columns='geometry'), check_dtype=True, check_like=True, **kwargs)
    else:
        gpd_testing.assert_geodataframe_equal(gdf1, gdf2, check_dtype=True, **kwargs) # Use gpd_testing
    if gdf1.crs and gdf2.crs:
        assert gdf1.crs.equals(gdf2.crs)
    elif gdf1.crs is None and gdf2.crs is None:
        pass # both are None, which is fine
    else:
        assert False, f"CRS mismatch: {gdf1.crs} != {gdf2.crs}"


@pytest.fixture
def basic_nodes_data():
    return {
        "id": ["N1", "N2", "N3"],
        "feat1": [1.0, 2.0, 3.0],
        "feat2": [10, 20, 30],
        "label1": [0, 1, 0],
        "geometry": [Point(0, 0), Point(1, 1), Point(0, 1)],
    }

@pytest.fixture
def basic_nodes_gdf(basic_nodes_data):
    gdf = gpd.GeoDataFrame(basic_nodes_data, crs=TEST_CRS)
    return gdf

@pytest.fixture
def basic_edges_data():
    return {
        "source_id": ["N1", "N2", "N1"],
        "target_id": ["N2", "N3", "N3"],
        "edge_feat1": [0.5, 1.5, 2.5],
        "geometry": [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (0, 1)]),
            LineString([(0, 0), (0, 1)]),
        ],
    }

@pytest.fixture
def basic_edges_gdf(basic_edges_data):
    return gpd.GeoDataFrame(basic_edges_data, crs=TEST_CRS)

@pytest.fixture
def hetero_nodes_data():
    poi_gdf = gpd.GeoDataFrame({
        "poi_id": ["P1", "P2"],
        "category": ["food", "shop"],
        "geometry": [Point(10, 10), Point(20, 20)],
    }, crs=TEST_CRS)

    junction_gdf = gpd.GeoDataFrame({
        "junc_id": ["J1", "J2", "J3"],
        "traffic_signal": [True, False, True],
        "geometry": [Point(10, 0), Point(0, 10), Point(20, 10)],
    }, crs=TEST_CRS)

    return {
        "poi": poi_gdf,
        "junction": junction_gdf,
    }

@pytest.fixture
def hetero_edges_data(hetero_nodes_data):
    # Ensure source/target IDs exist in hetero_nodes_data
    return {
        ("poi", "links_to", "junction"): gpd.GeoDataFrame({
            "source": ["P1", "P2"], # poi_id
            "target": ["J1", "J2"], # junc_id
            "distance": [10.0, 5.0],
            "geometry": [
                LineString([hetero_nodes_data["poi"].loc["P1"].geometry, hetero_nodes_data["junction"].loc["J1"].geometry]),
                LineString([hetero_nodes_data["poi"].loc["P2"].geometry, hetero_nodes_data["junction"].loc["J2"].geometry]),
            ]
        }, crs=TEST_CRS),
        ("junction", "connects", "junction"): gpd.GeoDataFrame({
            "u": ["J1"], # junc_id
            "v": ["J3"], # junc_id
            "road_type": ["primary"],
            "geometry": [
                 LineString([hetero_nodes_data["junction"].loc["J1"].geometry, hetero_nodes_data["junction"].loc["J3"].geometry]),
            ]
        }, crs=TEST_CRS),
    }

@pytest.fixture
def sample_nx_graph():
    g = nx.Graph()
    g.add_node(0, feat1=1.0, label1=0, pos=(0,0))
    g.add_node(1, feat1=2.0, label1=1, pos=(1,1))
    g.add_node(2, feat1=3.0, label1=0, pos=(0,1))
    g.add_edge(0, 1, edge_feat1=0.5)
    g.add_edge(1, 2, edge_feat1=1.5)
    g.graph['crs'] = TEST_CRS # Store CRS in graph attributes
    return g

# --- Test is_torch_available ---
def test_is_torch_available(monkeypatch):
    monkeypatch.setattr(c2g_graph, "TORCH_AVAILABLE", True)
    assert c2g_graph.is_torch_available() is True
    monkeypatch.setattr(c2g_graph, "TORCH_AVAILABLE", False)
    assert c2g_graph.is_torch_available() is False

# --- Test _get_device ---
@skip_if_no_torch
class TestGetDevice:
    def test_get_device_none(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert c2g_graph._get_device(None) == torch.device("cpu")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert c2g_graph._get_device(None) == torch.device("cuda")

    def test_get_device_str(self):
        assert c2g_graph._get_device("cpu") == torch.device("cpu")
        if torch.cuda.is_available():
            assert c2g_graph._get_device("cuda") == torch.device("cuda")
        else:
            with pytest.raises(ValueError, match="Device must be 'cuda', 'cpu'"): # PyTorch itself might raise error if CUDA not avail
                 c2g_graph._get_device("cuda")


    def test_get_device_torch_device(self):
        cpu_dev = torch.device("cpu")
        assert c2g_graph._get_device(cpu_dev) == cpu_dev
        if torch.cuda.is_available():
            cuda_dev = torch.device("cuda")
            assert c2g_graph._get_device(cuda_dev) == cuda_dev

    def test_get_device_invalid_str(self):
        with pytest.raises(ValueError, match="Device must be 'cuda', 'cpu'"):
            c2g_graph._get_device("tpu")

    def test_get_device_invalid_type(self):
        with pytest.raises(TypeError, match="Device must be 'cuda', 'cpu'"):
            c2g_graph._get_device(123) # type: ignore[arg-type]

# --- Tests for gdf_to_pyg ---
@skip_if_no_torch
class TestGdfToPygHomogeneous:
    def test_nodes_only(self, basic_nodes_gdf):
        data = c2g_graph.gdf_to_pyg(nodes=basic_nodes_gdf, node_feature_cols=['feat1', 'feat2'])
        assert isinstance(data, Data)
        assert data.x.shape == (3, 2)
        assert data.pos.shape == (3, 2)
        assert 'edge_index' not in data or data.edge_index.shape[1] == 0
        assert data.crs == basic_nodes_gdf.crs

    def test_nodes_and_edges(self, basic_nodes_gdf, basic_edges_gdf):
        data = c2g_graph.gdf_to_pyg(
            nodes=basic_nodes_gdf,
            edges=basic_edges_gdf,
            node_id_cols='id', # Using the GDF index name
            node_feature_cols=['feat1'],
            node_label_cols=['label1'],
            edge_source_col='source_id',
            edge_target_col='target_id',
            edge_feature_cols=['edge_feat1']
        )
        assert data.x.shape == (3, 1)
        assert data.y.shape == (3, 1)
        assert data.pos.shape == (3, 2)
        assert data.edge_index.shape == (2, 3) # 3 edges
        assert data.edge_attr.shape == (3, 1)
        assert data.crs == basic_nodes_gdf.crs

    def test_auto_detect_edge_columns(self, basic_nodes_gdf, basic_edges_gdf):
        # Rename columns for auto-detection
        renamed_edges_gdf = basic_edges_gdf.rename(columns={'source_id': 'u', 'target_id': 'v'})
        data = c2g_graph.gdf_to_pyg(
            nodes=basic_nodes_gdf,
            edges=renamed_edges_gdf,
            node_feature_cols=['feat1']
        )
        assert data.edge_index.shape == (2, 3)

    def test_node_id_from_column(self, basic_nodes_data, basic_edges_data):
        nodes_df = pd.DataFrame(basic_nodes_data)
        nodes_gdf = gpd.GeoDataFrame(nodes_df, geometry=nodes_df['geometry'], crs=TEST_CRS) # 'id' is a column

        edges_df = pd.DataFrame(basic_edges_data)
        edges_gdf = gpd.GeoDataFrame(edges_df, geometry=edges_df['geometry'], crs=TEST_CRS)

        data = c2g_graph.gdf_to_pyg(
            nodes=nodes_gdf,
            edges=edges_gdf,
            node_id_cols='id',
            node_feature_cols=['feat1'],
            edge_source_col='source_id',
            edge_target_col='target_id'
        )
        assert data.num_nodes == 3
        assert data.num_edges == 3

    def test_empty_edges(self, basic_nodes_gdf):
        empty_edges_gdf = gpd.GeoDataFrame(columns=['source', 'target', 'geometry'], crs=TEST_CRS)
        data = c2g_graph.gdf_to_pyg(nodes=basic_nodes_gdf, edges=empty_edges_gdf)
        assert data.edge_index.shape[1] == 0
        assert data.edge_attr.shape[0] == 0

    def test_different_dtypes(self, basic_nodes_gdf):
        data_float32 = c2g_graph.gdf_to_pyg(nodes=basic_nodes_gdf, node_feature_cols=['feat1'], dtype=torch.float32)
        assert data_float32.x.dtype == torch.float32
        if torch_geometric_available and hasattr(torch, 'float16'): # float16 might not be available on all setups
            data_float16 = c2g_graph.gdf_to_pyg(nodes=basic_nodes_gdf, node_feature_cols=['feat1'], dtype=torch.float16)
            assert data_float16.x.dtype == torch.float16


@skip_if_no_torch
class TestGdfToPygHeterogeneous:
    def test_hetero_nodes_only(self, hetero_nodes_data):
        data = c2g_graph.gdf_to_pyg(
            nodes=hetero_nodes_data,
            node_id_cols={"poi": "poi_id", "junction": "junc_id"}, # Using index names
            node_feature_cols={"poi": ["category"], "junction": ["traffic_signal"]}
        )
        assert isinstance(data, HeteroData)
        assert "poi" in data.node_types
        assert "junction" in data.node_types
        assert data["poi"].x.shape[0] == 2 # 2 poi nodes
        assert data["junction"].x.shape[0] == 3 # 3 junction nodes
        assert data["poi"].pos.shape == (2,2)
        assert data.crs == TEST_CRS

    def test_hetero_nodes_and_edges(self, hetero_nodes_data, hetero_edges_data):
        data = c2g_graph.gdf_to_pyg(
            nodes=hetero_nodes_data,
            edges=hetero_edges_data,
            node_id_cols={"poi": "poi_id", "junction": "junc_id"},
            node_feature_cols={"poi": ["category"], "junction": ["traffic_signal"]},
            edge_feature_cols={"links_to": ["distance"], "connects": ["road_type"]}
        )
        assert isinstance(data, HeteroData)
        assert data["poi", "links_to", "junction"].edge_index.shape[1] == 2 # 2 poi-junction edges
        assert data["poi", "links_to", "junction"].edge_attr.shape[1] == 1
        assert data["junction", "connects", "junction"].edge_index.shape[1] == 1
        assert data["junction", "connects", "junction"].edge_attr.shape[1] == 1
        assert data.crs == TEST_CRS


# --- Tests for pyg_to_gdf (Round Trip) ---
@skip_if_no_torch
class TestPygToGdfRoundtripHomogeneous:
    @pytest.mark.parametrize("with_edges", [True, False])
    @pytest.mark.parametrize("with_node_labels", [True, False])
    @pytest.mark.parametrize("with_edge_features", [True, False])
    def test_roundtrip(self, basic_nodes_gdf, basic_edges_gdf, with_edges, with_node_labels, with_edge_features):
        node_feat_cols = ['feat1', 'feat2']
        node_lbl_cols = ['label1'] if with_node_labels else None
        edge_feat_cols = ['edge_feat1'] if with_edge_features else None
        
        current_edges_gdf = basic_edges_gdf if with_edges else None
        if current_edges_gdf is not None and not with_edge_features: # drop edge features if not used
            current_edges_gdf = current_edges_gdf[['source_id', 'target_id', 'geometry']].copy()


        pyg_data = c2g_graph.gdf_to_pyg(
            nodes=basic_nodes_gdf,
            edges=current_edges_gdf,
            node_id_cols='id', # index name
            node_feature_cols=node_feat_cols,
            node_label_cols=node_lbl_cols,
            edge_source_col='source_id',
            edge_target_col='target_id',
            edge_feature_cols=edge_feat_cols
        )

        re_nodes_gdf, re_edges_gdf = c2g_graph.pyg_to_gdf(pyg_data)

        # Prepare original basic_nodes_gdf for comparison (select relevant columns)
        expected_nodes_cols = ['geometry'] + node_feat_cols
        if node_lbl_cols:
            expected_nodes_cols += node_lbl_cols
        
        # pyg_to_gdf might add 'node_id' if original index was not named 'id' or if id_col was not 'index'
        # and it was a regular column. Here, basic_nodes_gdf index is 'id'.
        # _reconstruct_node_gdf uses original_ids for index if id_col was 'index'
        expected_nodes_gdf = basic_nodes_gdf[expected_nodes_cols].copy()
        
        assert_gdf_equals(re_nodes_gdf, expected_nodes_gdf, check_like=False, check_index_type=False)


        if with_edges:
            assert re_edges_gdf is not None
            expected_edges_cols = ['geometry']
            if edge_feat_cols:
                expected_edges_cols += edge_feat_cols
            
            # pyg_to_gdf reconstructs edges without source/target columns, geometry is LineString
            # The original edge GDF has source/target columns.
            # We need to compare the features and geometry.
            # The index of re_edges_gdf might be simple RangeIndex.
            assert_gdf_equals(re_edges_gdf[expected_edges_cols], current_edges_gdf[expected_edges_cols], check_like=True, check_index_type=False, check_names=False)

        else:
            assert re_edges_gdf is None or re_edges_gdf.empty


@skip_if_no_torch
class TestPygToGdfRoundtripHeterogeneous:
    def test_roundtrip_hetero(self, hetero_nodes_data, hetero_edges_data):
        node_id_cols_map = {ntype: gdf.index.name for ntype, gdf in hetero_nodes_data.items()}
        
        pyg_data = c2g_graph.gdf_to_pyg(
            nodes=hetero_nodes_data,
            edges=hetero_edges_data,
            node_id_cols=node_id_cols_map,
            node_feature_cols={"poi": ["category"], "junction": ["traffic_signal"]},
            edge_feature_cols={"links_to": ["distance"], "connects": ["road_type"]}
        )

        re_nodes_gdfs, re_edges_gdfs = c2g_graph.pyg_to_gdf(pyg_data)

        for node_type, original_gdf in hetero_nodes_data.items():
            assert node_type in re_nodes_gdfs
            re_gdf = re_nodes_gdfs[node_type]
            
            expected_cols = ['geometry']
            if node_type == "poi": expected_cols.append("category")
            if node_type == "junction": expected_cols.append("traffic_signal")
            
            assert_gdf_equals(re_gdf, original_gdf[expected_cols], check_like=False, check_index_type=False)


        for edge_type, original_gdf in hetero_edges_data.items():
            assert edge_type in re_edges_gdfs
            re_gdf = re_edges_gdfs[edge_type]
            
            expected_cols = ['geometry']
            rel_type = edge_type[1]
            if rel_type == "links_to": expected_cols.append("distance")
            if rel_type == "connects": expected_cols.append("road_type")

            assert_gdf_equals(re_gdf[expected_cols], original_gdf[expected_cols], check_like=True, check_index_type=False, check_names=False)

# --- Tests for nx_to_pyg ---
@skip_if_no_torch
class TestNxToPyg:
    def test_basic_conversion(self, sample_nx_graph):
        data = c2g_graph.nx_to_pyg(
            sample_nx_graph,
            node_feature_cols=['feat1'],
            node_label_cols=['label1'],
            edge_feature_cols=['edge_feat1']
        )
        assert isinstance(data, Data)
        assert data.x.shape == (3, 1)
        assert data.y.shape == (3, 1)
        assert data.pos.shape == (3, 2) # From 'pos' attribute
        assert data.edge_index.shape == (2, 2) # 2 edges
        assert data.edge_attr.shape == (2, 1)
        assert data.crs == TEST_CRS

    def test_nx_to_pyg_no_features(self, sample_nx_graph):
        # Create a graph with no features to test defaults
        g = nx.Graph()
        g.add_node(0, pos=(0,0))
        g.add_node(1, pos=(1,1))
        g.add_edge(0,1)
        g.graph['crs'] = TEST_CRS

        data = c2g_graph.nx_to_pyg(g)
        assert data.x.shape == (2,0) # No node features
        assert data.pos.shape == (2,2)
        assert data.edge_index.shape == (2,1)
        assert data.edge_attr.shape == (1,0) # No edge features
        assert data.crs == TEST_CRS


# --- Tests for pyg_to_nx (Round Trip) ---
@skip_if_no_torch
class TestPygToNxRoundtrip:
    def test_roundtrip_homogeneous_nx(self, sample_nx_graph):
        # NX -> PyG
        pyg_data = c2g_graph.nx_to_pyg(
            sample_nx_graph,
            node_feature_cols=['feat1'],
            node_label_cols=['label1'],
            edge_feature_cols=['edge_feat1']
        )
        # PyG -> NX
        re_nx_graph = c2g_graph.pyg_to_nx(pyg_data)

        assert re_nx_graph.graph.get('crs') == sample_nx_graph.graph.get('crs')
        assert len(re_nx_graph.nodes) == len(sample_nx_graph.nodes)
        assert len(re_nx_graph.edges) == len(sample_nx_graph.edges)

        for node_id, attrs in sample_nx_graph.nodes(data=True):
            re_attrs = re_nx_graph.nodes[node_id]
            assert pytest.approx(attrs['feat1']) == re_attrs['feat1']
            assert attrs['label1'] == re_attrs['label1']
            # Position might be tuple of floats
            assert np.allclose(attrs['pos'], re_attrs['pos'])


        # Edge attributes can be tricky due to order if multiple edges
        # For simple graph, this is okay
        for u, v, attrs in sample_nx_graph.edges(data=True):
            # Check if edge exists (order might be swapped in undirected graph)
            if (u,v) in re_nx_graph.edges:
                re_attrs = re_nx_graph.edges[(u,v)]
            elif (v,u) in re_nx_graph.edges:
                re_attrs = re_nx_graph.edges[(v,u)]
            else:
                assert False, f"Edge ({u},{v}) not found in reconstructed graph"
            assert pytest.approx(attrs['edge_feat1']) == re_attrs['edge_feat1']


    def test_hetero_pyg_to_nx(self, hetero_nodes_data, hetero_edges_data):
        node_id_cols_map = {ntype: gdf.index.name for ntype, gdf in hetero_nodes_data.items()}
        pyg_hetero_data = c2g_graph.gdf_to_pyg(
            nodes=hetero_nodes_data,
            edges=hetero_edges_data,
            node_id_cols=node_id_cols_map,
            node_feature_cols={"poi": ["category"], "junction": ["traffic_signal"]},
            edge_feature_cols={"links_to": ["distance"], "connects": ["road_type"]}
        )

        nx_graph = c2g_graph.pyg_to_nx(pyg_hetero_data)

        assert nx_graph.graph.get('is_hetero') is True
        assert nx_graph.graph.get('crs') == TEST_CRS
        
        num_expected_nodes = sum(len(gdf) for gdf in hetero_nodes_data.values())
        assert len(nx_graph.nodes) == num_expected_nodes
        
        num_expected_edges = sum(len(gdf) for gdf in hetero_edges_data.values())
        assert len(nx_graph.edges) == num_expected_edges

        # Check for node_type and edge_type attributes
        node_types_in_nx = {data['node_type'] for _, data in nx_graph.nodes(data=True)}
        assert "poi" in node_types_in_nx
        assert "junction" in node_types_in_nx

        edge_types_in_nx = {data['edge_type'] for _, _, data in nx_graph.edges(data=True)}
        assert "links_to" in edge_types_in_nx
        assert "connects" in edge_types_in_nx
        
        # Verify some attribute presence
        for _, data in nx_graph.nodes(data=True):
            if data['node_type'] == 'poi':
                assert 'category' in data
            if data['node_type'] == 'junction':
                assert 'traffic_signal' in data
        
        for _, _, data in nx_graph.edges(data=True):
            if data['edge_type'] == 'links_to':
                assert 'distance' in data
            if data['edge_type'] == 'connects':
                assert 'road_type' in data


# --- Test Error Handling and Edge Cases ---
class TestErrorHandlingAndEdgeCases:
    def test_gdf_to_pyg_torch_unavailable(self, monkeypatch, basic_nodes_gdf):
        monkeypatch.setattr(c2g_graph, "TORCH_AVAILABLE", False)
        with pytest.raises(ImportError, match="PyTorch required"):
            c2g_graph.gdf_to_pyg(nodes=basic_nodes_gdf)

    def test_pyg_to_gdf_torch_unavailable(self, monkeypatch):
        monkeypatch.setattr(c2g_graph, "TORCH_AVAILABLE", False)
        # _validate_pyg is called first, which raises the error
        dummy_data = Data() # type: ignore[call-arg]
        if torch_geometric_available: # if torch is actually available, make a real dummy
             dummy_data = Data(edge_index=torch.empty((2,0), dtype=torch.long))


        with pytest.raises(ImportError, match="PyTorch required"):
            c2g_graph.pyg_to_gdf(dummy_data)


    def test_nx_to_pyg_torch_unavailable(self, monkeypatch, sample_nx_graph):
        monkeypatch.setattr(c2g_graph, "TORCH_AVAILABLE", False)
        with pytest.raises(ImportError, match="PyTorch required"):
            c2g_graph.nx_to_pyg(sample_nx_graph)

    def test_pyg_to_nx_torch_unavailable(self, monkeypatch):
        monkeypatch.setattr(c2g_graph, "TORCH_AVAILABLE", False)
        dummy_data = Data() # type: ignore[call-arg]
        if torch_geometric_available:
             dummy_data = Data(edge_index=torch.empty((2,0), dtype=torch.long))

        with pytest.raises(ImportError, match="PyTorch required"):
            c2g_graph.pyg_to_nx(dummy_data)

    @skip_if_no_torch
    def test_gdf_to_pyg_empty_nodes_gdf(self):
        empty_nodes = gpd.GeoDataFrame(columns=['id', 'geometry'], crs=TEST_CRS)
        data = c2g_graph.gdf_to_pyg(nodes=empty_nodes)
        assert data.num_nodes == 0
        assert data.x.shape[0] == 0
        assert data.pos is None or data.pos.shape[0] == 0 # pos might be None if no geometry

    @skip_if_no_torch
    def test_gdf_to_pyg_nodes_no_geometry(self, basic_nodes_data):
        nodes_df_no_geom = pd.DataFrame(basic_nodes_data).drop(columns=['geometry'])
        # Create GDF with an explicit geometry column of Nones to be valid for GeoPandas when CRS is set
        nodes_gdf_no_geom = gpd.GeoDataFrame(
            nodes_df_no_geom, 
            geometry=[None] * len(nodes_df_no_geom), 
            crs=TEST_CRS
        )
        
        data = c2g_graph.gdf_to_pyg(nodes=nodes_gdf_no_geom)
        assert data.pos is None or data.pos.shape[0] == 0 or torch.all(torch.isnan(data.pos))

    @skip_if_no_torch
    def test_gdf_to_pyg_invalid_node_id_col(self, basic_nodes_gdf):
        with pytest.raises(ValueError, match="Provided id_col 'invalid_id_col' not found"):
            c2g_graph.gdf_to_pyg(nodes=basic_nodes_gdf, node_id_cols='invalid_id_col')
