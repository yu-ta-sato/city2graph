import pathlib
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon

# Try to import torch, skip tests if not available
try:
    import importlib.util

    TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture(scope="session")
def sample_crs() -> str:
    """Coordinate Reference System fixture."""
    return "EPSG:27700"


@pytest.fixture
def sample_nodes_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Fixture for a homogeneous nodes GeoDataFrame."""
    data = {
        "node_id": [1, 2, 3, 4],
        "feature1": [10.0, 20.0, 30.0, 40.0],
        "label1": [0, 1, 0, 1],
        "geometry": [Point(0, 0), Point(1, 1), Point(0, 1), Point(1, 0)],
    }
    return gpd.GeoDataFrame(data, crs=sample_crs).set_index("node_id")


@pytest.fixture
def sample_edges_gdf() -> gpd.GeoDataFrame:
    """Fixture for a homogeneous edges GeoDataFrame."""
    data = {
        "edge_id": ["e1", "e2", "e3", "e4"],
        "source_id": [1, 1, 2, 3],
        "target_id": [2, 3, 4, 4],
        "edge_feature1": [0.5, 0.8, 1.2, 2.5],
        "geometry": [
            LineString([(0, 0), (1, 1)]),
            LineString([(0, 0), (0, 1)]),
            LineString([(1, 1), (1, 0)]),
            LineString([(0, 1), (1, 0)]),
        ],
    }
    # Create a MultiIndex for source and target IDs
    multi_index = pd.MultiIndex.from_arrays(
        [data["source_id"], data["target_id"]], names=("source_id", "target_id"),
    )
    return gpd.GeoDataFrame(data, index=multi_index, crs="EPSG:27700")


@pytest.fixture
def sample_hetero_nodes_dict(sample_crs: str) -> dict[str, gpd.GeoDataFrame]:
    """Fixture for a dictionary of heterogeneous nodes GeoDataFrames."""
    buildings_data = {
        "building_id": ["b1", "b2", "b3"],
        "b_feat1": [100.0, 150.0, 120.0],
        "b_label": [1, 0, 1],
        "geometry": [Point(10, 10), Point(11, 11), Point(10, 11)],
    }
    buildings_gdf = gpd.GeoDataFrame(buildings_data, crs=sample_crs).set_index(
        "building_id",
    )

    roads_data = {
        "road_id": ["r1", "r2"],
        "r_feat1": [5.5, 6.0],
        "r_label": [0, 0],
        "geometry": [Point(10, 12), Point(12, 12)],
    }
    roads_gdf = gpd.GeoDataFrame(roads_data, crs=sample_crs).set_index("road_id")

    return {"building": buildings_gdf, "road": roads_gdf}


@pytest.fixture
def sample_hetero_edges_dict(sample_crs: str) -> dict[tuple[str, str, str], gpd.GeoDataFrame]:
    """Fixture for a dictionary of heterogeneous edges GeoDataFrames."""
    # Connects buildings to roads
    connections_data = {
        "building_id": ["b1", "b2", "b3"],
        "road_id": ["r1", "r1", "r2"],
        "conn_feat1": [1.0, 2.0, 3.0],
        "geometry": [
            LineString([(10, 10), (10, 12)]),
            LineString([(11, 11), (10, 12)]),
            LineString([(10, 11), (12, 12)]),
        ],
    }
    connections_multi_index = pd.MultiIndex.from_arrays(
        [connections_data["building_id"], connections_data["road_id"]],
        names=("building_id", "road_id"),
    )
    connections_gdf = gpd.GeoDataFrame(
        connections_data, index=connections_multi_index, crs=sample_crs,
    )

    # Roads connect to other roads (example of same-type connection)
    road_links_data = {
        "source_road_id": ["r1", "r2"],
        "target_road_id": ["r2", "r1"],
        "link_feat1": [0.7, 0.7],
        "geometry": [LineString([(10, 12), (12, 12)]), LineString([(12, 12), (10, 12)])],
    }
    road_links_multi_index = pd.MultiIndex.from_arrays(
        [road_links_data["source_road_id"], road_links_data["target_road_id"]],
        names=("source_road_id", "target_road_id"),
    )
    road_links_gdf = gpd.GeoDataFrame(
        road_links_data, index=road_links_multi_index, crs=sample_crs,
    )

    return {
        ("building", "connects_to", "road"): connections_gdf,
        ("road", "links_to", "road"): road_links_gdf,
    }


@pytest.fixture
def sample_nx_graph() -> nx.Graph:
    """Fixture for a NetworkX graph."""
    graph = nx.Graph()
    graph.add_node(1, feature1=10.0, label1=0, pos=(0, 0), geometry=Point(0, 0))
    graph.add_node(2, feature1=20.0, label1=1, pos=(1, 1), geometry=Point(1, 1))
    graph.add_node(3, feature1=30.0, label1=0, pos=(0, 1), geometry=Point(0, 1))
    graph.add_edge(1, 2, edge_feature1=0.5, geometry=LineString([(0, 0), (1, 1)]))
    graph.add_edge(1, 3, edge_feature1=0.8, geometry=LineString([(0, 0), (0, 1)]))
    graph.graph["crs"] = "EPSG:27700"  # Add CRS to graph attributes
    graph.graph["is_hetero"] = False  # Add is_hetero flag to graph attributes
    return graph


@pytest.fixture
def sample_nx_graph_no_crs(sample_nx_graph: nx.Graph) -> nx.Graph:
    """A NetworkX graph with no CRS information."""
    graph = sample_nx_graph.copy()
    del graph.graph["crs"]
    return graph


@pytest.fixture
def sample_nx_graph_no_pos(sample_crs: str) -> nx.Graph:
    """Fixture for a NetworkX graph with no position data."""
    graph = nx.Graph()
    graph.add_node(1, feature1=10.0, label1=0)
    graph.add_node(2, feature1=20.0, label1=1)
    graph.add_edge(1, 2, edge_feature1=0.5)
    graph.graph["crs"] = sample_crs
    graph.graph["is_hetero"] = False  # Add is_hetero flag to graph attributes
    return graph


@pytest.fixture
def custom_center_point() -> Point:
    """Provides a custom center point for testing."""
    point = Point(-2.9879004, 53.4062724)
    return gpd.GeoSeries([point], crs="EPSG:4326").to_crs(epsg=27700)


# Helper function to create GeoDataFrames for testing
def _create_gdf(
    geometries: list[Any],
    attributes_list: list[dict[str, Any]],
    crs: str = "EPSG:27700",
) -> gpd.GeoDataFrame:
    """
    Creates a GeoDataFrame.
    geometries: list of shapely geometry objects
    attributes_list: list of dictionaries, where each dict contains attributes for a geometry.
    """
    if not geometries:
        columns = ["geometry"]
        # Ensure attributes_list[0] is a dict and has keys before extending
        if (
            attributes_list
            and isinstance(attributes_list[0], dict)
            and attributes_list[0]
        ):
            columns.extend(attributes_list[0].keys())
        # Remove duplicates while preserving order, ensure 'geometry' is present
        seen = set()
        unique_cols = []
        if "geometry" not in columns:  # Ensure geometry is always an option
            columns.append("geometry")
        for item in columns:
            if item not in seen:
                seen.add(item)
                unique_cols.append(item)
        return gpd.GeoDataFrame(columns=unique_cols, geometry="geometry", crs=crs)

    return gpd.GeoDataFrame(attributes_list, geometry=geometries, crs=crs)


# --- General Fixtures ---
@pytest.fixture
def empty_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame for testing."""
    return _create_gdf([], [], crs=sample_crs)


@pytest.fixture
def geojson_data_path() -> pathlib.Path:
    """Path to the directory containing GeoJSON test files."""
    # Assuming 'data' directory is at the same level as the 'tests' directory
    # and this test file is /tests/test_morphology.py
    # Adjust if your 'data' directory is elsewhere, e.g., inside 'tests'
    return pathlib.Path(__file__).parent / "data"  # Assumes data is in tests/data


# --- Core Data Fixtures from GeoJSON files ---
@pytest.fixture
def sample_buildings_gdf(geojson_data_path: pathlib.Path, sample_crs: str) -> gpd.GeoDataFrame:
    """Loads sample building data from GeoJSON."""
    file_path = geojson_data_path / "sample_buildings.geojson"
    try:
        gdf = gpd.read_file(file_path)
    except Exception as e:
        # Provide a more informative error if the file is not found
        if not file_path.exists():
            pytest.fail(
                f"GeoJSON file not found at {file_path}. Make sure 'sample_buildings.geojson' is in the '{geojson_data_path.name}' directory relative to the test file.",
            )
        pytest.fail(f"Failed to load sample_buildings.geojson from {file_path}: {e}")
    gdf = gdf.to_crs(sample_crs)
    if "bldg_id" not in gdf.columns:
        # Create a unique string ID if 'bldg_id' is missing
        gdf["bldg_id"] = gdf.index.astype(str) + "_bldg"
    # Ensure 'private_id' is not present, as it's generated by tessellation
    if "private_id" in gdf.columns:
        gdf = gdf.drop(columns=["private_id"])
    return gdf


@pytest.fixture
def sample_segments_gdf(geojson_data_path: pathlib.Path, sample_crs: str) -> gpd.GeoDataFrame:
    """Loads sample segment data from GeoJSON."""
    file_path = geojson_data_path / "sample_segments.geojson"
    try:
        gdf = gpd.read_file(file_path)
    except Exception as e:
        if not file_path.exists():
            pytest.fail(
                f"GeoJSON file not found at {file_path}. Make sure 'sample_segments.geojson' is in the '{geojson_data_path.name}' directory relative to the test file.",
            )
        pytest.fail(f"Failed to load sample_segments.geojson from {file_path}: {e}")
    gdf = gdf.to_crs(sample_crs)
    if "seg_id" not in gdf.columns:
        gdf["seg_id"] = gdf.index.astype(str) + "_seg"
    # Add 'public_id' from index, as morphological_graph would do for its internal 'segs' GDF
    # This is important if sub-functions are tested directly with this fixture.
    gdf["public_id"] = gdf.index
    return gdf


@pytest.fixture
def sample_tessellation_gdf(sample_buildings_gdf: gpd.GeoDataFrame, sample_crs: str) -> gpd.GeoDataFrame:
    """Creates a simple tessellation-like GDF from sample_buildings_gdf for testing p2p and p2pub.
    This is a placeholder for actual tessellation output.
    It assigns 'private_id' and a dummy 'enclosure_index'.
    """
    if sample_buildings_gdf.empty:
        return _create_gdf(
            [], [{"private_id": None, "enclosure_index": None}], crs=sample_crs,
        )

    # For simplicity, let's assume each building corresponds to one tessellation cell
    # In reality, tessellation is more complex.
    tess_polys = sample_buildings_gdf.geometry.tolist()
    attrs = [
        {"private_id": i, "enclosure_index": 0}  # Dummy enclosure index
        for i in range(len(sample_buildings_gdf))
    ]
    return _create_gdf(tess_polys, attrs, crs=sample_crs)


@pytest.fixture
def mg_center_point(sample_segments_gdf: gpd.GeoDataFrame, sample_crs: str) -> gpd.GeoSeries:
    """Provides a center point for filtering, e.g., centroid of the first segment."""
    if not sample_segments_gdf.empty:
        return gpd.GeoSeries(
            [sample_segments_gdf.geometry.iloc[0].centroid], crs=sample_crs,
        )
    return gpd.GeoSeries(
        [Point(0, 0)], crs=sample_crs,
    )  # Default if segments are empty


# --- Fixtures for single-item GDFs for edge case testing ---
@pytest.fixture
def single_building_gdf(sample_buildings_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """A GDF with a single building polygon."""
    if not sample_buildings_gdf.empty:
        return sample_buildings_gdf.iloc[[0]].copy()
    return sample_buildings_gdf.copy()  # Return empty if source is empty


@pytest.fixture
def single_segment_gdf(sample_segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """A GDF with a single segment line."""
    if not sample_segments_gdf.empty:
        return sample_segments_gdf.iloc[[0]].copy()
    return sample_segments_gdf.copy()


@pytest.fixture
def single_tessellation_cell_gdf(sample_tessellation_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """A GDF with a single tessellation cell."""
    if not sample_tessellation_gdf.empty:
        return sample_tessellation_gdf.iloc[[0]].copy()
    return sample_tessellation_gdf.copy()


# --- Fixtures for Error/Edge Cases ---
@pytest.fixture
def private_gdf_no_private_id(
    sample_crs: str,
) -> gpd.GeoDataFrame:  # Represents a tessellation-like input
    """Synthetic private polygons GDF (like tessellation) lacking 'private_id'."""
    return _create_gdf(
        [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])],
        [{"some_other_id": 100}],
        crs=sample_crs,
    )


@pytest.fixture
def segments_no_public_id_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Synthetic segments GDF lacking 'public_id'."""
    return _create_gdf(
        [LineString([(0, 0), (1, 1)])], [{"other_id": 10}], crs=sample_crs,
    )


@pytest.fixture
def segments_gdf_far_away(sample_segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sample segments translated far away, for testing no priv_pub connections."""
    segments_far = sample_segments_gdf.copy()
    if not segments_far.empty:
        segments_far.geometry = segments_far.geometry.translate(xoff=100000, yoff=100000)
    return segments_far


@pytest.fixture
def segments_gdf_no_crs(sample_segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sample segments GDF with CRS removed."""
    gdf_no_crs = sample_segments_gdf.copy()
    gdf_no_crs.crs = None
    return gdf_no_crs


@pytest.fixture
def segments_gdf_alt_crs(sample_segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sample segments GDF with an alternative CRS."""
    gdf = sample_segments_gdf.copy()
    if not gdf.empty:
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


@pytest.fixture
def nodes_gdf_no_crs(sample_nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """A nodes GeoDataFrame with no CRS."""
    gdf = sample_nodes_gdf.copy()
    gdf.crs = None
    return gdf


@pytest.fixture
def sample_nodes_gdf_alt_crs(sample_nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """A nodes GeoDataFrame with an alternative CRS."""
    gdf = sample_nodes_gdf.copy()
    gdf.crs = "EPSG:4326"
    return gdf


@pytest.fixture
def segments_gdf_alt_geom(sample_segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sample segments GDF with an alternative 'barrier_geometry' column."""
    gdf = sample_segments_gdf.copy()
    if not gdf.empty:
        # Create simple circular buffers around centroids as alternative geometries
        alt_geoms = [pt.buffer(0.2) for pt in gdf.geometry.centroid]
        gdf["barrier_geometry"] = gpd.GeoSeries(alt_geoms, crs=gdf.crs)
    else:
        # Ensure the column exists even if empty, with correct dtype and CRS
        gdf["barrier_geometry"] = gpd.GeoSeries([], dtype="geometry", crs=gdf.crs)
    return gdf


@pytest.fixture
def segments_gdf_with_custom_barrier(sample_segments_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Sample segments GDF (first segment) with a 'custom_barrier' column."""
    if sample_segments_gdf.empty:
        # Handle empty input: create the column with correct dtype and CRS
        segments_gdf_copy = sample_segments_gdf.copy()
        segments_gdf_copy["custom_barrier"] = gpd.GeoSeries(
            [], dtype="geometry", crs=segments_gdf_copy.crs,
        )
        return segments_gdf_copy

    segments_gdf = sample_segments_gdf.iloc[[0]].copy()  # Take only the first segment
    # Create a plausible barrier polygon based on the first segment's bounds
    first_segment_bounds = segments_gdf.geometry.iloc[0].bounds
    minx, miny, maxx, maxy = first_segment_bounds
    # Create a slightly larger bounding box as the barrier
    barrier_poly = Polygon(
        [
            (minx - 1, miny - 1),
            (minx - 1, maxy + 1),
            (maxx + 1, maxy + 1),
            (maxx + 1, miny - 1),
        ],
    )
    segments_gdf["custom_barrier"] = gpd.GeoSeries(
        [barrier_poly], crs=segments_gdf.crs,
    )
    return segments_gdf


@pytest.fixture
def buildings_invalid_geom_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Synthetic buildings GDF with invalid (LineString) geometry."""
    lines = [LineString([(0, 0), (1, 1)])]
    attrs = [{"bldg_id": "InvalidLineGeom"}]  # Ensure bldg_id for consistency
    return _create_gdf(lines, attrs, crs=sample_crs)


@pytest.fixture
def segments_invalid_geom_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Synthetic segments GDF with invalid (Polygon) geometry."""
    polys = [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]
    attrs = [{"seg_id": "InvalidPolyGeom"}]  # Ensure seg_id for consistency
    return _create_gdf(polys, attrs, crs=sample_crs)


@pytest.fixture
def not_a_gdf() -> pd.DataFrame:
    """Return a regular DataFrame for testing error cases."""
    return pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})


# Pytest skipif marker for tests requiring torch
requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch or PyTorch Geometric is not available.",
)

# --- Fixtures for validate_nx testing ---

@pytest.fixture
def sample_nx_multigraph() -> nx.MultiGraph:
    """Fixture for a NetworkX MultiGraph."""
    graph = nx.MultiGraph()
    graph.add_node(1, feature1=10.0, label1=0, pos=(0, 0), geometry=Point(0, 0))
    graph.add_node(2, feature1=20.0, label1=1, pos=(1, 1), geometry=Point(1, 1))
    graph.add_edge(1, 2, key=0, edge_feature1=0.5, geometry=LineString([(0, 0), (1, 1)]))
    graph.add_edge(1, 2, key=1, edge_feature1=0.8, geometry=LineString([(0, 0), (1, 1)]))
    graph.graph["crs"] = "EPSG:27700"
    graph.graph["is_hetero"] = False
    return graph


@pytest.fixture
def sample_nx_digraph() -> nx.DiGraph:
    """Fixture for a NetworkX DiGraph."""
    graph = nx.DiGraph()
    graph.add_node(1, feature1=10.0, pos=(0, 0), geometry=Point(0, 0))
    graph.add_node(2, feature1=20.0, pos=(1, 1), geometry=Point(1, 1))
    graph.add_edge(1, 2, edge_feature1=0.5, geometry=LineString([(0, 0), (1, 1)]))
    graph.graph["crs"] = "EPSG:27700"
    graph.graph["is_hetero"] = False
    return graph


@pytest.fixture
def sample_nx_multidigraph() -> nx.MultiDiGraph:
    """Fixture for a NetworkX MultiDiGraph."""
    graph = nx.MultiDiGraph()
    graph.add_node(1, feature1=10.0, pos=(0, 0), geometry=Point(0, 0))
    graph.add_node(2, feature1=20.0, pos=(1, 1), geometry=Point(1, 1))
    graph.add_edge(1, 2, key=0, edge_feature1=0.5, geometry=LineString([(0, 0), (1, 1)]))
    graph.add_edge(1, 2, key=1, edge_feature1=0.8, geometry=LineString([(0, 0), (1, 1)]))
    graph.graph["crs"] = "EPSG:27700"
    graph.graph["is_hetero"] = False
    return graph


@pytest.fixture
def nx_graph_missing_graph_attr() -> nx.Graph:
    """Fixture for a NetworkX graph missing the graph attribute dictionary."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_node(2, pos=(1, 1))
    graph.add_edge(1, 2)
    # Remove the graph attribute entirely
    delattr(graph, "graph")
    return graph


@pytest.fixture
def nx_graph_non_dict_graph_attr() -> nx.Graph:
    """Fixture for a NetworkX graph with non-dict graph attribute."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_node(2, pos=(1, 1))
    graph.add_edge(1, 2)
    graph.graph = "not_a_dict"  # Invalid type
    return graph


@pytest.fixture
def nx_graph_missing_is_hetero() -> nx.Graph:
    """Fixture for a NetworkX graph missing is_hetero metadata."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_node(2, pos=(1, 1))
    graph.add_edge(1, 2)
    graph.graph = {"crs": "EPSG:27700"}  # Missing is_hetero
    return graph


@pytest.fixture
def nx_graph_missing_crs() -> nx.Graph:
    """Fixture for a NetworkX graph missing crs metadata."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))
    graph.add_node(2, pos=(1, 1))
    graph.add_edge(1, 2)
    graph.graph = {"is_hetero": False}  # Missing crs
    return graph


@pytest.fixture
def nx_graph_no_pos_no_geom() -> nx.Graph:
    """Fixture for a NetworkX graph with nodes missing pos and geometry attributes."""
    graph = nx.Graph()
    graph.add_node(1, feature1=10.0)  # No pos or geometry
    graph.add_node(2, feature1=20.0)  # No pos or geometry
    graph.add_edge(1, 2)
    graph.graph = {"is_hetero": False, "crs": "EPSG:27700"}
    return graph


@pytest.fixture
def nx_graph_with_pos_only() -> nx.Graph:
    """Fixture for a NetworkX graph with nodes having only pos attributes."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0), feature1=10.0)
    graph.add_node(2, pos=(1, 1), feature1=20.0)
    graph.add_edge(1, 2)
    graph.graph = {"is_hetero": False, "crs": "EPSG:27700"}
    return graph


@pytest.fixture
def nx_graph_with_geometry_only() -> nx.Graph:
    """Fixture for a NetworkX graph with nodes having only geometry attributes."""
    graph = nx.Graph()
    graph.add_node(1, geometry=Point(0, 0), feature1=10.0)
    graph.add_node(2, geometry=Point(1, 1), feature1=20.0)
    graph.add_edge(1, 2)
    graph.graph = {"is_hetero": False, "crs": "EPSG:27700"}
    return graph


@pytest.fixture
def nx_hetero_graph_valid() -> nx.Graph:
    """Fixture for a valid heterogeneous NetworkX graph."""
    graph = nx.Graph()
    graph.add_node(1, node_type="building", pos=(0, 0), geometry=Point(0, 0))
    graph.add_node(2, node_type="road", pos=(1, 1), geometry=Point(1, 1))
    graph.add_edge(1, 2, edge_type="connects_to")
    graph.graph = {
        "is_hetero": True,
        "crs": "EPSG:27700",
        "node_types": ["building", "road"],
        "edge_types": [("building", "connects_to", "road")],
    }
    return graph


@pytest.fixture
def nx_hetero_graph_missing_node_types() -> nx.Graph:
    """Fixture for a heterogeneous graph missing node_types metadata."""
    graph = nx.Graph()
    graph.add_node(1, node_type="building", pos=(0, 0))
    graph.add_node(2, node_type="road", pos=(1, 1))
    graph.add_edge(1, 2, edge_type="connects_to")
    graph.graph = {
        "is_hetero": True,
        "crs": "EPSG:27700",
        "edge_types": [("building", "connects_to", "road")],
        # Missing node_types
    }
    return graph


@pytest.fixture
def nx_hetero_graph_empty_node_types() -> nx.Graph:
    """Fixture for a heterogeneous graph with empty node_types."""
    graph = nx.Graph()
    graph.add_node(1, node_type="building", pos=(0, 0))
    graph.add_node(2, node_type="road", pos=(1, 1))
    graph.add_edge(1, 2, edge_type="connects_to")
    graph.graph = {
        "is_hetero": True,
        "crs": "EPSG:27700",
        "node_types": [],  # Empty list
        "edge_types": [("building", "connects_to", "road")],
    }
    return graph


@pytest.fixture
def nx_hetero_graph_missing_edge_types() -> nx.Graph:
    """Fixture for a heterogeneous graph missing edge_types metadata."""
    graph = nx.Graph()
    graph.add_node(1, node_type="building", pos=(0, 0))
    graph.add_node(2, node_type="road", pos=(1, 1))
    graph.add_edge(1, 2, edge_type="connects_to")
    graph.graph = {
        "is_hetero": True,
        "crs": "EPSG:27700",
        "node_types": ["building", "road"],
        # Missing edge_types
    }
    return graph


@pytest.fixture
def nx_hetero_graph_missing_node_type_attr() -> nx.Graph:
    """Fixture for a heterogeneous graph with nodes missing node_type attribute."""
    graph = nx.Graph()
    graph.add_node(1, pos=(0, 0))  # Missing node_type
    graph.add_node(2, node_type="road", pos=(1, 1))
    graph.add_edge(1, 2, edge_type="connects_to")
    graph.graph = {
        "is_hetero": True,
        "crs": "EPSG:27700",
        "node_types": ["building", "road"],
        "edge_types": [("building", "connects_to", "road")],
    }
    return graph


@pytest.fixture
def nx_hetero_graph_missing_edge_type_attr() -> nx.Graph:
    """Fixture for a heterogeneous graph with edges missing edge_type attribute."""
    graph = nx.Graph()
    graph.add_node(1, node_type="building", pos=(0, 0))
    graph.add_node(2, node_type="road", pos=(1, 1))
    graph.add_edge(1, 2)  # Missing edge_type
    graph.graph = {
        "is_hetero": True,
        "crs": "EPSG:27700",
        "node_types": ["building", "road"],
        "edge_types": [("building", "connects_to", "road")],
    }
    return graph


@pytest.fixture
def invalid_input_string() -> str:
    """Fixture for invalid input (string instead of graph)."""
    return "not_a_graph"


@pytest.fixture
def invalid_input_dict() -> dict:
    """Fixture for invalid input (dict instead of graph)."""
    return {"not": "a_graph"}


@pytest.fixture
def invalid_input_list() -> list:
    """Fixture for invalid input (list instead of graph)."""
    return [1, 2, 3]


@pytest.fixture
def p2pub_private_single_cell(sample_crs: str) -> gpd.GeoDataFrame:
    """A single private cell for private_to_public tests."""
    poly = Polygon([(0.4, 0.4), (0.4, 0.6), (0.6, 0.6), (0.6, 0.4)])
    return _create_gdf([poly], [{"private_id": 0}], crs=sample_crs)


@pytest.fixture
def p2pub_public_single_segment(sample_crs: str) -> gpd.GeoDataFrame:
    """A single public segment for private_to_public tests."""
    line = LineString([(0.5, 0.3), (0.5, 0.7)])
    return _create_gdf([line], [{"public_id": 10}], crs=sample_crs)


@pytest.fixture
def p2p_isolated_polys_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """A GDF with polygons that are not contiguous, for testing private_to_private_graph."""
    polys = [
        Polygon([(0, 0), (0, 1), (1, 1), (1, 0)]),
        Polygon([(10, 10), (10, 11), (11, 11), (11, 10)]),
        Polygon([(20, 20), (20, 21), (21, 21), (21, 20)]),
    ]
    attrs = [{"private_id": i} for i in range(len(polys))]
    return _create_gdf(polys, attrs, crs=sample_crs)


@pytest.fixture
def segments_gdf_with_multiindex_public_id(sample_crs: str) -> gpd.GeoDataFrame:
    """A GDF with segments and a MultiIndex for public_id, for public_to_public_graph."""
    data = {
        "seg_id": ["s1", "s2", "s3"],
        "geometry": [
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 0)]),
            LineString([(2, 0), (3, 0)]),
        ],
    }
    gdf = gpd.GeoDataFrame(data, crs=sample_crs)
    # Create a dummy MultiIndex for public_id
    gdf["public_id"] = pd.MultiIndex.from_tuples([("A", 1), ("B", 2), ("C", 3)], names=["type", "idx"])
    return gdf


@pytest.fixture
def empty_nodes_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Return an empty nodes GeoDataFrame for testing edge cases."""
    return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=sample_crs)


@pytest.fixture
def single_node_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with a single node for testing edge cases."""
    data = {
        "node_id": [1],
        "feature1": [10.0],
        "geometry": [Point(0, 0)],
    }
    return gpd.GeoDataFrame(data, crs=sample_crs).set_index("node_id")


@pytest.fixture
def two_nodes_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with two nodes for testing edge cases."""
    data = {
        "node_id": [1, 2],
        "feature1": [10.0, 20.0],
        "geometry": [Point(0, 0), Point(1, 1)],
    }
    return gpd.GeoDataFrame(data, crs=sample_crs).set_index("node_id")


@pytest.fixture
def coincident_nodes_gdf(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame with coincident nodes for testing edge cases."""
    data = {
        "node_id": [1, 2, 3],
        "feature1": [10.0, 20.0, 30.0],
        "geometry": [Point(0, 0), Point(0, 0), Point(1, 1)],  # Two coincident points
    }
    return gpd.GeoDataFrame(data, crs=sample_crs).set_index("node_id")


@pytest.fixture
def network_gdf_with_pos(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a network GeoDataFrame with proper pos attributes for testing."""
    from city2graph.utils import gdf_to_nx
    from city2graph.utils import nx_to_gdf

    # Create a simple network
    data = {
        "edge_id": ["e1", "e2"],
        "source_id": [1, 2],
        "target_id": [2, 3],
        "geometry": [
            LineString([(0, 0), (1, 1)]),
            LineString([(1, 1), (2, 2)]),
        ],
    }
    multi_index = pd.MultiIndex.from_arrays(
        [data["source_id"], data["target_id"]], names=("source_id", "target_id"),
    )
    edges_gdf = gpd.GeoDataFrame(data, index=multi_index, crs=sample_crs)

    # Convert to NetworkX and back to ensure pos attributes are added
    G = gdf_to_nx(edges=edges_gdf)
    _, edges_with_pos = nx_to_gdf(G, nodes=True, edges=True)

    return edges_with_pos


@pytest.fixture
def network_gdf_no_pos(sample_crs: str) -> gpd.GeoDataFrame:
    """Return a network GeoDataFrame without pos attributes for testing error cases."""
    data = {
        "edge_id": ["e1"],
        "source_id": [1],
        "target_id": [2],
        "geometry": [LineString([(0, 0), (1, 1)])],
    }
    multi_index = pd.MultiIndex.from_arrays(
        [data["source_id"], data["target_id"]], names=("source_id", "target_id"),
    )
    return gpd.GeoDataFrame(data, index=multi_index, crs=sample_crs)
