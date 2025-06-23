"""Test fixtures and configuration for the city2graph test suite."""

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
        "source_road_id": ["r1"],
        "target_road_id": ["r2"],
        "link_feat1": [0.7],
        "geometry": [LineString([(10, 12), (12, 12)])],
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
def nodes_gdf_no_crs(sample_nodes_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """A nodes GeoDataFrame with no CRS."""
    gdf = sample_nodes_gdf.copy()
    gdf.crs = None
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
