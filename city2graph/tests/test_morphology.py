"""Comprehensive tests for morphology.py module.

This module contains unit tests for all functions in the morphology module,
including input validation, geometry processing, and graph creation functionality.
"""

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString
from shapely.geometry import MultiLineString
from shapely.geometry import Point
from shapely.geometry import Polygon

from city2graph import morphology

# ============================================================================
# COMMON TEST FIXTURES
# ============================================================================


def make_standard_priv_pub() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create a standard private polygon and public line GeoDataFrames for testing."""
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    line = LineString([(-1, 1), (3, 1)])
    priv = gpd.GeoDataFrame({"idx": [10]}, geometry=[poly], crs="EPSG:4326")
    pub = gpd.GeoDataFrame({"id": ["L1"]}, geometry=[line], crs="EPSG:4326")
    return priv, pub


@pytest.fixture(name="standard_priv_pub")
def fixture_standard_priv_pub() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Fixture providing a standard private-public geometry pair for testing."""
    return make_standard_priv_pub()


@pytest.fixture
def sample_polygon() -> Polygon:
    """Return a unit square polygon for testing."""
    return Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


@pytest.fixture
def sample_linestring() -> LineString:
    """Return a simple horizontal line for testing."""
    return LineString([(0, 0), (1, 0)])


@pytest.fixture
def empty_geodataframe() -> gpd.GeoDataFrame:
    """Return an empty GeoDataFrame for testing edge cases."""
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


@pytest.fixture
def adjacent_polygons() -> gpd.GeoDataFrame:
    """Return two adjacent polygons for contiguity testing."""
    poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    poly2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    return gpd.GeoDataFrame(
        {"id": [1, 2]},
        geometry=[poly1, poly2],
        crs="EPSG:4326",
    )


# ============================================================================
# INPUT VALIDATION AND ERROR HANDLING TESTS
# ============================================================================


def test_validate_inputs_and_empty_check_errors() -> None:
    """Test input validation and empty dataframe checks."""
    # Arrange: create empty and valid GeoDataFrames for testing
    empty = gpd.GeoDataFrame()
    valid = gpd.GeoDataFrame(geometry=[Point(0, 0)])

    # Act & Assert: invalid types should raise TypeError
    with pytest.raises(TypeError):
        morphology._validate_inputs(1, valid, 1)

    with pytest.raises(TypeError):
        morphology._validate_inputs(valid, 1, 1)

    with pytest.raises(TypeError):
        morphology._validate_inputs(valid, valid, "buf")

    # Act & Assert: empty DataFrames should trigger warnings
    with pytest.warns(RuntimeWarning):
        assert morphology._check_empty_dataframes(empty, valid)

    with pytest.warns(RuntimeWarning):
        assert morphology._check_empty_dataframes(valid, empty)

    # Assert: non-empty inputs produce no warnings
    assert not morphology._check_empty_dataframes(valid, valid)


def test_validate_columns_and_geometries() -> None:
    """Test column validation and geometry checks."""
    # Arrange: create sample privates and publics
    priv = gpd.GeoDataFrame(
        {"a": [1]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
    )
    pub = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[LineString([(0, 0), (1, 1)])],
    )

    # Act & Assert: missing public_id column raises ValueError
    with pytest.raises(ValueError, match="public_id_col 'id' not found"):
        morphology._validate_columns(pub.drop(columns=["id"]), priv, "id", None, None)

    # Act & Assert: missing private_id column raises ValueError
    with pytest.raises(ValueError, match="private_id_col 'a' not found in privates"):
        morphology._validate_columns(
            pub,
            priv.drop(columns=[priv.columns[0]]),
            "id",
            "a",
            None,
        )

    # Act & Assert: missing public_geom column raises ValueError
    with pytest.raises(ValueError, match="public_geom_col 'geom' not found in publics"):
        morphology._validate_columns(
            pub,
            priv,
            "id",
            None,
            "geom",
        )

    # Act & Assert: invalid geometry types issue warnings
    with pytest.warns(RuntimeWarning):
        # Invalid geometry types in privates
        morphology._validate_geometries(priv.assign(geometry=[Point(0, 0)]), pub)

    with pytest.warns(RuntimeWarning):
        # Invalid geometry types in publics
        morphology._validate_geometries(priv, pub.assign(geometry=[Point(0, 0)]))


# ============================================================================
# ADJACENT PUBLICS AND GEOMETRY PROCESSING TESTS
# ============================================================================


def test_get_adjacent_publics_basic_and_defaults(
    standard_priv_pub: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame],
) -> None:
    """Test the function that retrieves adjacent public geometries."""
    # Arrange: get privates and publics from fixture
    priv, pub = standard_priv_pub

    # Act: find adjacent publics using default parameters
    adj = morphology._get_adjacent_publics(priv, pub)

    # Assert: expect default index mapping to single id
    assert adj == {0: ["L1"]}

    # Act & Assert: missing id columns should raise ValueError
    with pytest.raises(ValueError, match="public_id_col 'None' not found in publics"):
        morphology._get_adjacent_publics(
            priv,
            pub,
            public_id_col=None,
            private_id_col=None,
        )

    # Arrange: create empty publics DataFrame
    empty = gpd.GeoDataFrame(columns=["id"], geometry=[], crs="EPSG:4326")

    # Act & Assert: empty publics DataFrame returns empty dict
    assert morphology._get_adjacent_publics(priv, empty) == {}

    # Arrange: create publics with different CRS
    pub2 = pub.copy().set_crs("EPSG:3857", allow_override=True)

    # Act: test CRS mismatch handling
    adj2 = morphology._get_adjacent_publics(priv, pub2)

    # Assert: should return valid dictionary despite CRS difference
    assert isinstance(adj2, dict)


def test_extract_nodes_and_connections() -> None:
    """Test extraction of dual graph nodes and connections."""
    # Arrange: build a simple dual graph with two nodes
    g = nx.Graph()
    g.add_node(1, id=100)
    g.add_node(2, id=200)
    g.add_edge(1, 2)

    # Act & Assert: extract nodes with non-coordinate index should error
    with pytest.raises(TypeError):
        _ = morphology._extract_dual_graph_nodes(g, "id", "EPSG:4326")

    # Arrange: define an empty graph
    empty_nodes = morphology._extract_dual_graph_nodes(nx.Graph(), "id", None)

    # Assert: empty GeoDataFrame with proper columns is returned
    assert empty_nodes.empty
    assert list(empty_nodes.columns) == ["id", "geometry"]


def test_find_additional_connections() -> None:
    """Test finding additional connections between geometries."""
    # Arrange: two line segments with close endpoints
    l1 = LineString([(0, 0), (1, 0)])
    l2 = LineString([(1.001, 0), (2, 0)])
    test_lines = gpd.GeoDataFrame({"id": ["A", "B"]}, geometry=[l1, l2], crs="EPSG:4326")

    # Act: find additional connections based on endpoint proximity
    conns = morphology._find_additional_connections(test_lines, "id", 0.01)

    # Assert: both IDs should be connected
    assert set(conns.keys()) == {"A", "B"}

    # Arrange: MultiLineString should be ignored
    mls = MultiLineString([[(0, 0), (1, 1)], [(1, 1), (2, 2)]])
    multi_line_gdf = gpd.GeoDataFrame({"id": ["X"]}, geometry=[mls], crs="EPSG:4326")

    # Act & Assert: multi-line geometries produce no connections
    assert morphology._find_additional_connections(multi_line_gdf, "id", 1, {}) == {}


# ============================================================================
# DUAL GRAPH CONVERSION AND PROCESSING TESTS
# ============================================================================


def test_convert_gdf_to_dual_errors_and_empty() -> None:
    """Test conversion of GeoDataFrame to dual graph with error handling."""
    # Act & Assert: invalid input type raises TypeError
    with pytest.raises(TypeError):
        morphology.convert_gdf_to_dual(123)

    # Act & Assert: invalid tolerance type raises TypeError
    with pytest.raises(TypeError):
        morphology.convert_gdf_to_dual(gpd.GeoDataFrame(), tolerance="x")

    # Act & Assert: negative tolerance raises ValueError
    with pytest.raises(ValueError, match="Tolerance must be non-negative"):
        morphology.convert_gdf_to_dual(
            gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])]),
            tolerance=-1,
        )

    # Arrange: empty GeoDataFrame
    empty_gdf = gpd.GeoDataFrame(geometry=[])

    # Act: convert empty GeoDataFrame to dual graph
    nodes, conns = morphology.convert_gdf_to_dual(empty_gdf)

    # Assert: empty outputs returned
    assert isinstance(nodes, gpd.GeoDataFrame)
    assert conns == {}


# ============================================================================
# CONNECTING LINES AND CONTIGUITY TESTS
# ============================================================================


def test_create_connecting_lines_and_errors() -> None:
    """Test creation of connecting lines between private and public geometries."""
    # Arrange: sample private polygon and public point
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    priv = gpd.GeoDataFrame({"pid": [0]}, geometry=[poly], crs="EPSG:4326")
    pub = gpd.GeoDataFrame({"id": [0]}, geometry=[Point(0.5, 1.5)], crs="EPSG:4326")

    # Act: create connecting lines for valid inputs
    lines = morphology._create_connecting_lines(priv, pub, {0: [0]})

    # Assert: should create one connecting line
    assert len(lines) == 1

    # Act & Assert: type errors for invalid inputs
    with pytest.raises(TypeError):
        morphology._create_connecting_lines("bad", pub, {})

    with pytest.raises(TypeError):
        morphology._create_connecting_lines(priv, "bad", {})

    with pytest.raises(TypeError):
        morphology._create_connecting_lines(priv, pub, "bad")


def test_prep_and_private_private() -> None:
    """Test preparation of contiguity graph and private-to-private graph creation."""
    # Arrange: single polygon yields no graph
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    single = gpd.GeoDataFrame({"id": [1]}, geometry=[poly], crs="EPSG:4326")

    # Act: prepare contiguity graph for single polygon
    gnx, m = morphology._prep_contiguity_graph(single, "id", "queen")

    # Assert: no graph or mapping returned for single polygon
    assert gnx is None
    assert m is None

    # Arrange: two adjacent polygons with rook contiguity
    poly2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    adjacent_polys = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[poly, poly2], crs="EPSG:4326")

    # Act: prepare contiguity graph for adjacent polygons
    g2, m2 = morphology._prep_contiguity_graph(adjacent_polys, None, "rook")

    # Assert: valid graph and mapping returned
    assert isinstance(g2, nx.Graph)
    assert isinstance(m2, dict)

    # Act & Assert: invalid private_to_private_graph type
    with pytest.raises(TypeError):
        morphology.private_to_private_graph("bad")

    # Arrange: empty private GeoDataFrame
    empty = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Act & Assert: empty input returns empty result
    assert morphology.private_to_private_graph(empty).empty

    # Act & Assert: invalid contiguity parameter
    with pytest.raises(ValueError, match="contiguity must be"):
        morphology.private_to_private_graph(adjacent_polys, contiguity="invalid")


def test_private_to_public_and_public_to_public() -> None:
    """Test conversion from private to public graph and public to public graph."""
    # Arrange: empty private gdf
    empty_priv = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    pub = gpd.GeoDataFrame({"id": [1]}, geometry=[LineString([(0, 0), (1, 1)])], crs="EPSG:4326")

    # Act & Assert: empty private input returns empty result
    assert morphology.private_to_public_graph(empty_priv, pub).empty

    # Arrange: simple private polygon and conversion parameters
    poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    priv = gpd.GeoDataFrame({"pid": ["P"]}, geometry=[poly], crs="EPSG:4326")

    # Act: convert private to public graph
    out = morphology.private_to_public_graph(priv, pub, private_id_col="pid", public_id_col="id")

    # Assert: output columns include private_id and public_id
    assert "private_id" in out.columns
    assert "public_id" in out.columns

    # Arrange: single segment for public_to_public_graph
    single = gpd.GeoDataFrame({"id": [1]}, geometry=[LineString([(0, 0), (1, 0)])], crs="EPSG:4326")

    # Act & Assert: single segment returns empty DataFrame
    assert morphology.public_to_public_graph(single).empty

    # Arrange: two close segments for public-to-public connections
    l1 = LineString([(0, 0), (1, 0)])
    l2 = LineString([(1.001, 0), (2, 0)])
    test_segments = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[l1, l2], crs="EPSG:4326")

    # Act: create public-to-public connections
    edges = morphology.public_to_public_graph(test_segments, public_id_col="id", tolerance=0.01)

    # Assert: edges DataFrame has proper columns
    assert "from_public_id" in edges.columns
    assert "to_public_id" in edges.columns


# ============================================================================
# END-TO-END FUNCTIONALITY TESTS
# ============================================================================


def test_morphological_graph_end_to_end() -> None:
    """Test end-to-end functionality of morphological graph creation."""
    # Arrange: minimal buildings and segments
    buildings = gpd.GeoDataFrame(
        {"tess_id": [1]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:4326",
    )
    segments = gpd.GeoDataFrame(
        {"barrier_geometry": [LineString([(0, 1), (1, 1)])]},
        geometry=[LineString([(0, 1), (1, 1)])],
        crs="EPSG:4326",
    )

    # Act & Assert: morphological_graph with invalid inputs raises error
    with pytest.raises(AttributeError):
        morphology.morphological_graph(buildings, segments)

    # Arrange: segments without barrier_geometry column
    segments2 = segments.drop(columns=["barrier_geometry"])

    # Act & Assert: missing barrier_geometry column causes concatenation error
    with pytest.raises(ValueError, match="No objects to concatenate"):
        morphology.morphological_graph(buildings, segments2)
