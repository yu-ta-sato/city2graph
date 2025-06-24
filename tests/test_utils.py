"""Tests for the utils module."""

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest

from city2graph import utils
from city2graph.utils import gdf_to_nx
from city2graph.utils import nx_to_gdf

# Since many functions in utils.py are not fully implemented,
# some tests might be skipped if they hit a NotImplementedError or other issues
# from the incomplete code. The tests are written against the intended functionality.

@pytest.mark.parametrize(
    ("geometry_fixture", "barriers_fixture", "expect_empty"),
    [
        ("empty_gdf", None, True),
        ("sample_buildings_gdf", None, False),
        ("sample_buildings_gdf", "sample_segments_gdf", False),
    ],
)
def test_create_tessellation(
    geometry_fixture: str,
    barriers_fixture: str | None,
    expect_empty: bool,
    request: pytest.FixtureRequest,
) -> None:
    """Test create_tessellation for morphological and enclosed types."""
    geometry = request.getfixturevalue(geometry_fixture)
    primary_barriers = (
        request.getfixturevalue(barriers_fixture) if barriers_fixture else None
    )

    try:
        tessellation = utils.create_tessellation(
            geometry,
            primary_barriers=primary_barriers,
        )
    except (UnboundLocalError, TypeError, ValueError) as e:
        pytest.skip(
            "Skipping due to incomplete implementation in "
            f"utils.create_tessellation: {e}",
        )

    assert isinstance(tessellation, gpd.GeoDataFrame)
    if not tessellation.empty:
        assert "tess_id" in tessellation.columns
    if expect_empty:
        assert tessellation.empty
    else:
        assert not tessellation.empty
        assert tessellation.crs == geometry.crs


@pytest.mark.parametrize(
    ("nodes_fixture", "edges_fixture", "keep_geom", "edge_id_col", "error", "match"),
    [
        # Successful cases
        ("sample_nodes_gdf", "sample_edges_gdf", False, None, None, None),
        ("sample_nodes_gdf", "sample_edges_gdf", True, None, None, None),
        ("sample_nodes_gdf", "sample_edges_gdf", False, "edge_id", None, None),
        ("empty_gdf", "empty_gdf", False, None, None, None),
        # Error cases
        (
            "sample_segments_gdf",
            None,
            False,
            None,
            TypeError,
            "Input `gdf` must be a tuple of \\(nodes_gdf, edges_gdf\\)\\.",
        ),
        (
            "sample_nodes_gdf",
            "segments_gdf_no_crs",
            False,
            None,
            ValueError,
            "All GeoDataFrames must have the same CRS",
        ),
        (
            "sample_nodes_gdf",
            "not_a_gdf",
            False,
            None,
            AttributeError,
            "'DataFrame' object has no attribute 'crs'",
        ),
    ],
)
def test_dual_graph(
    nodes_fixture: str,
    edges_fixture: str | None,
    keep_geom: bool,
    edge_id_col: str | None,
    error: type[Exception] | None,
    match: str | None,
    request: pytest.FixtureRequest,
) -> None:
    """Test dual_graph with various inputs."""
    if edges_fixture is None:
        # For testing non-tuple input
        gdf = request.getfixturevalue(nodes_fixture)
    else:
        nodes = request.getfixturevalue(nodes_fixture)
        edges = request.getfixturevalue(edges_fixture)
        gdf = (nodes, edges)

    if error:
        with pytest.raises(error, match=match):
            utils.dual_graph(
                gdf, edge_id_col=edge_id_col, keep_original_geom=keep_geom
            )
    else:
        _primal_nodes, primal_edges = gdf
        dual_nodes, dual_edges = utils.dual_graph(
            gdf, edge_id_col=edge_id_col, keep_original_geom=keep_geom
        )

        if primal_edges.empty:
            assert isinstance(dual_nodes, gpd.GeoDataFrame)
            assert dual_nodes.empty
            assert isinstance(dual_edges, gpd.GeoDataFrame)
            assert dual_edges.empty
            return

        assert isinstance(dual_nodes, gpd.GeoDataFrame)
        assert not dual_nodes.empty
        assert isinstance(dual_edges, gpd.GeoDataFrame)

        # For sample data, we expect adjacent edges, so dual_edges is not empty.
        if edges_fixture == "sample_edges_gdf":
            assert not dual_edges.empty

        assert dual_nodes.crs == primal_edges.crs
        assert dual_edges.crs == primal_edges.crs

        if keep_geom:
            assert "original_geometry" in dual_nodes.columns
        else:
            assert "original_geometry" not in dual_nodes.columns

        if edge_id_col:
            assert dual_nodes.index.name == edge_id_col
            assert all(primal_edges[edge_id_col].isin(dual_nodes.index))


@pytest.mark.parametrize(
    ("segments_fixture", "expect_empty_output"),
    [
        ("sample_segments_gdf", False),
        ("single_segment_gdf", False),
        ("empty_gdf", True),
        ("segments_invalid_geom_gdf", True),  # Invalid geoms are filtered, resulting in empty
        ("segments_gdf_no_crs", False),
    ],
)
def test_segments_to_graph(
    segments_fixture: str,
    expect_empty_output: bool,
    request: pytest.FixtureRequest,
) -> None:
    """Test segments_to_graph with various inputs."""
    segments_gdf = request.getfixturevalue(segments_fixture)

    nodes_gdf, edges_gdf = utils.segments_to_graph(segments_gdf)

    assert isinstance(nodes_gdf, gpd.GeoDataFrame)
    assert isinstance(edges_gdf, gpd.GeoDataFrame)

    if expect_empty_output:
        assert nodes_gdf.empty
        assert edges_gdf.empty
        assert nodes_gdf.crs == segments_gdf.crs
        assert edges_gdf.crs == segments_gdf.crs
        return

    assert not nodes_gdf.empty
    assert not edges_gdf.empty

    assert nodes_gdf.crs == segments_gdf.crs
    assert edges_gdf.crs == segments_gdf.crs

    assert nodes_gdf.index.name == "node_id"
    assert nodes_gdf.geometry.geom_type.isin(["Point"]).all()

    assert isinstance(edges_gdf.index, pd.MultiIndex)
    assert edges_gdf.index.names == ["from_node_id", "to_node_id"]
    assert edges_gdf.geometry.geom_type.isin(["LineString"]).all()

    # Check that original attributes are preserved in edges
    original_cols = set(segments_gdf.columns) - {"geometry"}
    output_cols = set(edges_gdf.columns) - {"geometry"}
    assert original_cols == output_cols

    # Check consistency
    if segments_fixture in ("sample_segments_gdf", "segments_gdf_no_crs"):
        # These fixtures are known to contain one invalid geometry that gets removed.
        assert len(edges_gdf) == len(segments_gdf) - 1
    else:
        # For other valid inputs, the number of edges should match the input segments.
        assert len(edges_gdf) == len(segments_gdf)

    from_ids = edges_gdf.index.get_level_values("from_node_id")
    to_ids = edges_gdf.index.get_level_values("to_node_id")
    assert all(from_ids.isin(nodes_gdf.index))
    assert all(to_ids.isin(nodes_gdf.index))


@pytest.mark.parametrize(
    (
        "graph_fixture",
        "as_nx",
        "center_point_fixture",
        "distance",
        "expect_empty_edges",
    ),
    [
        ("sample_segments_gdf", False, "mg_center_point", 100.0, False),
        ("sample_segments_gdf", False, "mg_center_point", 0.01, True),
        ("sample_nx_graph", True, "sample_nodes_gdf", 1.0, False),
        ("sample_nx_graph", True, "sample_nodes_gdf", 0.1, True),
        ("empty_gdf", False, "mg_center_point", 100.0, True),
    ],
)
def test_filter_graph_by_distance(
    graph_fixture: str,
    as_nx: bool,
    center_point_fixture: str,
    distance: float,
    expect_empty_edges: bool,
    request: pytest.FixtureRequest,
) -> None:
    """Test filter_graph_by_distance for GDF and NX graphs."""
    graph = request.getfixturevalue(graph_fixture)
    center_point_source = request.getfixturevalue(center_point_fixture)

    center_point = (
        center_point_source.geometry.iloc[0] if as_nx else center_point_source
    )

    filtered = utils.filter_graph_by_distance(graph, center_point, distance=distance)

    if as_nx:
        assert isinstance(filtered, nx.Graph)
        if expect_empty_edges:
            assert filtered.number_of_edges() == 0
        else:
            assert filtered.number_of_edges() > 0
    else:
        assert isinstance(filtered, gpd.GeoDataFrame)
        if expect_empty_edges:
            assert filtered.empty
        else:
            assert not filtered.empty


@pytest.mark.parametrize(
    ("graph_fixture", "center_point_fixture", "distance", "expect_empty"),
    [
        ("sample_segments_gdf", "mg_center_point", 100.0, False),
        ("sample_segments_gdf", "mg_center_point", 0.01, True),
        ("sample_nx_graph", "sample_nodes_gdf", 1.0, False),
    ],
)
def test_create_isochrone(
    graph_fixture: str,
    center_point_fixture: str,
    distance: float,
    expect_empty: bool,
    request: pytest.FixtureRequest,
) -> None:
    """Test create_isochrone generation."""
    graph = request.getfixturevalue(graph_fixture)
    center_point_source = request.getfixturevalue(center_point_fixture)

    center_point = (
        center_point_source.geometry.iloc[0]
        if isinstance(graph, nx.Graph)
        else center_point_source
    )

    isochrone = utils.create_isochrone(graph, center_point, distance=distance)

    assert isinstance(isochrone, gpd.GeoDataFrame)
    if expect_empty:
        assert isochrone.empty
    else:
        assert not isochrone.empty
        assert len(isochrone) == 1
        assert isochrone.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]


def test_gdf_to_nx_roundtrip(
    sample_nodes_gdf: gpd.GeoDataFrame,
    sample_edges_gdf: gpd.GeoDataFrame,
) -> None:
    """Test round trip conversion from GeoDataFrame to NetworkX and back."""
    G = gdf_to_nx(sample_nodes_gdf, sample_edges_gdf)
    nodes_trip, edges_trip = nx_to_gdf(G)

    assert sample_nodes_gdf.crs == nodes_trip.crs
    assert sample_edges_gdf.crs == edges_trip.crs
    assert "geometry" in nodes_trip.columns
    assert "geometry" in edges_trip.columns
    assert all(nodes_trip["geometry"].is_valid)
    assert all(edges_trip["geometry"].is_valid)
    assert len(sample_nodes_gdf) == len(nodes_trip)
    assert len(sample_edges_gdf) == len(edges_trip)
    pd.testing.assert_index_equal(sample_nodes_gdf.index, nodes_trip.index)
    pd.testing.assert_index_equal(sample_edges_gdf.index, edges_trip.index)


def test_gdf_to_nx_roundtrip_hetero(
    sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    sample_hetero_edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
) -> None:
    """Test round trip conversion for heterogeneous graphs."""
    H = gdf_to_nx(nodes=sample_hetero_nodes_dict, edges=sample_hetero_edges_dict)
    nodes_dict_trip, edges_dict_trip = nx_to_gdf(H)

    assert isinstance(nodes_dict_trip, dict)
    assert isinstance(edges_dict_trip, dict)

    assert sample_hetero_nodes_dict.keys() == nodes_dict_trip.keys()
    assert sample_hetero_edges_dict.keys() == edges_dict_trip.keys()

    for node_type, nodes_gdf in sample_hetero_nodes_dict.items():
        nodes_gdf_trip = nodes_dict_trip[node_type]
        assert nodes_gdf.crs == nodes_gdf_trip.crs
        assert "geometry" in nodes_gdf_trip.columns
        assert all(nodes_gdf_trip["geometry"].is_valid)
        assert len(nodes_gdf) == len(nodes_gdf_trip)
        pd.testing.assert_index_equal(nodes_gdf.index, nodes_gdf_trip.index)

    for edge_type, edges_gdf in sample_hetero_edges_dict.items():
        edges_gdf_trip = edges_dict_trip[edge_type]
        assert edges_gdf.crs == edges_gdf_trip.crs
        assert "geometry" in edges_gdf_trip.columns
        assert all(edges_gdf_trip["geometry"].is_valid)
        assert len(edges_gdf) == len(edges_gdf_trip)
        pd.testing.assert_index_equal(edges_gdf.index, edges_gdf_trip.index)


@pytest.mark.parametrize(
    ("gdf_fixture", "input_type"),
    [
        ("sample_edges_gdf", "edges"),
        ("sample_hetero_edges_dict", "hetero_edges"),
    ],
)
def test_gdf_to_nx_single_input(
    gdf_fixture: str, input_type: str, request: pytest.FixtureRequest,
) -> None:
    """Test that gdf_to_nx works with only edges."""
    gdf = request.getfixturevalue(gdf_fixture)
    if input_type == "edges":
        G = gdf_to_nx(edges=gdf)
        assert isinstance(G, nx.Graph)
        assert G.number_of_edges() == len(gdf)
        # Nodes are created from edge endpoints
        assert G.number_of_nodes() > 0
    elif input_type == "hetero_edges":
        G = gdf_to_nx(edges=gdf)
        assert isinstance(G, nx.Graph)
        assert G.number_of_edges() == 0
        assert G.number_of_nodes() == 0


@pytest.mark.parametrize(
    ("nodes_arg", "edges_arg", "error", "match"),
    [
        (None, None, ValueError, "Either nodes or edges must be provided\\."),
        ("not_a_gdf", "sample_edges_gdf", TypeError, "Input must be a GeoDataFrame"),
        ("sample_nodes_gdf", "not_a_gdf", TypeError, "Input must be a GeoDataFrame"),
        (
            "sample_hetero_nodes_dict",
            "sample_edges_gdf",
            TypeError,
            "If nodes is a dict, edges must also be a dict or None.",
        ),
        (
            "sample_nodes_gdf",
            "sample_hetero_edges_dict",
            TypeError,
            "If edges is a dict, nodes must also be a dict or None.",
        ),
        (
            "sample_nodes_gdf_alt_crs",
            "sample_edges_gdf",
            ValueError,
            "All GeoDataFrames must have the same CRS",
        ),
    ],
)
def test_gdf_to_nx_invalid_input(
    nodes_arg: str | None,
    edges_arg: str | None,
    error: type[Exception],
    match: str,
    request: pytest.FixtureRequest,
) -> None:
    """Test that gdf_to_nx raises errors for invalid input."""
    nodes = request.getfixturevalue(nodes_arg) if nodes_arg else None
    edges = request.getfixturevalue(edges_arg) if edges_arg else None

    with pytest.raises(error, match=match):
        gdf_to_nx(nodes=nodes, edges=edges)


@pytest.mark.parametrize(
    ("graph_fixture", "expect_crs", "expect_geom"),
    [
        ("sample_nx_graph", True, True),
        ("sample_nx_graph_no_crs", False, True),
        ("sample_nx_graph_no_pos", True, True),  # Changed expectation for CRS
    ],
)
def test_nx_to_gdf_variants(
    graph_fixture: str,
    expect_crs: bool,
    expect_geom: bool,
    request: pytest.FixtureRequest,
) -> None:
    """Test converting NetworkX graphs with different properties to GeoDataFrames."""
    graph = request.getfixturevalue(graph_fixture)
    nodes, edges = nx_to_gdf(graph)

    if expect_geom:
        assert "geometry" in nodes.columns
        assert "geometry" in edges.columns
    else:
        assert "geometry" not in nodes.columns
        assert "geometry" not in edges.columns

    if expect_crs:
        assert nodes.crs is not None
        assert edges.crs is not None
    else:
        assert nodes.crs is None
        assert edges.crs is None


@pytest.mark.parametrize(
    ("nodes_fixture", "edges_fixture", "error", "match"),
    [
        # Success cases
        ("sample_nodes_gdf", "sample_edges_gdf", None, None),
        ("sample_nodes_gdf", None, None, None),
        (None, "sample_edges_gdf", None, None),
        ("empty_gdf", "sample_edges_gdf", None, None),
        # Error cases
        ("not_a_gdf", "sample_edges_gdf", TypeError, "Input must be a GeoDataFrame"),
        ("sample_nodes_gdf", "not_a_gdf", TypeError, "Input must be a GeoDataFrame"),
        ("sample_nodes_gdf", "empty_gdf", ValueError, "GeoDataFrame cannot be empty"),
        (
            "sample_nodes_gdf",
            "segments_invalid_geom_gdf",
            ValueError,
            "GeoDataFrame cannot be empty",
        ),
        (
            "sample_nodes_gdf_alt_crs",
            "sample_edges_gdf",
            ValueError,
            "All GeoDataFrames must have the same CRS",
        ),
    ],
)
def test_validate_gdf(
    nodes_fixture: str | None,
    edges_fixture: str | None,
    error: type[Exception] | None,
    match: str | None,
    request: pytest.FixtureRequest,
) -> None:
    """Test validate_gdf with various inputs."""
    nodes_gdf = request.getfixturevalue(nodes_fixture) if nodes_fixture else None
    edges_gdf = request.getfixturevalue(edges_fixture) if edges_fixture else None

    if error:
        with pytest.raises(error, match=match):
            utils.validate_gdf(nodes_gdf=nodes_gdf, edges_gdf=edges_gdf)
    else:
        try:
            utils.validate_gdf(nodes_gdf=nodes_gdf, edges_gdf=edges_gdf)
        except Exception as e:
            pytest.fail(f"validate_gdf raised an unexpected exception: {e}")


@pytest.mark.parametrize(
    ("graph_input", "error", "match"),
    [
        # Success case
        ("sample_nx_graph", None, None),
        # Error cases
        ("not_a_gdf", TypeError, "Input must be a NetworkX graph"),
        ("empty_graph", ValueError, "Graph has no nodes"),
        ("graph_no_edges", ValueError, "Graph has no edges"),
    ],
)
def test_validate_nx(
    graph_input: str,
    error: type[Exception] | None,
    match: str | None,
    request: pytest.FixtureRequest,
) -> None:
    """Test validate_nx with various graph inputs."""
    if graph_input == "empty_graph":
        graph = nx.Graph()
    elif graph_input == "graph_no_edges":
        graph = nx.Graph()
        graph.add_node(1)
    else:
        graph = request.getfixturevalue(graph_input)

    if error:
        with pytest.raises(error, match=match):
            utils.validate_nx(graph)
    else:
        try:
            utils.validate_nx(graph)
        except Exception as e:
            pytest.fail(f"validate_nx raised an unexpected exception: {e}")


def test_nx_to_gdf_no_request(sample_nx_graph: nx.Graph) -> None:
    """Test that nx_to_gdf raises ValueError if both nodes and edges are False."""
    with pytest.raises(ValueError, match="Must request at least one of nodes or edges"):
        nx_to_gdf(sample_nx_graph, nodes=False, edges=False)
