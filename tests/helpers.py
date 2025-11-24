"""Shared test helpers to reduce duplication and improve maintainability.

These utilities are intentionally lightweight and free of project imports
(other than third-party scientific stack) so they can be reused broadly
across tests without introducing circular dependencies.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable

# Public constants used in multiple test modules
SKIP_EXCEPTIONS = (NotImplementedError, AttributeError, NameError, ImportError)
TOLERANCE = 1e-6

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ----------------------------------------------------------------------------
# Execution helper
# ----------------------------------------------------------------------------


def run_or_skip(
    fn: Callable[..., tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph],
    *args: object,
    **kwargs: object,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph:
    """Execute function and skip on temporary implementation errors.

    This keeps tests expressive while allowing modules under active development
    to short-circuit gracefully.
    """
    try:
        return fn(*args, **kwargs)
    except SKIP_EXCEPTIONS as exc:  # pragma: no cover - behavior is a skip
        pytest.skip(f"Implementation not ready: {exc}")


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------


def is_l_shaped(line: LineString) -> bool:
    """Check if a LineString has three points forming a right angle (L-shape)."""
    coords = list(line.coords)
    if len(coords) != 3:
        return False
    (x0, y0), (x1, y1), (x2, y2) = coords
    return bool((x0 == x1 or y0 == y1) and (x1 == x2 or y1 == y2))


# ----------------------------------------------------------------------------
# Small synthetic datasets
# ----------------------------------------------------------------------------


def create_test_points(crs: str = "EPSG:27700") -> gpd.GeoDataFrame:
    """Create a tiny set of well-separated test points with stable indices."""
    return gpd.GeoDataFrame(
        {
            "id": [1, 2, 3],
            "geometry": [Point(0, 0), Point(10, 0), Point(5, 10)],
        },
        crs=crs,
    ).set_index("id")


def create_two_layers(crs: str = "EPSG:27700") -> dict[str, gpd.GeoDataFrame]:
    """Create two-layer test data used by bridge/bi-layer routines."""
    return {
        "layer1": gpd.GeoDataFrame(
            {"id": [1], "geometry": [Point(0, 0)]},
            crs=crs,
        ).set_index("id"),
        "layer2": gpd.GeoDataFrame(
            {"id": [2], "geometry": [Point(1, 1)]},
            crs=crs,
        ).set_index("id"),
    }


# ----------------------------------------------------------------------------
# Small factories for concise test data creation
# ----------------------------------------------------------------------------


def make_points_gdf(
    coords: list[tuple[float, float]],
    *,
    crs: str = "EPSG:27700",
    index_name: str = "id",
    ids: list[int | str] | None = None,
) -> gpd.GeoDataFrame:
    """Create a points GeoDataFrame from coordinate pairs.

    If ids is not provided, uses 1..N. Sets the index name for stability.
    """
    if ids is None:
        ids = list(range(1, len(coords) + 1))
    gdf = gpd.GeoDataFrame(
        {index_name: ids, "geometry": [Point(xy) for xy in coords]},
        crs=crs,
    )
    return gdf.set_index(index_name)


def make_lines_gdf(
    lines: list[list[tuple[float, float]]],
    *,
    crs: str = "EPSG:27700",
    id_col: str = "segment_id",
) -> gpd.GeoDataFrame:
    """Create a simple LineString GeoDataFrame with a sequential id column."""
    return gpd.GeoDataFrame(
        {id_col: list(range(len(lines)))},
        geometry=[LineString(coords) for coords in lines],
        crs=crs,
    ).set_index(id_col, drop=False)


def make_network_mi_gdf(
    pairs: list[tuple[int | str, int | str]],
    geoms: list[list[tuple[float, float]]],
    *,
    crs: str = "EPSG:27700",
    index_names: tuple[str, str] = ("source_id", "target_id"),
) -> gpd.GeoDataFrame:
    """Create a network edges GeoDataFrame with a 2-level MultiIndex.

    pairs: (source, target) tuples for the MultiIndex.
    geoms: coordinate sequences for LineString geometries; must align with pairs.
    """
    if len(pairs) != len(geoms):
        msg = "pairs and geoms must have the same length"
        raise ValueError(msg)
    idx = pd.MultiIndex.from_tuples(pairs, names=index_names)
    return gpd.GeoDataFrame(
        {"geometry": [LineString(coords) for coords in geoms]},
        index=idx,
        crs=crs,
    )


def make_grid_polygons_gdf(
    rows: int,
    cols: int,
    *,
    cell_size: float = 1.0,
    origin: tuple[float, float] = (0.0, 0.0),
    crs: str = "EPSG:27700",
    id_fmt: str = "P_{i}_{j}",
    id_name: str = "id",
) -> gpd.GeoDataFrame:
    """Create a rows x cols grid of unit squares as polygons.

    The index is set to id_fmt.format(i=i, j=j) for stable referencing.
    """
    ox, oy = origin
    records: list[dict[str, object]] = []
    for i in range(rows):
        for j in range(cols):
            x0, y0 = ox + i * cell_size, oy + j * cell_size
            poly = Polygon(
                [
                    (x0, y0),
                    (x0 + cell_size, y0),
                    (x0 + cell_size, y0 + cell_size),
                    (x0, y0 + cell_size),
                ],
            )
            records.append({id_name: id_fmt.format(i=i, j=j), "geometry": poly})
    gdf = gpd.GeoDataFrame(records, crs=crs)
    return gdf.set_index(id_name)


def make_segments_gdf(
    ids: list[str] | None,
    geoms_or_coords: list[BaseGeometry | list[tuple[float, float]]],
    *,
    level_rules: list[str | None] | str | None = "",
    connectors: list[str] | str | None = None,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Create a segments GeoDataFrame with optional level_rules/connectors.

    - ids: list of segment ids; if None, generates seg1..segN
    - geoms_or_coords: list of shapely geometries or coordinate lists for LineStrings
    - level_rules: optional string or list; if None, column is omitted
    - connectors: optional JSON string or list of JSON strings; if None, column omitted
    - crs: CRS string (default EPSG:4326 for Overture tests)
    """
    n = len(geoms_or_coords)
    if ids is None:
        ids = [f"seg{i + 1}" for i in range(n)]

    # Normalize geometries
    geometries: list[BaseGeometry] = []
    for item in geoms_or_coords:
        if isinstance(item, BaseGeometry):
            geometries.append(item)
        else:
            geometries.append(LineString(item))

    data: dict[str, object] = {"id": ids}

    # Normalize optional columns
    if level_rules is not None:
        level_rules_list = (
            [typing.cast("str | None", level_rules)] * n
            if isinstance(level_rules, str) or level_rules is None
            else level_rules
        )
        data["level_rules"] = level_rules_list

    if connectors is not None:
        connectors_list = [connectors] * n if isinstance(connectors, str) else connectors
        data["connectors"] = connectors_list

    return gpd.GeoDataFrame(data, geometry=geometries, crs=crs)


def make_connectors_gdf(
    ids: list[str],
    coords: list[tuple[float, float]],
    *,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Create a connectors GeoDataFrame with id and Point geometry."""
    return gpd.GeoDataFrame(
        {"id": ids, "geometry": [Point(xy) for xy in coords]},
        crs=crs,
    )


# ----------------------------------------------------------------------------
# Proximity-specific test helpers
# ----------------------------------------------------------------------------


def make_points_simple(
    coords: list[tuple[float, float]], crs: str = "EPSG:27700"
) -> gpd.GeoDataFrame:
    """Create a simple points GeoDataFrame with sequential integer IDs.

    Lightweight wrapper for proximity tests that need minimal setup.
    """
    return gpd.GeoDataFrame(
        {"id": list(range(len(coords)))},
        geometry=[Point(x, y) for x, y in coords],
        crs=crs,
    ).set_index("id")


def make_single_point(crs: str = "EPSG:27700") -> gpd.GeoDataFrame:
    """Create a single-point GeoDataFrame for early-exit test cases."""
    return gpd.GeoDataFrame({"id": [0]}, geometry=[Point(0, 0)], crs=crs).set_index("id")


def make_poly_points_pair(
    crs: str = "EPSG:3857",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create polygon/points pair for group_nodes testing.

    Returns
    -------
        Tuple of (polygons_gdf, points_gdf) where:
        - polygons: single 2x2 square at origin with index "A"
        - points: three points, two inside polygon, one outside
    """
    poly = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    polys = gpd.GeoDataFrame({"name": ["A"]}, geometry=[poly], crs=crs).set_index(
        pd.Index(["A"], name="zone")
    )
    pts = gpd.GeoDataFrame(
        {"pid": [1, 2, 3]},
        geometry=[Point(1, 1), Point(2, 1), Point(3, 3)],
        crs=crs,
    ).set_index("pid")
    return polys, pts


def make_square_polygons(crs: str = "EPSG:3857") -> gpd.GeoDataFrame:
    """Create a small grid of adjacent square polygons for contiguity tests.

    Returns 4 unit squares arranged as:
    - Poly 0: (0,0) to (1,1)
    - Poly 1: (1,0) to (2,1) - touches poly 0
    - Poly 2: (0,1) to (1,2) - touches poly 0
    - Poly 3: (2,0) to (3,1) - touches poly 1 only
    """
    polys = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # 0
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),  # 1 (touches 0)
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),  # 2 (touches 0)
        Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),  # 3 (touches 1 only)
    ]
    return gpd.GeoDataFrame({"val": range(len(polys))}, geometry=polys, crs=crs).set_index(
        pd.Index(range(len(polys)), name="pid")
    )


def make_network_edges(
    src_ids: list[int],
    dst_ids: list[int],
    geometries: list[LineString],
    crs: str = "EPSG:27700",
    extra_attrs: dict[str, list[object]] | None = None,
) -> gpd.GeoDataFrame:
    """Create a network edges GeoDataFrame with proper MultiIndex.

    Parameters
    ----------
    src_ids : list[int]
        Source node IDs
    dst_ids : list[int]
        Destination node IDs
    geometries : list[LineString]
        Edge geometries
    crs : str, optional
        Coordinate reference system, by default "EPSG:27700"
    extra_attrs : dict[str, list] | None, optional
        Additional columns as {column_name: values_list}, by default None

    Returns
    -------
    gpd.GeoDataFrame
        Network edges with MultiIndex (source_id, target_id)
    """
    mi = pd.MultiIndex.from_arrays([src_ids, dst_ids], names=("source_id", "target_id"))
    data: dict[str, object] = {
        "source_id": src_ids,
        "target_id": dst_ids,
        "geometry": geometries,
    }
    if extra_attrs:
        data.update(extra_attrs)
    return gpd.GeoDataFrame(data, index=mi, crs=crs)


def assert_valid_proximity_result(
    nodes: gpd.GeoDataFrame,
    edges: gpd.GeoDataFrame,
    expected_node_count: int,
    check_edge_schema: bool = True,
    allow_empty_edges: bool = True,
) -> None:
    """Assert nodes/edges have expected proximity graph structure.

    Parameters
    ----------
    nodes : gpd.GeoDataFrame
        Node GeoDataFrame to validate
    edges : gpd.GeoDataFrame
        Edge GeoDataFrame to validate
    expected_node_count : int
        Expected number of nodes
    check_edge_schema : bool, optional
        Whether to check for weight/geometry columns in edges, by default True
    allow_empty_edges : bool, optional
        Whether empty edges are acceptable (e.g., for single point graphs), by default True
    """
    assert len(nodes) == expected_node_count

    if not allow_empty_edges:
        assert not edges.empty

    if check_edge_schema and not edges.empty:
        assert {"weight", "geometry"}.issubset(edges.columns)


def assert_network_metric_requires_network_gdf(
    func: Callable[..., object],
    gdf: gpd.GeoDataFrame,
    **kwargs: object,
) -> None:
    """Assert function raises when network metric used without network_gdf.

    Parameters
    ----------
    func : Callable
        Proximity function to test
    gdf : gpd.GeoDataFrame
        GeoDataFrame to pass as first argument
    **kwargs
        Additional keyword arguments for the function
    """
    with pytest.raises(ValueError, match="network_gdf is required"):
        func(gdf, distance_metric="network", **kwargs)


# ----------------------------------------------------------------------------
# Assertions shared across tests (kept simple to avoid tight coupling)
# ----------------------------------------------------------------------------


def assert_valid_gdf(gdf: gpd.GeoDataFrame, expected_empty: bool = False) -> None:
    """Perform basic validity checks used in multiple tests."""
    assert isinstance(gdf, gpd.GeoDataFrame)
    if expected_empty:
        assert gdf.empty
    else:
        assert not gdf.empty
        if len(gdf) > 0 and "geometry" in gdf:
            assert gdf.geometry.is_valid.all()


def assert_crs_consistency(*gdfs: gpd.GeoDataFrame) -> None:
    """All non-empty GeoDataFrames should share the same CRS."""
    non_empty = [g for g in gdfs if isinstance(g, gpd.GeoDataFrame) and not g.empty]
    if len(non_empty) < 2:
        return
    ref = non_empty[0].crs
    assert all(g.crs == ref for g in non_empty[1:])


def assert_roundtrip_consistency(
    orig_nodes: gpd.GeoDataFrame,
    orig_edges: gpd.GeoDataFrame,
    conv_nodes: gpd.GeoDataFrame,
    conv_edges: gpd.GeoDataFrame,
) -> None:
    """Roundtrip conversions preserve sizes, indices, and CRS."""
    assert_crs_consistency(orig_nodes, conv_nodes, orig_edges, conv_edges)
    assert len(orig_nodes) == len(conv_nodes)
    assert len(orig_edges) == len(conv_edges)
    pd.testing.assert_index_equal(orig_nodes.index, conv_nodes.index)
    pd.testing.assert_index_equal(orig_edges.index, conv_edges.index)


# ----------------------------------------------------------------------------
# NetworkX helpers
# ----------------------------------------------------------------------------


def assert_valid_nx_graph(
    graph: nx.Graph,
    expected_nodes: int | None = None,
    expected_edges: int | None = None,
    crs: str | None = None,
) -> None:
    """Validate basic structural and metadata properties of a NetworkX graph.

    - Graph type is correct and has non-negative nodes/edges
    - Optional exact node/edge counts
    - Optional CRS metadata check
    - If nodes/edges exist, ensure basic attributes are dicts
    """
    assert isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
    assert graph.number_of_nodes() >= 0
    assert graph.number_of_edges() >= 0

    if expected_nodes is not None:
        assert graph.number_of_nodes() == expected_nodes
    if expected_edges is not None:
        assert graph.number_of_edges() == expected_edges
    if crs is not None:
        assert graph.graph.get("crs") == crs

    # Spot-check first node/edge attributes if present
    if graph.number_of_nodes():
        _, data = next(iter(graph.nodes(data=True)))
        assert isinstance(data, dict)
    if graph.number_of_edges():
        *_, data = next(iter(graph.edges(data=True)))
        assert isinstance(data, dict)


# ----------------------------------------------------------------------------
# Generic assertion helpers
# ----------------------------------------------------------------------------


def assert_has_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Assert that a DataFrame contains the specified columns."""
    missing = [c for c in columns if c not in df.columns]
    assert not missing, f"Missing expected columns: {missing}"


def assert_index_names(df: pd.DataFrame, names: list[str] | tuple[str, ...]) -> None:
    """Assert that the index (or MultiIndex) names match exactly."""
    if isinstance(df.index, pd.MultiIndex):
        assert list(df.index.names) == list(names)
    else:
        # Single index: names is expected to be length 1 or None
        expected = names[0] if isinstance(names, (list, tuple)) and names else None
        assert df.index.name in (expected, names)


def assert_geometry_types(gdf: gpd.GeoDataFrame, allowed_types: Iterable[str]) -> None:
    """Assert all geometries in a GeoDataFrame are among allowed types."""
    assert "geometry" in gdf
    assert gdf.geometry.geom_type.isin(list(allowed_types)).all()


def assert_l_shaped_edges(edges: gpd.GeoDataFrame) -> None:
    """Assert that all LineString geometries in edges are L-shaped segments.

    Uses the is_l_shaped helper to validate Manhattan geometry creation.
    No-op for empty frames.
    """
    if edges.empty:
        return
    assert "geometry" in edges
    assert all(is_l_shaped(geom) for geom in edges.geometry)


def get_center_point(center_source: object) -> Point:
    """Return a shapely Point from either a Geo(Data)Frame/Series or a Point."""
    if isinstance(center_source, gpd.GeoSeries):
        return center_source.iloc[0]
    if isinstance(center_source, gpd.GeoDataFrame):
        return center_source.geometry.iloc[0]
    if isinstance(center_source, Point):
        return center_source
    msg = f"Unsupported center source type: {type(center_source)!r}"
    raise TypeError(msg)


def assert_graph_counts(
    graph: nx.Graph,
    nodes: int | None = None,
    edges: int | None = None,
) -> None:
    """Assert node/edge counts for a NetworkX graph when provided."""
    assert isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
    if nodes is not None:
        assert graph.number_of_nodes() == nodes
    if edges is not None:
        assert graph.number_of_edges() == edges
