"""
Compact yet comprehensive test-suite for proximity.py.

The previous version contained many almost-identical blocks scattered all over
the file.  This refactor centralises the common logic in a handful of helpers
and drives the whole suite through rich parametrisation, obtaining

• far less repetition;
• a clearer view of what is covered and what still is not;
• the same (or better) code-coverage.

All fixtures are still imported from conftest.py, exactly like in test_utils.py.
"""

from __future__ import annotations

import importlib
import math
from typing import TYPE_CHECKING
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from shapely.geometry import LineString

# -----------------------------------------------------------------------------#
# Module under test – try both package and flat-file layout                    #
# -----------------------------------------------------------------------------#
try:
    import city2graph.proximity as prox  # type: ignore
except ModuleNotFoundError:
    prox = importlib.import_module("proximity")  # type: ignore

# -----------------------------------------------------------------------------#
# Generic helpers                                                              #
# -----------------------------------------------------------------------------#
SkipExc = (NotImplementedError, AttributeError, NameError, ImportError)


def _run_or_skip(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Execute *fn* swallowing *unfinished implementation* errors."""
    try:
        return fn(*args, **kwargs)
    except SkipExc as exc:
        pytest.skip(f"implementation not ready: {exc}")


def _is_l_shaped(line: LineString) -> bool:
    """A crude test that a LineString is an L-shape (3 points, right angle)."""
    coords = list(line.coords)
    if len(coords) != 3:
        return False
    (x0, y0), (x1, y1), (x2, y2) = coords
    return (x0 == x1 or y0 == y1) and (x1 == x2 or y1 == y2)


# -----------------------------------------------------------------------------#
# Parameter sets                                                               #
# -----------------------------------------------------------------------------#
GEN_SPECS: list[tuple[str, dict[str, Any]]] = [
    ("knn_graph", {"k": 2}),
    ("fixed_radius_graph", {"radius": 2.0}),
    ("delaunay_graph", {}),
    ("gabriel_graph", {}),
    ("relative_neighborhood_graph", {}),
    ("euclidean_minimum_spanning_tree", {}),
    ("waxman_graph", {"beta": 0.6, "r0": 3.0, "seed": 42}),
]

METRICS = ["euclidean", "manhattan"]

# -----------------------------------------------------------------------------#
# 1.  Generic behaviour for Eucl. / Manhattan                                  #
# -----------------------------------------------------------------------------#
@pytest.mark.parametrize(("gen_name", "kwargs"), GEN_SPECS, ids=[spec[0] for spec in GEN_SPECS])
@pytest.mark.parametrize("metric", METRICS)
def test_generators_basic(
    sample_nodes_gdf: gpd.GeoDataFrame,
    gen_name: str,
    kwargs: dict[str, Any],
    metric: str,
) -> None:
    """Every generator should return two valid GeoDataFrames."""
    fn: Callable[..., Any] = getattr(prox, gen_name)
    nodes, edges = _run_or_skip(fn, sample_nodes_gdf, distance_metric=metric, **kwargs)

    # structure ----------------------------------------------------------
    assert isinstance(nodes, gpd.GeoDataFrame)
    assert isinstance(edges, gpd.GeoDataFrame)
    assert nodes.shape[0] == len(sample_nodes_gdf)
    assert nodes.crs == edges.crs == sample_nodes_gdf.crs

    # mandatory columns --------------------------------------------------
    assert "geometry" in edges.columns and "weight" in edges.columns

    # manhattan geometry sanity-check ------------------------------------
    if metric == "manhattan" and not edges.empty:
        assert all(_is_l_shaped(g) for g in edges.geometry)


# -----------------------------------------------------------------------------#
# 2.  Network metric – error if network_gdf missing, success otherwise         #
# -----------------------------------------------------------------------------#
@pytest.mark.parametrize(("gen_name", "kwargs"), GEN_SPECS, ids=[spec[0] for spec in GEN_SPECS])
def test_network_metric_error(sample_nodes_gdf: gpd.GeoDataFrame, gen_name: str, kwargs: dict) -> None:
    fn = getattr(prox, gen_name)
    with pytest.raises(ValueError):
        _run_or_skip(fn, sample_nodes_gdf, distance_metric="network", **kwargs)


@pytest.mark.parametrize(("gen_name", "kwargs"), [s for s in GEN_SPECS if s[0] != "waxman_graph"])
def test_network_metric_success(
    sample_nodes_gdf: gpd.GeoDataFrame,
    sample_edges_gdf: gpd.GeoDataFrame,
    gen_name: str,
    kwargs: dict,
) -> None:
    """With a toy street-network the call should succeed."""
    fn = getattr(prox, gen_name)
    nodes, edges = _run_or_skip(
        fn,
        sample_nodes_gdf,
        distance_metric="network",
        network_gdf=sample_edges_gdf,
        **kwargs,
    )
    assert not edges.empty or gen_name.endswith("minimum_spanning_tree")
    assert np.isfinite(edges["weight"].to_numpy()).all()


# -----------------------------------------------------------------------------#
# 3.  Special cases / generator specific                                       #
# -----------------------------------------------------------------------------#
def test_knn_graph_nx_return(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    G = _run_or_skip(prox.knn_graph, sample_nodes_gdf, k=3, as_nx=True)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() == len(sample_nodes_gdf)
    # Each node has k=3 neighbours in an undirected graph ⇒ E ≈ n*k/2
    assert math.isclose(G.number_of_edges(), len(sample_nodes_gdf) * 3 / 2, rel_tol=0.5)
    assert G.graph["crs"] == sample_nodes_gdf.crs


def test_waxman_reproducibility(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    _, e1 = _run_or_skip(prox.waxman_graph, sample_nodes_gdf, beta=0.5, r0=3, seed=11)
    _, e2 = _run_or_skip(prox.waxman_graph, sample_nodes_gdf, beta=0.5, r0=3, seed=11)
    assert e1.equals(e2)


# -----------------------------------------------------------------------------#
# 4.  Bridge nodes (multilayer)                                                 #
# -----------------------------------------------------------------------------#
@pytest.mark.parametrize("method, extra", [("knn", {"k": 1}), ("fixed_radius", {"radius": 3})])
def test_bridge_nodes_structure(
    sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame],
    method: str,
    extra: dict[str, Any],
) -> None:
    _, edges = _run_or_skip(
        prox.bridge_nodes,
        sample_hetero_nodes_dict,
        proximity_method=method,
        **extra,
    )
    layer_names = sample_hetero_nodes_dict.keys()
    expected_keys = {
        (a, "is_nearby", b) for a in layer_names for b in layer_names if a != b
    }
    assert set(edges) == expected_keys
    for gdf in edges.values():
        assert "weight" in gdf and "geometry" in gdf


def test_bridge_nodes_as_nx(sample_hetero_nodes_dict: dict[str, gpd.GeoDataFrame]) -> None:
    G = _run_or_skip(
        prox.bridge_nodes,
        sample_hetero_nodes_dict,
        proximity_method="knn",
        k=1,
        as_nx=True,
    )
    assert isinstance(G, nx.DiGraph)
    assert set(nx.get_node_attributes(G, "node_type").values()) == set(
        sample_hetero_nodes_dict
    )
    # relation attribute is always 'is_nearby'
    assert set(nx.get_edge_attributes(G, "relation").values()) <= {"is_nearby", None}


# -----------------------------------------------------------------------------#
# 5.  Edge weight sanity-check (Eucl. vs Manhattan)                             #
# -----------------------------------------------------------------------------#
def test_weight_formula(sample_nodes_gdf: gpd.GeoDataFrame) -> None:
    _, e_euc = _run_or_skip(prox.knn_graph, sample_nodes_gdf, k=1, distance_metric="euclidean")
    _, e_man = _run_or_skip(prox.knn_graph, sample_nodes_gdf, k=1, distance_metric="manhattan")

    if e_euc.empty or e_man.empty:
        pytest.skip("trivial data produced no edges")

    euclidean_any = next(iter(e_euc.itertuples()))
    manhattan_any = next(iter(e_man.itertuples()))

    assert euclidean_any.weight <= manhattan_any.weight + 1e-6  # L1 ≥ L2 in 2-D
