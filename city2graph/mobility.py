"""
Mobility / OD matrix utilities.

This module introduces the public function ``od_matrix_to_graph`` which
converts Origin-Destination (OD) data (adjacency matrices or edge lists)
into spatial graph representations used throughout the city2graph
ecosystem.

Notes
-----
This module includes a complete implementation of ``od_matrix_to_graph``:
input validation, zone alignment, conversion to a canonical edgelist,
thresholding and self-loop handling, optional geometry creation, and an
optional NetworkX export path.

Examples
--------
See the function docstring for usage examples with adjacency matrices,
NumPy arrays and edge lists (single/multi weight columns).
"""

from __future__ import annotations

import logging
import numbers
import warnings
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Mapping

    import networkx as nx
    from numpy.typing import NDArray

# Optional runtime import: keep ruff happy (top-level import), but tolerate
# environments without NetworkX installed. Type checking is covered above.
try:  # pragma: no cover - trivial import shim
    import networkx as nx
except Exception:  # noqa: BLE001 - broad by design to keep optional dep truly optional
    nx = cast("Any", None)

__all__ = ["od_matrix_to_graph"]

# Logger for informational summaries and errors (warnings used for data quality)
logger = logging.getLogger(__name__)


def od_matrix_to_graph(  # noqa: PLR0913, PLR0915 (public API requires many parameters)
    od_data: pd.DataFrame | np.ndarray,
    zones_gdf: gpd.GeoDataFrame,
    zone_id_col: str | None = None,
    *,
    matrix_type: str = "edgelist",
    source_col: str = "source",
    target_col: str = "target",
    weight_cols: list[str] | None = None,
    threshold: float | None = None,
    threshold_col: str | None = None,
    include_self_loops: bool = False,
    compute_edge_geometry: bool = True,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | object:
    """
    Convert OD data (edge list or adjacency matrix) into graph structures.

    Creates spatially-aware graphs from OD data following city2graph's
    GeoDataFrame-first design. Supports adjacency matrices (DataFrame or
    ndarray) and edgelists with one or multiple numeric weight columns.
    By default, this function returns a pair of GeoDataFrames representing
    nodes and edges. Optionally, it can return a NetworkX DiGraph when
    ``as_nx=True``. Edges are directed and thresholded with the rule
    weight >= threshold (or, when no threshold provided, strictly > 0).

    Parameters
    ----------
    od_data : pandas.DataFrame | numpy.ndarray
        * When ``matrix_type='adjacency'``: a square DataFrame whose
          index & columns are zone IDs, or a square ndarray whose ordering
          matches ``zones_gdf``.
        * When ``matrix_type='edgelist'``: a DataFrame containing origin,
          destination and one or more numeric flow columns.
    zones_gdf : geopandas.GeoDataFrame
        GeoDataFrame of zones. Must contain unique identifiers in
        ``zone_id_col``.
    zone_id_col : str, optional
        Name of the zone ID column in ``zones_gdf`` (required in this
        initial skeleton; automatic inference may be added later).
    matrix_type : {'edgelist','adjacency'}, default 'edgelist'
        Declares how to interpret ``od_data``.
    source_col, target_col : str, default 'source','target'
        Column names for origins / destinations when using an edge list.
    weight_cols : Sequence[str] | None
        Edge list weight (flow) columns to preserve. A single column acts as
        the canonical weight. If multiple columns are provided a
        ``threshold_col`` must be designated in the full implementation.
    threshold : float, optional
        Minimum flow retained (>=) applied to ``threshold_col`` (future logic).
    threshold_col : str, optional
        Column among ``weight_cols`` used for thresholding & canonical weight
        (required when ``len(weight_cols) > 1`` in full implementation).
    include_self_loops : bool, default False
        Keep flows where origin == destination (defaults drop when False).
    compute_edge_geometry : bool, default True
        Whether to build LineString geometries from zone centroids.
    as_nx : bool, default False
        If True, final output will be an ``nx.DiGraph``.

    Returns
    -------
    tuple of (geopandas.GeoDataFrame, geopandas.GeoDataFrame)
        The nodes and edges GeoDataFrames.
    networkx.DiGraph
        When ``as_nx=True``, a directed graph with node and edge attributes.

    Notes
    -----
                * Always directed; no flow symmetrization is performed by default.
                * Multiple weight columns are supported for edgelists via ``weight_cols``;
                    the column used for thresholding and canonical 'weight' is selected by
                    ``threshold_col`` (or implicitly the sole weight column).
    """
    # --- Validation (Task 2) ------------------------------------------------
    _validate_zones_gdf(zones_gdf, zone_id_col)
    _validate_crs(zones_gdf)

    # Validate matrix_type and that threshold is numeric if provided (Req 2.7)
    if threshold is not None and not isinstance(threshold, numbers.Number):
        msg = "threshold must be numeric (int or float)"
        raise ValueError(msg)

    validators: Mapping[str, Callable[..., None]] = {
        "adjacency": _validate_adjacency_data,
        "edgelist": _validate_edgelist_data,
    }
    if matrix_type not in validators:
        _msg = "matrix_type must be 'edgelist' or 'adjacency'"
        raise ValueError(_msg)
    validators[matrix_type](
        od_data,
        zones_gdf=zones_gdf,
        source_col=source_col,
        target_col=target_col,
        weight_cols=weight_cols,
    )
    if matrix_type == "edgelist":
        if not isinstance(od_data, pd.DataFrame):
            # Defensive: mypy narrowing + runtime safety
            msg = "For matrix_type='edgelist', od_data must be a pandas DataFrame"
            raise TypeError(msg)
        _validate_weights_threshold(
            od_data, weight_cols=weight_cols, _threshold=threshold, threshold_col=threshold_col
        )

    # --- Conversion to canonical edgelist (Tasks 3-4) ----------------------
    # Establish non-optional zone_id after validation for type checkers
    assert zone_id_col is not None
    zone_id = zone_id_col

    if matrix_type == "edgelist":
        if not isinstance(od_data, pd.DataFrame):
            msg = "For matrix_type='edgelist', od_data must be a pandas DataFrame"
            raise TypeError(msg)
        # Filter to zones and aggregate duplicates first
        aligned = _align_edgelist_zones(
            od_data,
            zones_gdf=zones_gdf,
            zone_id_col=zone_id,
            source_col=source_col,
            target_col=target_col,
        )
        # Normalize: thresholding, self-loops policy, canonical columns
        edge_df = _normalize_edgelist(
            aligned,
            source_col=source_col,
            target_col=target_col,
            weight_cols=weight_cols if weight_cols is not None else [],
            threshold=threshold,
            threshold_col=threshold_col,
            include_self_loops=include_self_loops,
        )
    elif matrix_type == "adjacency" and isinstance(od_data, pd.DataFrame):
        # Align labels with zones
        adj = _align_adjacency_zones(
            od_data,
            zones_gdf=zones_gdf,
            zone_id_col=zone_id,
        )
        edge_df = _adjacency_to_edgelist(
            adj,
            include_self_loops=include_self_loops,
            threshold=threshold,
        )
    elif matrix_type == "adjacency" and isinstance(od_data, np.ndarray):
        zone_ids = _align_numpy_array_zones(
            od_data,
            zones_gdf=zones_gdf,
            zone_id_col=zone_id,
        )
        edge_df = _adjacency_to_edgelist(
            od_data,
            zone_ids,
            include_self_loops=include_self_loops,
            threshold=threshold,
        )

    # Ensure canonical columns exist even if empty result
    if edge_df.empty:
        cols = (
            ["source", "target", "weight", *weight_cols]
            if matrix_type == "edgelist" and weight_cols
            else ["source", "target", "weight"]
        )
        edge_df = pd.DataFrame(columns=cols)

    # --- Spatial assembly (Task 5) -----------------------------------------
    nodes_gdf = zones_gdf.copy()
    edges_gdf = _create_edge_geometries(
        edge_df,
        zones_gdf,
        zone_id_col=zone_id,
        source_col="source",
        target_col="target",
        compute_edge_geometry=compute_edge_geometry,
    )

    # --- Output selection (Task 6) -----------------------------------------
    if not as_nx:
        # GeoDataFrame-first API: return (nodes, edges)
        logger.info("Created graph with %d nodes and %d edges", len(nodes_gdf), len(edges_gdf))
        return nodes_gdf, edges_gdf

    # Build NetworkX graph with attributes
    if nx is None:
        msg = (
            "NetworkX is required when as_nx=True. Please install 'networkx' to enable this option."
        )
        raise ImportError(msg)
    G = nx.DiGraph()
    # Add nodes with all attributes; node key equals zone_id
    for _, row in nodes_gdf.iterrows():
        node_id = row[zone_id]
        attrs = row.drop(labels=[zone_id]).to_dict()  # keep geometry and other attrs
        G.add_node(node_id, **attrs)

    # Add edges with attributes; include geometry only when computed
    edge_attr_cols = [c for c in edges_gdf.columns if c not in ("source", "target")]
    for _, row in edges_gdf.iterrows():
        u = row["source"]
        v = row["target"]
        attrs = {k: row[k] for k in edge_attr_cols}
        if not compute_edge_geometry and "geometry" in attrs:
            # Do not attach geometry attribute when geometry computation is disabled
            attrs.pop("geometry", None)
        G.add_edge(u, v, **attrs)

    logger.info(
        "Created graph with %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return cast("object", G)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _validate_zones_gdf(zones_gdf: gpd.GeoDataFrame, zone_id_col: str | None) -> None:
    """
    Validate the zones GeoDataFrame structure.

    Ensures the presence of a valid identifier column and basic integrity
    constraints (non-null, unique IDs) before downstream processing.

    Parameters
    ----------
    zones_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing zone geometries and attributes.
    zone_id_col : str | None
        Column name holding unique zone identifiers. Must be provided and
        present in ``zones_gdf``.

    Returns
    -------
    None
        This function validates input and raises on failure.
    """
    if not isinstance(zones_gdf, gpd.GeoDataFrame):
        msg = "zones_gdf must be a GeoDataFrame"
        raise TypeError(msg)

    if zone_id_col is None:
        msg = "zone_id_col must be provided"
        raise ValueError(msg)

    if zone_id_col not in zones_gdf.columns:
        msg = f"zone_id_col '{zone_id_col}' not found in zones_gdf columns"
        raise ValueError(msg)

    ids = zones_gdf[zone_id_col]
    if ids.isna().any():
        msg = "zone_id_col contains null values"
        raise ValueError(msg)

    if not ids.is_unique:
        msg = "zone_id_col values must be unique"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Data normalization helpers
#   - _adjacency_to_edgelist: fast vectorized conversion using NumPy
#   - _normalize_edgelist: multi-attribute aggregation and thresholding
#   - _apply_threshold: common filtering semantics
# ---------------------------------------------------------------------------
def _warn_and_clean_adjacency(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaNs with 0 and warn on data quality issues (NaNs, negatives).

    Sanitizes adjacency arrays by converting missing values to zeros while
    preserving negative values, emitting warnings to flag potential issues.

    Parameters
    ----------
    arr : numpy.ndarray
        Adjacency matrix values to sanitize.

    Returns
    -------
    numpy.ndarray
        Array with NaNs replaced by 0.0. Negative values are preserved.
    """
    n_nans = int(np.isnan(arr).sum())
    if n_nans:
        warnings.warn(
            f"Adjacency contains {n_nans} NaN values; treating as 0 (requirement 2.3)",
            UserWarning,
            stacklevel=2,
        )
        arr = np.nan_to_num(arr, nan=0.0)

    if (arr < 0).any():
        warnings.warn(
            "Adjacency contains negative weights; keeping values (requirement 5.6)",
            UserWarning,
            stacklevel=2,
        )
    return arr


def _extract_array_and_ids(
    adjacency: pd.DataFrame | np.ndarray,
    zone_ids: pd.Index | None,
) -> tuple[np.ndarray, pd.Index]:
    """
    Extract a numeric ndarray and matching zone ids from input.

    Accepts either a labeled DataFrame (preferred) or a raw ndarray with
    externally provided zone identifiers for consistent downstream mapping.

    Parameters
    ----------
    adjacency : pandas.DataFrame | numpy.ndarray
        Adjacency matrix as DataFrame (labels in index/columns) or ndarray.
    zone_ids : pandas.Index | None
        Zone identifiers when ``adjacency`` is an ndarray.

    Returns
    -------
    tuple[numpy.ndarray, pandas.Index]
        Tuple of (array, zone_ids).
    """
    if isinstance(adjacency, pd.DataFrame):
        ids = pd.Index(adjacency.index)
        arr = adjacency.to_numpy(dtype=float, copy=False)
        return arr, ids

    arr = np.asarray(adjacency, dtype=float)
    if zone_ids is None:
        msg = "zone_ids must be provided when adjacency is an ndarray"
        raise ValueError(msg)
    return arr, pd.Index(zone_ids)


def _build_adjacency_mask(
    arr: np.ndarray,
    *,
    include_self_loops: bool,
    threshold: float | None,
) -> NDArray[np.bool_]:
    """
    Create a boolean mask for edges based on threshold and self-loop policy.

    The mask flags retained (i, j) pairs according to the threshold rule and
    whether diagonal entries should be included.

    Parameters
    ----------
    arr : numpy.ndarray
        Adjacency array.
    include_self_loops : bool
        Whether to include diagonal entries in the mask.
    threshold : float | None
        If provided, keep values ``>= threshold``; else keep values ``> 0``.

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``arr.shape``.
    """
    mask: NDArray[np.bool_] = ((arr > 0) if threshold is None else (arr >= threshold)).astype(
        bool, copy=False
    )
    if include_self_loops:
        return mask
    n = arr.shape[0]
    result: NDArray[np.bool_] = np.logical_and(mask, ~np.eye(n, dtype=bool))
    return result


def _apply_threshold(series: pd.Series, *, threshold: float | None) -> pd.Series:
    """
    Return boolean mask for threshold semantics.

    Encodes the default behavior of dropping zeros when no threshold is
    provided, otherwise applying an inclusive cutoff (>= threshold).

    Parameters
    ----------
    series : pandas.Series
        Series of numeric values to filter.
    threshold : float | None
        Minimum value to retain. When ``None``, values strictly greater than
        zero are kept.

    Returns
    -------
    pandas.Series
        Boolean mask indicating which values pass the threshold.
    """
    return series > 0 if threshold is None else series >= threshold


def _adjacency_to_edgelist(
    adjacency: pd.DataFrame | np.ndarray,
    zone_ids: pd.Index | None = None,
    *,
    include_self_loops: bool = False,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Convert an adjacency matrix into a canonical edgelist DataFrame.

    Produces a long-form table of directed edges from a square matrix by
    applying thresholding and (optionally) dropping diagonal entries.

    Parameters
    ----------
    adjacency : DataFrame | ndarray
        Square matrix of flows. If DataFrame, index/columns are zone IDs and
        override ``zone_ids``. If ndarray, ``zone_ids`` must be provided and
        correspond to row/column order (validated upstream).
    zone_ids : pd.Index | None
        Zone identifiers in the order matching the matrix when ``adjacency``
        is an ndarray.
    include_self_loops : bool
        Whether to keep diagonal elements subject to thresholding.
    threshold : float | None
        Minimum flow to keep. None => drop zeros by default.

    Returns
    -------
    DataFrame
        Edge list with columns ['source', 'target', 'weight'].
    """
    arr, ids = _extract_array_and_ids(adjacency, zone_ids)

    # Defensive shape check (validated earlier but keep cheap sanity check)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        msg = "Adjacency must be a square matrix"
        raise ValueError(msg)

    # Clean and build mask
    arr = _warn_and_clean_adjacency(arr)
    mask = _build_adjacency_mask(arr, include_self_loops=include_self_loops, threshold=threshold)

    # Vectorized extraction of (i,j,weight)
    i, j = np.where(mask)
    if i.size == 0:
        return pd.DataFrame(columns=["source", "target", "weight"])  # empty result

    return pd.DataFrame(
        {
            "source": ids.take(i),
            "target": ids.take(j),
            "weight": arr[i, j],
        }
    )


def _coerce_weight_columns(
    df: pd.DataFrame,
    weight_cols: list[str],
) -> pd.DataFrame:
    """
    Coerce weight columns to numeric with NaN->0 and warnings.

    Converts specified columns to numeric dtype, treating pre-existing NaNs
    as zeros and warning when negative values are encountered.

    Parameters
    ----------
    df : pandas.DataFrame
        Input edge list DataFrame.
    weight_cols : list[str]
        Names of columns to coerce to numeric.

    Returns
    -------
    pandas.DataFrame
        DataFrame with specified columns coerced to numeric (NaNs filled where
        originally present) and warnings emitted for negatives.
    """
    coerced = df.copy()
    for col in weight_cols:
        original = coerced[col]
        original_nans = int(pd.isna(original).sum())
        s = pd.to_numeric(original, errors="coerce")
        n_total = len(s)
        after_nans = int(pd.isna(s).sum())

        if n_total > 0 and after_nans == n_total:
            msg = f"Column '{col}' could not be coerced to numeric (all values are non-numeric)"
            raise ValueError(msg)

        # Treat pre-existing NaNs as 0 with a warning (Req 2.3),
        # but if coercion introduced new NaNs (after_nans > original_nans) -> error (Req 5.10/5.11)
        if after_nans > original_nans:
            msg = f"Column '{col}' contains non-numeric values that cannot be coerced to numeric"
            raise ValueError(msg)
        if original_nans:
            warnings.warn(
                f"Column '{col}' has {original_nans} NaN values; treating as 0 (requirement 2.3)",
                UserWarning,
                stacklevel=2,
            )
            # Only fill positions that were originally NaN
            s = s.mask(pd.isna(original), 0)

        if (s < 0).any():
            warnings.warn(
                f"Column '{col}' contains negative weights; keeping values (requirement 5.6)",
                UserWarning,
                stacklevel=2,
            )
        coerced[col] = s
    return coerced


def _aggregate_edgelist(
    df: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    weight_cols: list[str],
) -> pd.DataFrame:
    """
    Aggregate duplicate (source, target) pairs by summing weight columns.

    Groups edges by origin and destination, summing all provided weight
    columns to remove duplicates in a single pass.

    Parameters
    ----------
    df : pandas.DataFrame
        Edge list DataFrame.
    source_col, target_col : str
        Column names for origin/destination identifiers.
    weight_cols : list[str]
        Weight columns to sum during aggregation.

    Returns
    -------
    pandas.DataFrame
        Aggregated edge list.
    """
    return df.groupby([source_col, target_col], as_index=False)[weight_cols].sum()


def _resolve_primary(weight_cols: list[str], threshold_col: str | None) -> str:
    """
    Return the primary weight column used for thresholding and canonical weight.

    When multiple weight columns exist, this selects the one used for
    thresholding and populating the canonical 'weight' field.

    Parameters
    ----------
    weight_cols : list[str]
        Candidate weight columns.
    threshold_col : str | None
        Preferred column to use if multiple weights are present.

    Returns
    -------
    str
        The selected primary weight column name.
    """
    if len(weight_cols) == 1:
        return weight_cols[0]
    if threshold_col and threshold_col in weight_cols:
        return threshold_col
    msg = "When multiple weight_cols are provided, a valid threshold_col must be specified"
    raise ValueError(msg)


def _normalize_edgelist(
    edgelist_df: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    weight_cols: list[str],
    threshold: float | None,
    threshold_col: str | None,
    include_self_loops: bool,
) -> pd.DataFrame:
    """
    Normalize an edgelist with optional multi-attribute weights.

    Coerces weights to numeric, aggregates duplicates, applies thresholding,
    and returns canonical columns with a primary 'weight' field.

    Parameters
    ----------
    edgelist_df : pandas.DataFrame
        Input edge list DataFrame containing at least the source and target
        columns and the specified weight columns.
    source_col : str
        Column name for origin identifiers in ``edgelist_df``.
    target_col : str
        Column name for destination identifiers in ``edgelist_df``.
    weight_cols : list[str]
        Names of numeric weight (flow) columns to process and preserve.
    threshold : float | None
        Inclusive threshold applied to the primary weight column; when
        ``None``, zeros are dropped by default.
    threshold_col : str | None
        Name of the primary weight column used for thresholding when multiple
        weights are provided; if a single weight column exists, this may be
        ``None``.
    include_self_loops : bool
        Whether to retain edges where source equals target; defaults to False.

    Returns
    -------
    pandas.DataFrame
        Canonical edge list with columns ['source', 'target', 'weight', ...],
        where 'weight' mirrors the chosen primary weight column.
    """
    if not weight_cols:
        msg = "weight_cols must be provided for edgelist normalization"
        raise ValueError(msg)

    # Coerce and aggregate in focused helpers
    coerced = _coerce_weight_columns(edgelist_df, weight_cols)
    agg_df = _aggregate_edgelist(
        coerced, source_col=source_col, target_col=target_col, weight_cols=weight_cols
    )

    # Remove self-loops unless explicitly included
    if not include_self_loops:
        agg_df = agg_df.loc[agg_df[source_col] != agg_df[target_col]]

    # Select primary column for thresholding and canonical weight
    primary = _resolve_primary(weight_cols, threshold_col)

    # Apply threshold
    mask = _apply_threshold(agg_df[primary], threshold=threshold)
    filtered = agg_df.loc[mask].copy()

    # Build canonical columns and rename source/target for consistency
    out = filtered.rename(columns={source_col: "source", target_col: "target"})
    out.insert(2, "weight", out[primary])  # canonical weight mirrors primary

    # Ensure canonical ordering: source, target, weight, then other weight columns
    # Keep only specified weight columns (MVP drops other non-weight attributes per 6.6)
    remaining_weights = [c for c in weight_cols]

    # 'weight' already mirrors primary; still preserve all requested columns
    cols = ["source", "target", "weight", *remaining_weights]
    return out.loc[:, cols]


def _validate_adjacency_data(
    od_data: pd.DataFrame | np.ndarray,
    *,
    zones_gdf: gpd.GeoDataFrame,
    **_: object,
) -> None:
    """
    Validate adjacency style OD data (square, labels/order).

    Checks shape, labeling, and compatibility with the provided zones when
    using ndarray inputs.

    Parameters
    ----------
    od_data : pandas.DataFrame | numpy.ndarray
        Adjacency data.
    zones_gdf : geopandas.GeoDataFrame
        Zones for validating shape/ordering when ndarray is used.

    Returns
    -------
    None
        This function validates input and raises on failure.

    Other Parameters
    ----------------
    **_ : object
        Additional keyword arguments ignored by this validator.
    """
    if isinstance(od_data, pd.DataFrame):
        checks = [
            (od_data.shape[0] == od_data.shape[1], "Adjacency DataFrame must be square"),
            (
                od_data.index.equals(od_data.columns),
                "Adjacency DataFrame index and columns must match exactly",
            ),
            (
                od_data.index.is_unique and od_data.columns.is_unique,
                "Adjacency DataFrame index and columns must be unique",
            ),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(msg)

    elif isinstance(od_data, np.ndarray):
        if not (od_data.ndim == 2 and od_data.shape[0] == od_data.shape[1]):
            msg = "Adjacency ndarray must be 2D square"
            raise ValueError(msg)

        if od_data.shape[0] != len(zones_gdf):
            msg = "Adjacency ndarray size must match number of zones in zones_gdf"
            raise ValueError(msg)

        warnings.warn(
            "Assuming ndarray row/column ordering matches zones_gdf order (requirement 5.7)",
            UserWarning,
            stacklevel=2,
        )

    else:
        msg = "For matrix_type='adjacency', od_data must be a pandas DataFrame or numpy ndarray"
        raise TypeError(msg)


def _validate_edgelist_data(
    od_data: pd.DataFrame,
    *,
    source_col: str,
    target_col: str,
    weight_cols: list[str] | None,
    **_: object,
) -> None:
    """
    Validate edgelist structural requirements (required cols, weights present).

    Ensures the presence of required columns and weight fields prior to
    normalization and thresholding.

    Parameters
    ----------
    od_data : pandas.DataFrame
        Edge list data.
    source_col, target_col : str
        Column names for origin/destination identifiers.
    weight_cols : list[str] | None
        Names of weight columns that must exist.

    Returns
    -------
    None
        This function validates input and raises on failure.

    Other Parameters
    ----------------
    **_ : object
        Additional keyword arguments ignored by this validator.
    """
    if not isinstance(od_data, pd.DataFrame):
        msg = "For matrix_type='edgelist', od_data must be a pandas DataFrame"
        raise TypeError(msg)

    missing_basic = [c for c in (source_col, target_col) if c not in od_data.columns]
    if missing_basic:
        msg = f"Edgelist DataFrame missing required columns: {', '.join(missing_basic)}"
        raise ValueError(msg)

    if not weight_cols:
        msg = "weight_cols must be provided (at least one weight column) for edgelist input"
        raise ValueError(msg)

    missing_w = [c for c in weight_cols if c not in od_data.columns]
    if missing_w:
        msg = f"weight_cols contain columns not present in edgelist: {', '.join(missing_w)}"
        raise ValueError(msg)


def _validate_weights_threshold(
    _od_edgelist: pd.DataFrame,
    *,
    weight_cols: list[str] | None,
    _threshold: float | None,
    threshold_col: str | None,
) -> None:
    """
    Validate weight columns and threshold semantics (subset of full spec).

    Confirms that a valid primary column is designated when multiple weights
    are supplied and that provided names are consistent.

    Parameters
    ----------
    _od_edgelist : pandas.DataFrame
        Edge list data (unused here; included for signature symmetry).
    weight_cols : list[str] | None
        Weight columns provided by the caller.
    _threshold : float | None
        Threshold value.
    threshold_col : str | None
        Name of the column used for thresholding when multiple weights exist.

    Returns
    -------
    None
        This function validates input and raises on failure.
    """
    if weight_cols is None:
        return  # already enforced earlier, defensive

    # Validate threshold_col selection and unify coercion using shared helper
    if len(weight_cols) > 1 and (threshold_col is None or threshold_col not in weight_cols):
        msg = "When multiple weight_cols are provided a valid threshold_col must be specified"
        raise ValueError(msg)
    if len(weight_cols) == 1 and (threshold_col is not None and threshold_col not in weight_cols):
        msg = "threshold_col not in weight_cols"
        raise ValueError(msg)

    # Do not coerce here; coercion happens once in _normalize_edgelist


# ---------------------------------------------------------------------------
# Zone alignment and mapping helpers
# ---------------------------------------------------------------------------
def _align_adjacency_zones(
    adjacency_df: pd.DataFrame,
    *,
    zones_gdf: gpd.GeoDataFrame,
    zone_id_col: str,
) -> pd.DataFrame:
    """
    Align an adjacency DataFrame to the provided zones.

    Reindexes the matrix to the intersection of zone identifiers and warns
    about labels that are missing on either side.

    Parameters
    ----------
    adjacency_df : pandas.DataFrame
        Square adjacency matrix with zone ID index/columns.
    zones_gdf : geopandas.GeoDataFrame
        Zones GeoDataFrame.
    zone_id_col : str
        Column name holding unique zone identifiers in ``zones_gdf``.

    Returns
    -------
    pandas.DataFrame
        Sub-matrix aligned to overlapping zone IDs in both inputs.
    """
    matrix_ids = pd.Index(adjacency_df.index)

    zones_ids = pd.Index(zones_gdf[zone_id_col])
    common = matrix_ids.intersection(zones_ids)
    if common.empty:
        msg = "No overlapping zone IDs between adjacency matrix and zones_gdf"
        logger.error(msg)
        raise ValueError(msg)

    missing_in_matrix = zones_ids.difference(matrix_ids)
    missing_in_zones = matrix_ids.difference(zones_ids)
    if len(missing_in_matrix) > 0:
        warnings.warn(
            f"{len(missing_in_matrix)} zone IDs in zones_gdf not present in adjacency; they will be isolated nodes",
            UserWarning,
            stacklevel=2,
        )
    if len(missing_in_zones) > 0:
        warnings.warn(
            f"{len(missing_in_zones)} labels in adjacency not present in zones_gdf; related edges will be dropped",
            UserWarning,
            stacklevel=2,
        )

    # Reindex to common ids (preserve the order from adjacency_df to keep consistency with provided matrix)
    return adjacency_df.reindex(index=common, columns=common)


def _align_numpy_array_zones(
    adjacency_array: np.ndarray,
    *,
    zones_gdf: gpd.GeoDataFrame,
    zone_id_col: str,
) -> pd.Index:
    """
    Validate ndarray size and return zone ids in ``zones_gdf`` order.

    Ensures the array size matches the number of zones and yields the
    identifier sequence in the same order as the zones GeoDataFrame.

    Parameters
    ----------
    adjacency_array : numpy.ndarray
        Square adjacency array.
    zones_gdf : geopandas.GeoDataFrame
        Zones GeoDataFrame.
    zone_id_col : str
        Column name holding unique zone identifiers in ``zones_gdf``.

    Returns
    -------
    pandas.Index
        Zone identifiers in the same order as ``zones_gdf``.
    """
    n = len(zones_gdf)
    if adjacency_array.shape[0] != n:
        msg = "Adjacency ndarray size must match number of zones in zones_gdf"
        raise ValueError(msg)

    return cast("pd.Index", pd.Index(zones_gdf[zone_id_col]))


def _align_edgelist_zones(
    edgelist_df: pd.DataFrame,
    *,
    zones_gdf: gpd.GeoDataFrame,
    zone_id_col: str,
    source_col: str,
    target_col: str,
) -> pd.DataFrame:
    """
    Filter edgelist to valid zones (no aggregation here).

    Drops edges referencing unknown zone identifiers while preserving
    duplicates for later aggregation.

    Parameters
    ----------
    edgelist_df : pandas.DataFrame
        Input edgelist containing at least ``source_col`` and ``target_col``.
    zones_gdf : geopandas.GeoDataFrame
        Zones with ``zone_id_col`` present.
    zone_id_col : str
        Column name holding unique zone identifiers.
    source_col, target_col : str
        Column names for origin/destination identifiers in ``edgelist_df``.

    Returns
    -------
    pandas.DataFrame
        Filtered edgelist containing only valid zone IDs. Duplicate
        ``(source, target)`` rows are preserved here and handled downstream.
    """
    valid_ids = set(zones_gdf[zone_id_col].tolist())
    before = len(edgelist_df)
    mask_valid = edgelist_df[source_col].isin(valid_ids) & edgelist_df[target_col].isin(valid_ids)
    filtered = edgelist_df.loc[mask_valid].copy()
    dropped = before - len(filtered)
    if dropped > 0:
        warnings.warn(
            f"Dropped {dropped} edges referencing unknown zone IDs (requirement 3.6)",
            UserWarning,
            stacklevel=2,
        )

    if filtered.empty:
        msg = "No overlapping zone IDs between edgelist and zones_gdf"
        logger.error(msg)
        raise ValueError(msg)

    return filtered


# ---------------------------------------------------------------------------
# Spatial geometry helpers (centroids and edge geometries)
# ---------------------------------------------------------------------------
def _validate_crs(zones_gdf: gpd.GeoDataFrame) -> None:
    """
    Warn about CRS issues and ensure CRS info is preserved.

    Emits warnings for missing CRS and geographic CRS to signal potential
    issues with distance-based computations.

    Parameters
    ----------
    zones_gdf : geopandas.GeoDataFrame
        Zones GeoDataFrame whose CRS will be validated.

    Returns
    -------
    None
        This function validates input and raises on failure.
    """
    crs = zones_gdf.crs
    if crs is None:
        warnings.warn(
            "zones_gdf has no CRS set; outputs will have undefined CRS (requirement 3.1)",
            UserWarning,
            stacklevel=2,
        )
        return

    # pyproj.CRS exposes is_geographic when available
    if getattr(crs, "is_geographic", False):
        warnings.warn(
            "Geographic CRS detected; distance/length measures may be inaccurate (requirement 3.5)",
            UserWarning,
            stacklevel=2,
        )


def _compute_centroids(zones_gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """
    Compute centroids for zones preserving CRS.

    Calculates feature centroids and explicitly propagates the CRS to the
    resulting GeoSeries for consistency.

    Parameters
    ----------
    zones_gdf : geopandas.GeoDataFrame
        Zones GeoDataFrame whose feature centroids will be computed.

    Returns
    -------
    geopandas.GeoSeries
        GeoSeries indexed like ``zones_gdf`` with the same CRS.
    """
    # GeoPandas returns a GeoSeries with the same CRS as input geometry
    centroids = zones_gdf.geometry.centroid
    # Explicitly carry CRS (some versions keep it automatically, we enforce)
    centroids.set_crs(zones_gdf.crs, allow_override=True, inplace=True)
    return centroids


def _create_edge_geometries(
    edges_df: pd.DataFrame,
    zones_gdf: gpd.GeoDataFrame,
    *,
    zone_id_col: str,
    source_col: str = "source",
    target_col: str = "target",
    compute_edge_geometry: bool = True,
) -> gpd.GeoDataFrame:
    """
    Create edge geometries as LineStrings between zone centroids.

    Builds straight-line connections between centroid coordinates for each
    edge; optionally returns an empty geometry when disabled.

    Parameters
    ----------
    edges_df : pandas.DataFrame
        Canonical edgelist with ``source``, ``target`` and weight columns.
    zones_gdf : geopandas.GeoDataFrame
        Zones GeoDataFrame providing geometry and CRS.
    zone_id_col : str
        Column name holding unique zone identifiers in ``zones_gdf``.
    source_col, target_col : str, default 'source', 'target'
        Column names in ``edges_df`` to map to zone centroids.
    compute_edge_geometry : bool, default True
        Whether to construct LineString geometries.

    Returns
    -------
    geopandas.GeoDataFrame
        Edge GeoDataFrame with a ``geometry`` column (possibly ``None`` values
        when ``compute_edge_geometry`` is False) and CRS preserved from zones.
    """
    # Use input edges directly; upstream alignment ensures valid IDs for edgelists.
    # For adjacency-derived edges or edge cases, missing IDs/centroids are handled below.
    e = edges_df.copy()

    # Fast-path: skip centroid computation when geometry is disabled or there are no edges
    if not compute_edge_geometry or e.empty:
        geom = gpd.GeoSeries([None] * len(e), crs=zones_gdf.crs)
        return gpd.GeoDataFrame(e, geometry=geom, crs=zones_gdf.crs)

    # Build centroid lookup once (vectorized via reindex below)
    centroids = _compute_centroids(zones_gdf)
    # Reindex centroids by zone_id values to allow direct .reindex with source/target IDs
    centroids.index = pd.Index(zones_gdf[zone_id_col])

    # Vectorized mapping of endpoints -> centroids using reindex (preserves order)
    src_pts = centroids.reindex(e[source_col].values)
    tgt_pts = centroids.reindex(e[target_col].values)

    # Some ids might still be missing (unknown IDs or missing centroids); drop such rows with warning
    missing_src = src_pts.isna()
    missing_tgt = tgt_pts.isna()
    missing_any = missing_src | missing_tgt
    n_missing = int(missing_any.sum())
    if n_missing:
        warnings.warn(
            f"Dropping {n_missing} edges with unknown zone IDs or missing centroid(s) (requirement 3.6)",
            UserWarning,
            stacklevel=2,
        )
        if n_missing == len(e):
            # All missing; return empty GeoDataFrame with CRS
            return gpd.GeoDataFrame(
                e.iloc[0:0], geometry=gpd.GeoSeries([], crs=zones_gdf.crs), crs=zones_gdf.crs
            )
        e = e.loc[~missing_any].copy()
        src_pts = src_pts.loc[~missing_any]
        tgt_pts = tgt_pts.loc[~missing_any]

    # Create LineStrings; use numpy.frompyfunc to avoid explicit Python loop semantics
    make_line = lambda a, b: LineString([a, b])  # noqa: E731
    # frompyfunc returns object dtype array; it's acceptable for geometry construction
    lines = np.frompyfunc(make_line, 2, 1)(src_pts.values, tgt_pts.values)
    geom = gpd.GeoSeries(lines.tolist(), crs=zones_gdf.crs)
    return gpd.GeoDataFrame(e.reset_index(drop=True), geometry=geom, crs=zones_gdf.crs)
