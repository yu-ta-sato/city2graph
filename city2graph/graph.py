"""
Module for creating heterogeneous graph representations of urban environments.

Converts geodataframes containing spatial data into PyTorch Geometric HeteroData objects.
"""

try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import to_networkx as pyg_to_networkx

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

    # Create placeholder classes to prevent import errors
    class HeteroData:
        pass

    class Data:
        pass


import logging
from typing import Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .utils import _validate_gdf
from .utils import _validate_nx

logger = logging.getLogger(__name__)

# Define the public API for this module
__all__ = [
    "from_morphological_graph",
    "gdf_to_pyg",
    "heterogeneous_graph",
    "homogeneous_graph",
    "is_torch_available",
    "nx_to_pyg",
    "pyg_to_gdf",
    "pyg_to_nx",
]


def is_torch_available() -> bool:
    """
    Check if PyTorch and PyTorch Geometric are available.

    Returns
    -------
    bool
        True if PyTorch and PyTorch Geometric are available, False otherwise.
    """
    return TORCH_AVAILABLE


def _get_device(
    device: Union[str, "torch.device", None] = None,
) -> Union["torch.device", str]:
    """
    Get the appropriate torch device (CUDA if available, otherwise CPU).

    Parameters
    ----------
    device : str or torch.device, default None
        Device to use for tensors. Must be 'cuda', 'cpu', torch.device or None.
        If None, will use CUDA if available, otherwise CPU.

    Returns
    -------
    torch.device
        The device to use for tensors

    Raises
    ------
    ValueError
        If device is not None, 'cuda', 'cpu', or torch.device
    ImportError
        If PyTorch is not installed
    """
    if not TORCH_AVAILABLE:
        msg = (
            "PyTorch and PyTorch Geometric are required for this function. "
            "Please install them using: poetry install --with torch or "
            "pip install city2graph[torch]"
        )
        raise ImportError(
            msg,
        )

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if device in ["cuda", "cpu"]:
        return torch.device(device)
    msg = "Device must be 'cuda', 'cpu', a torch.device object, or None"
    raise ValueError(msg)


def _detect_edge_columns(edge_gdf: gpd.GeoDataFrame,
                         id_col: str | None = None,
                         source_hint: list[str] | None = None,
                         target_hint: list[str] | None = None) -> tuple[str | None, str | None]:
    """
    Detect appropriate source and target columns in an edge GeoDataFrame.

    Parameters
    ----------
    edge_gdf : pandas.DataFrame
        DataFrame containing edge data
    id_col : str, default None
        Column name used to identify nodes, used as a hint
    source_hint : list, default None
        Additional keywords to look for in source column names
    target_hint : list, default None
        Additional keywords to look for in target column names

    Returns
    -------
    tuple
        (source_col, target_col) - detected column names
    """
    if edge_gdf.empty or len(edge_gdf.columns) < 2:
        return None, None

    # Default hint keywords
    source_keywords = ["from", "source", "start", "u"]
    target_keywords = ["to", "target", "end", "v"]

    # Add custom hints if provided
    if source_hint:
        source_keywords.extend([hint.lower() for hint in source_hint])
    if target_hint:
        target_keywords.extend([hint.lower() for hint in target_hint])

    # Add id_col as hint if provided
    if id_col:
        source_keywords.append(id_col.lower())
        target_keywords.append(id_col.lower())

    # Find columns matching source keywords
    from_candidates = [
        col
        for col in edge_gdf.columns
        if any(keyword in col.lower() for keyword in source_keywords)
    ]

    # Find columns matching target keywords
    to_candidates = [
        col
        for col in edge_gdf.columns
        if any(keyword in col.lower() for keyword in target_keywords)
    ]

    # Select best candidates
    if from_candidates and to_candidates:
        return from_candidates[0], to_candidates[0]

    # Fall back to first two columns if needed
    if len(edge_gdf.columns) >= 2 and "geometry" not in edge_gdf.columns[:2]:
        return edge_gdf.columns[0], edge_gdf.columns[1]
    if len(edge_gdf.columns) >= 3 and "geometry" in edge_gdf.columns[0]:
        return edge_gdf.columns[1], edge_gdf.columns[2]

    return None, None


def _get_edge_columns(
    edge_gdf: gpd.GeoDataFrame,
    source_col: str | None,
    target_col: str | None,
    source_mapping: dict[str, int],
    target_mapping: dict[str, int],
    id_col: str | None = None,
) -> tuple[str | None, str | None]:
    """Consolidate logic for detecting or confirming source/target columns."""
    if source_col is None or target_col is None:
        detected_source, detected_target = _detect_edge_columns(
            edge_gdf,
            id_col=id_col,
            source_hint=list(source_mapping.keys())[:1] if source_mapping else None,
            target_hint=list(target_mapping.keys())[:1] if target_mapping else None,
        )
        if source_col is None:
            source_col = detected_source
        if target_col is None:
            target_col = detected_target
    return source_col, target_col


def _extract_node_id_mapping(node_gdf: gpd.GeoDataFrame,
                             id_col: str | None = None) -> tuple[dict[str, int], str]:
    """
    Extract a mapping from node IDs to indices.

    Parameters
    ----------
    node_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing node data.
    id_col : str, optional
        Column name that uniquely identifies each node.
        If provided and missing from node_gdf, a ValueError is raised to prevent unintended errors.

    Returns
    -------
    tuple
        (id_mapping, used_id_col) - mapping from IDs to indices and the ID column used
    """
    # If id_col is provided but not found, raise a ValueError to alert the user.
    if id_col is not None and id_col not in node_gdf.columns:
        msg = f"Provided id_col '{id_col}' not found in node GeoDataFrame columns."
        raise ValueError(
            msg,
        )
    if id_col is None:
        # Use index if id_col is None.
        id_mapping = {str(idx): i for i, idx in enumerate(node_gdf.index)}
        return id_mapping, "index"

    # Use specified column if found.
    id_mapping = {
        str(node_id): idx for idx, node_id in enumerate(node_gdf[id_col].unique())
    }
    return id_mapping, id_col


# Modified _create_node_features: renamed parameter "attribute_cols" to "feature_cols"
def _create_node_features(node_gdf: gpd.GeoDataFrame,
                          feature_cols: list[str] | None = None,
                          device: Union[str, "torch.device"] | None = None) -> "torch.Tensor":
    """
    Create node feature tensors from attribute columns.

    Parameters
    ----------
    node_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing node data
    feature_cols : list, default None
        List of column names to use as node features
    device : str, default None
        Device to use for tensors. Must be 'cuda' or 'cpu' if provided.

    Returns
    -------
    torch.Tensor
        Tensor of node features
    """
    device = _get_device(device)

    if feature_cols is None:
        return torch.zeros((len(node_gdf), 0), dtype=torch.float, device=device)

    # Vectorized column validation using set intersection
    valid_cols = list(set(feature_cols) & set(node_gdf.columns))

    if valid_cols:
        # Direct numpy array conversion for better performance
        features_array = node_gdf[valid_cols].to_numpy().astype(np.float32)
        return torch.from_numpy(features_array).to(device=device, dtype=torch.float)

    return torch.zeros((len(node_gdf), 0), dtype=torch.float, device=device)


def _create_edge_features(edge_gdf: gpd.GeoDataFrame,
                          feature_cols: list[str] | None = None,
                          device: Union[str, "torch.device"] | None = None) -> "torch.Tensor":
    """
    Create edge feature tensors from attribute columns in edge_gdf.

    Parameters
    ----------
    edge_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing edge data.
    feature_cols : list, default None
        List of column names to use as edge features.
    device : str or torch.device, default None
        Device to use for tensors.

    Returns
    -------
    torch.Tensor
        Tensor of edge features.
    """
    device = _get_device(device)
    if feature_cols is None:
        return torch.empty((edge_gdf.shape[0], 0), dtype=torch.float, device=device)

    # Vectorized column validation using set intersection
    valid_cols = list(set(feature_cols) & set(edge_gdf.columns))
    if not valid_cols:
        return torch.empty((edge_gdf.shape[0], 0), dtype=torch.float, device=device)

    # Direct numpy conversion for better performance
    features_array = edge_gdf[valid_cols].to_numpy().astype(np.float32)
    return torch.from_numpy(features_array).to(device=device, dtype=torch.float)


def _map_edge_strings(edge_gdf: gpd.GeoDataFrame,
                      source_col: str,
                      target_col: str) -> gpd.GeoDataFrame:
    """Convert source/target columns to string once for vectorized lookups."""
    edge_gdf[f"__{source_col}_str"] = edge_gdf[source_col].astype(str)
    edge_gdf[f"__{target_col}_str"] = edge_gdf[target_col].astype(str)
    return edge_gdf


def _create_edge_idx_pairs(edge_gdf: gpd.GeoDataFrame,
                           source_mapping: dict[str, int],
                           target_mapping: dict[str, int] | None = None,
                           source_col: str | None = None,
                           target_col: str | None = None) -> list[list[int]]:
    """
    Process edges to create edge indices using vectorized operations.

    Parameters
    ----------
    edge_gdf : pandas.DataFrame
        DataFrame containing edge data
    source_mapping : dict
        Mapping from source IDs to indices
    target_mapping : dict, default None
        Mapping from target IDs to indices. If None, will use source_mapping.
    source_col : str, default None
        Column name for source node IDs
    target_col : str, default None
        Column name for target node IDs

    Returns
    -------
    list
        List of [source_idx, target_idx] pairs
    """
    if target_mapping is None:
        target_mapping = source_mapping

    # Detect columns if not provided
    source_col, target_col = _get_edge_columns(
        edge_gdf, source_col, target_col, source_mapping, target_mapping,
    )

    # Skip if we couldn't determine columns
    if (
        source_col is None
        or target_col is None
        or source_col not in edge_gdf.columns
        or target_col not in edge_gdf.columns
    ):
        return []

    # Convert IDs to strings once for better performance
    source_ids = edge_gdf[source_col].astype(str)
    target_ids = edge_gdf[target_col].astype(str)

    # Vectorized filtering for valid edges using pandas Series operations
    valid_src_mask = source_ids.isin(source_mapping.keys())
    valid_dst_mask = target_ids.isin(target_mapping.keys())
    valid_edges_mask = valid_src_mask & valid_dst_mask

    # Count missing IDs
    missing_src_count = (~valid_src_mask).sum()
    missing_dst_count = (~valid_dst_mask).sum()

    if missing_src_count > 0 or missing_dst_count > 0:
        logger.warning(
            "Missing source IDs: %d, missing target IDs: %d",
            missing_src_count,
            missing_dst_count,
        )

    # Process valid edges vectorized
    if not valid_edges_mask.any():
        return []

    valid_sources = source_ids[valid_edges_mask]
    valid_targets = target_ids[valid_edges_mask]

    # Vectorized mapping using pandas map
    from_indices = valid_sources.map(source_mapping).to_numpy()
    to_indices = valid_targets.map(target_mapping).to_numpy()

    # Create edge list
    return np.column_stack([from_indices, to_indices]).tolist()


# New helper to check if an edge GeoDataFrame is valid
def _is_valid_edge_df(edge_gdf: gpd.GeoDataFrame | None) -> bool:
    return edge_gdf is not None and not edge_gdf.empty


# Remove the is_hetero parameter and always expect dictionaries for nodes and edges.
# Updated _build_graph_data to accept node_y_attribute_cols parameter
# Modified _build_graph_data: renamed parameters and references
def _process_node_type(node_type: str,
                       node_gdf: gpd.GeoDataFrame,
                       node_id_cols: dict[str, str],
                       node_feature_cols: dict[str, list[str]],
                       node_label_cols: dict[str, list[str]] | None,
                       device: Union[str, "torch.device"],
                       data: HeteroData) -> dict[str, dict]:
    """Process a single node type and add to HeteroData."""
    if not isinstance(node_gdf, gpd.GeoDataFrame):
        logger.warning("Expected GeoDataFrame for node type %s, got %s", node_type, type(node_gdf))
        # Convert regular DataFrame to GeoDataFrame if it has x/y columns or lat/lon columns
        import pandas as pd
        if isinstance(node_gdf, pd.DataFrame):
            if "x" in node_gdf.columns and "y" in node_gdf.columns:
                # Vectorized Point creation
                geometry = gpd.points_from_xy(node_gdf.x, node_gdf.y)
                node_gdf = gpd.GeoDataFrame(node_gdf, geometry=geometry)
            elif "lat" in node_gdf.columns and "lon" in node_gdf.columns:
                # Vectorized Point creation
                geometry = gpd.points_from_xy(node_gdf.lon, node_gdf.lat)
                node_gdf = gpd.GeoDataFrame(node_gdf, geometry=geometry)
            else:
                return {}
        else:
            return {}

    id_col = node_id_cols.get(node_type)
    id_mapping, actual_id_col = _extract_node_id_mapping(node_gdf, id_col)

    feature_cols = node_feature_cols.get(node_type)
    data[node_type].x = _create_node_features(node_gdf, feature_cols, device)

    # Vectorized position extraction if geometry is present
    if "geometry" in node_gdf.columns and hasattr(node_gdf.geometry, "values"):
        # Use vectorized operations for position extraction
        geom_series = node_gdf.geometry

        # Check if all geometries have x, y attributes (Points)
        is_point_mask = geom_series.geom_type == "Point"

        if is_point_mask.all():
            # All points - direct coordinate extraction
            pos_data = np.column_stack([geom_series.x.to_numpy(), geom_series.y.to_numpy()])
        else:
            # Mixed geometries - use centroid for non-points
            pos_data = np.zeros((len(geom_series), 2))

            # Points - direct coordinates
            if is_point_mask.any():
                point_coords = np.column_stack([
                    geom_series[is_point_mask].x.to_numpy(),
                    geom_series[is_point_mask].y.to_numpy(),
                ])
                pos_data[is_point_mask] = point_coords

            # Non-points - centroids
            if (~is_point_mask).any():
                centroids = geom_series[~is_point_mask].centroid
                centroid_coords = np.column_stack([
                    centroids.x.to_numpy(),
                    centroids.y.to_numpy(),
                ])
                pos_data[~is_point_mask] = centroid_coords

        data[node_type].pos = torch.tensor(pos_data, dtype=torch.float, device=device)

    # Add label columns
    if node_label_cols and node_label_cols.get(node_type):
        data[node_type].y = _create_node_features(
            node_gdf, node_label_cols[node_type], device,
        )
    elif "y" in node_gdf.columns:
        data[node_type].y = torch.tensor(
            node_gdf["y"].to_numpy(), dtype=torch.float, device=device,
        )

    return {"mapping": id_mapping, "id_col": actual_id_col}


def _process_edge_type(edge_type: tuple[str, str, str],
                       edge_gdf: gpd.GeoDataFrame,
                       node_id_mappings: dict,
                       edge_source_cols: dict[tuple[str, str, str], str],
                       edge_target_cols: dict[tuple[str, str, str], str],
                       edge_feature_cols: dict[tuple[str, str, str], list[str]] | None,
                       device: Union[str, "torch.device"],
                       data: HeteroData) -> None:
    """Process a single edge type and add to HeteroData."""
    if not isinstance(edge_type, tuple) or len(edge_type) != 3:
        logger.warning(
            "Edge type key must be a tuple of (source_type, relation_type, "
            "target_type). Got %s instead. Skipping.",
            edge_type,
        )
        return

    src_type, rel_type, dst_type = edge_type

    if src_type not in node_id_mappings or dst_type not in node_id_mappings:
        logger.warning(
            "Edge type %s references node type(s) not present in nodes. Skipping.",
            edge_type,
        )
        return

    src_mapping = node_id_mappings[src_type]["mapping"]
    dst_mapping = node_id_mappings[dst_type]["mapping"]
    source_col = edge_source_cols.get(edge_type)
    target_col = edge_target_cols.get(edge_type)

    if _is_valid_edge_df(edge_gdf):
        pairs = _create_edge_idx_pairs(
            edge_gdf,
            source_mapping=src_mapping,
            target_mapping=dst_mapping,
            source_col=source_col,
            target_col=target_col,
        )
        if pairs:
            data[src_type, rel_type, dst_type].edge_index = torch.tensor(
                np.array(pairs).T, dtype=torch.long, device=device,
            )
        else:
            data[src_type, rel_type, dst_type].edge_index = torch.zeros(
                (2, 0), dtype=torch.long, device=device,
            )
        feature_cols = (
            edge_feature_cols.get(edge_type) if edge_feature_cols else None
        )
        data[src_type, rel_type, dst_type].edge_attr = _create_edge_features(
            edge_gdf, feature_cols, device,
        )
    else:
        data[src_type, rel_type, dst_type].edge_index = torch.zeros(
            (2, 0), dtype=torch.long, device=device,
        )
        data[src_type, rel_type, dst_type].edge_attr = torch.empty(
            (0, 0), dtype=torch.float, device=device,
        )


def _build_graph_data(nodes: dict[str, gpd.GeoDataFrame],
                      edges: dict[tuple[str, str, str], gpd.GeoDataFrame],
                      node_id_cols: dict[str, str],
                      node_feature_cols: dict[str, list[str]],
                      node_label_cols: dict[str, list[str]] | None,
                      edge_source_cols: dict[tuple[str, str, str], str],
                      edge_target_cols: dict[tuple[str, str, str], str],
                      edge_feature_cols: dict[tuple[str, str, str], list[str]] | None,
                      device: Union[str, "torch.device"] | None) -> HeteroData:
    """
    Build a heterogeneous graph (HeteroData) from node and edge GeoDataFrames.

    Parameters
    ----------
    nodes : dict
        Dictionary of node GeoDataFrames keyed by node type.
    edges : dict
        Dictionary of edge GeoDataFrames keyed by (source_type, relation, target_type).
    node_id_cols : dict
        Dictionary mapping node types to the ID column name.
    node_feature_cols : dict
        Dictionary mapping node types to lists of feature column names.
    node_label_cols : dict, optional
        Dictionary mapping node types to lists of label column names.
    edge_source_cols : dict
        Dictionary mapping edge type tuples to source column names.
    edge_target_cols : dict
        Dictionary mapping edge type tuples to target column names.
    edge_feature_cols : dict, optional
        Dictionary mapping edge type tuples to lists of edge feature columns.
    device : torch.device or str, optional
        Device to be used for tensor creation.

    Returns
    -------
    torch_geometric.data.HeteroData
        A PyTorch Geometric HeteroData graph object.
    """
    device = _get_device(device)
    data = HeteroData()
    node_id_mappings = {}

    # Process nodes across types
    for node_type, node_gdf in nodes.items():
        mapping_info = _process_node_type(
            node_type, node_gdf, node_id_cols, node_feature_cols,
            node_label_cols, device, data,
        )
        if mapping_info:
            node_id_mappings[node_type] = mapping_info

    # Process edges across types
    for edge_type, edge_gdf in edges.items():
        _process_edge_type(
            edge_type, edge_gdf, node_id_mappings, edge_source_cols,
            edge_target_cols, edge_feature_cols, device, data,
        )

    # Set CRS metadata from node GeoDataFrames
    crs_values = [gdf.crs for gdf in nodes.values() if hasattr(gdf, "crs") and gdf.crs]

    if not crs_values:
        data.crs = {}
    elif all(crs == crs_values[0] for crs in crs_values):
        data.crs = crs_values[0]
    else:
        msg = "CRS mismatch among node GeoDataFrames."
        raise ValueError(msg)

    # Store metadata for reconstruction
    _store_reconstruction_metadata(
        data,
        nodes=nodes,
        edges=edges,
        node_id_cols=node_id_cols,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_source_cols=edge_source_cols,
        edge_target_cols=edge_target_cols,
        edge_feature_cols=edge_feature_cols,
    )

    return data


def homogeneous_graph(nodes_gdf: gpd.GeoDataFrame,
                      edges_gdf: gpd.GeoDataFrame | None = None,
                      node_id_col: str | None = None,
                      node_feature_cols: list[str] | None = None,
                      node_label_cols: list[str] | None = None,
                      edge_source_col: str | None = None,
                      edge_target_col: str | None = None,
                      edge_feature_cols: list[str] | None = None,
                      device: Union[str, "torch.device"] | None = None) -> Data:
    """
    Create a homogeneous graph Data object from nodes and edges GeoDataFrames.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        GeoDataFrame containing node data.
    edges_gdf : GeoDataFrame, optional
        GeoDataFrame containing edge data.
    node_id_col : str, optional
        Column name that uniquely identifies each node.
    node_feature_cols : list of str, optional
        List of columns to use as node features.
    node_label_cols : list of str, optional
        List of columns to use as node labels.
    edge_source_col : str, optional
        Column name for source node IDs in the edge GeoDataFrame.
    edge_target_col : str, optional
        Column name for target node IDs in the edge GeoDataFrame.
    edge_feature_cols : list of str, optional
        List of columns to use as edge features.
    device : torch.device or str, optional
        Device for tensor creation.

    Returns
    -------
    torch_geometric.data.Data
        A PyTorch Geometric Data graph object.

    Raises
    ------
    ImportError
        If PyTorch and PyTorch Geometric are not installed
    """
    if not TORCH_AVAILABLE:
        msg = (
            "PyTorch and PyTorch Geometric are required for this function. "
            "Please install them using: poetry install --with torch or "
            "pip install city2graph[torch]"
        )
        raise ImportError(msg)

    # Preprocess homogeneous graph inputs into dictionaries.
    # Ensure at least empty edge type entry for homogeneous graphs
    nodes_dict = {"node": nodes_gdf}

    # Use explicit None check to avoid ambiguous truth of GeoDataFrame
    edges_dict = {
        ("node", "edge", "node"): edges_gdf
        if edges_gdf is not None
        else gpd.GeoDataFrame(),
    }
    node_id_cols = {"node": node_id_col} if node_id_col else {}
    node_feature_cols = {"node": node_feature_cols} if node_feature_cols else {}
    node_label_cols = {"node": node_label_cols} if node_label_cols else None
    edge_source_cols = {("node", "edge", "node"): edge_source_col}
    edge_target_cols = {("node", "edge", "node"): edge_target_col}

    # Wrap edge features into dict mapping for builder
    edge_feature_map = (
        {("node", "edge", "node"): edge_feature_cols}
        if edge_feature_cols is not None
        else None
    )

    hetero_data = _build_graph_data(
        nodes=nodes_dict,
        edges=edges_dict,
        node_id_cols=node_id_cols,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_source_cols=edge_source_cols,
        edge_target_cols=edge_target_cols,
        edge_feature_cols=edge_feature_map,
        device=device,
    )

    data = Data(
        x=hetero_data["node"].x,
        edge_index=hetero_data[("node", "edge", "node")].edge_index,
        edge_attr=hetero_data[("node", "edge", "node")].edge_attr,
        pos=hetero_data["node"].get("pos", None),
    )

    # Assign "y" node attribute if exists
    data.y = hetero_data["node"].get("y", None)

    # Assign CRS metadata from hetero_data
    data.crs = hetero_data.crs

    # Store metadata for reconstruction using the helper function
    _store_reconstruction_metadata(
        data,
        nodes=nodes_gdf,
        edges=edges_gdf,
        node_id_cols=node_id_col,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_source_cols=edge_source_col,
        edge_target_cols=edge_target_col,
        edge_feature_cols=edge_feature_cols,
    )

    return data


def _process_single_nodes_gdf(nodes_gdf: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    """Process a single nodes GeoDataFrame into a dictionary by type."""
    if "type" in nodes_gdf.columns:
        # Split by type column
        nodes_dict = {}
        for node_type in nodes_gdf["type"].unique():
            subset = nodes_gdf[nodes_gdf["type"] == node_type].copy()
            # Ensure we maintain GeoDataFrame type
            if not isinstance(subset, gpd.GeoDataFrame):
                geometry = subset.geometry if hasattr(subset, "geometry") else None
                subset = gpd.GeoDataFrame(subset, geometry=geometry)
            nodes_dict[node_type] = subset
        return nodes_dict
    # Default to single node type
    return {"default": nodes_gdf}


def _process_single_edges_gdf(edges_gdf: gpd.GeoDataFrame) -> dict[tuple[str, str, str], gpd.GeoDataFrame]:
    """Process a single edges GeoDataFrame into a dictionary by edge type."""
    if "edge_type" in edges_gdf.columns:
        # Split by edge_type column - assume format is "source_relation_target"
        edges_dict = {}
        for edge_type_str in edges_gdf["edge_type"].unique():
            # Try to parse edge type string
            parts = str(edge_type_str).split("_", 2)
            edge_key = tuple(parts) if len(parts) == 3 else ("default", "edge", "default")
            edges_dict[edge_key] = edges_gdf[edges_gdf["edge_type"] == edge_type_str].copy()
        return edges_dict
    # Default to single edge type
    return {("default", "edge", "default"): edges_gdf}


def heterogeneous_graph(nodes_dict: dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame,
                        edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] | gpd.GeoDataFrame,
                        node_id_cols: dict[str, str] | None = None,
                        node_feature_cols: dict[str, list[str]] | None = None,
                        node_label_cols: dict[str, list[str]] | None = None,
                        edge_source_cols: dict[tuple[str, str, str], str] | None = None,
                        edge_target_cols: dict[tuple[str, str, str], str] | None = None,
                        edge_feature_cols: dict[tuple[str, str, str], list[str]] | None = None,
                        device: Union[str, "torch.device"] | None = None) -> HeteroData:
    """
    Create a heterogeneous graph HeteroData object from node and edge dictionaries.

    Parameters
    ----------
    nodes_dict : dict or GeoDataFrame
        Dictionary of GeoDataFrames for each node type, or a single GeoDataFrame
        with a 'type' column to automatically split by node type.
    edges_dict : dict or GeoDataFrame
        Dictionary of GeoDataFrames for each edge type, with keys as (source_type, relation, target_type),
        or a single GeoDataFrame with an 'edge_type' column.
    node_id_cols : dict, optional
        Dictionary mapping node types to their ID column.
    node_feature_cols : dict, optional
        Dictionary mapping node types to lists of feature columns.
    node_label_cols : dict, optional
        Dictionary mapping node types to lists of label columns.
    edge_source_cols : dict, optional
        Dictionary mapping edge types to source column names.
    edge_target_cols : dict, optional
        Dictionary mapping edge types to target column names.
    edge_feature_cols : dict, optional
        Dictionary mapping edge types to lists of edge attribute columns.
    device : torch.device or str, optional
        Device for tensor creation.

    Returns
    -------
    torch_geometric.data.HeteroData
        A PyTorch Geometric HeteroData graph object.

    Raises
    ------
    ImportError
        If PyTorch and PyTorch Geometric are not installed
    """
    if not TORCH_AVAILABLE:
        msg = (
            "PyTorch and PyTorch Geometric are required for this function. "
            "Please install them using: poetry install --with torch or "
            "pip install city2graph[torch]"
        )
        raise ImportError(msg)

    if node_id_cols is None:
        node_id_cols = {}
    if node_feature_cols is None:
        node_feature_cols = {}
    if edge_source_cols is None:
        edge_source_cols = {}
    if edge_target_cols is None:
        edge_target_cols = {}

    # Handle case where nodes_dict is a single GeoDataFrame
    if isinstance(nodes_dict, gpd.GeoDataFrame):
        nodes_dict = _process_single_nodes_gdf(nodes_dict)

    # Handle case where edges_dict is a single GeoDataFrame
    if isinstance(edges_dict, gpd.GeoDataFrame):
        edges_dict = _process_single_edges_gdf(edges_dict)

    return _build_graph_data(
        nodes=nodes_dict,
        edges=edges_dict,
        node_id_cols=node_id_cols,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_source_cols=edge_source_cols,
        edge_target_cols=edge_target_cols,
        edge_feature_cols=edge_feature_cols,
        device=device,
    )


def from_morphological_graph(network_output: dict,  # noqa: PLR0915
                             private_id_col: str = "tess_id",
                             public_id_col: str = "id",
                             private_node_feature_cols: list[str] | None = None,
                             public_node_feature_cols: list[str] | None = None,
                             device: Union[str, "torch.device"] | None = None) -> HeteroData | Data:
    """
    Create a graph representation from the output of morphological_graph.

    Parameters
    ----------
    network_output : dict
        Output dictionary from morphological_graph containing:
        - 'tessellation': GeoDataFrame of tessellation cells (private spaces)
        - 'segments': GeoDataFrame of road segments (public spaces)
        - 'private_to_private': GeoDataFrame of connections between tessellation cells
        - 'public_to_public': GeoDataFrame of connections between road segments
        - 'private_to_public': GeoDataFrame of connections between tessellation cells and road segments
    private_id_col : str, default='tess_id'
        Column name in tessellation GeoDataFrame that uniquely identifies each private space.
    public_id_col : str, default='id'
        Column name in segments GeoDataFrame that uniquely identifies each public space.
    private_node_feature_cols : list, default None
        Attributes in tessellation GeoDataFrame to use as node features.
    public_node_feature_cols : list, default None
        Attributes in segments GeoDataFrame to use as node features.
    device : str, default None
        Device to use for tensors. Must be 'cuda' or 'cpu' if provided.
        If None, will use CUDA if available, otherwise CPU.

    Returns
    -------
    torch_geometric.data.HeteroData or torch_geometric.data.Data
        Graph representation. HeteroData is returned if both node types exist,
        otherwise a homogeneous Data object.

    Raises
    ------
    ImportError
        If PyTorch and PyTorch Geometric are not installed
    ValueError
        If required data is missing from the network_output dictionary
    """
    if not TORCH_AVAILABLE:
        msg = (
            "PyTorch and PyTorch Geometric are required for this function. "
            "Please install them using: poetry install --with torch or "
            "pip install city2graph[torch]"
        )
        raise ImportError(msg)

    # Validate device
    device = _get_device(device)

    # Extract data from network_output
    if not isinstance(network_output, dict):
        msg = "network_output must be a dictionary returned from morphological_graph"
        raise TypeError(msg)

    # Check if we have the new pyg_to_gdf compatible structure
    if "nodes" in network_output and "edges" in network_output:
        # New structure - extract from nodes/edges dictionaries
        nodes_dict = network_output["nodes"]
        edges_dict = network_output["edges"]

        private_gdf = nodes_dict.get("private")
        public_gdf = nodes_dict.get("public")
        private_to_private_gdf = edges_dict.get(("private", "touched_to", "private"))
        public_to_public_gdf = edges_dict.get(("public", "connected_to", "public"))
        private_to_public_gdf = edges_dict.get(("private", "faced_to", "public"))
    else:
        # Legacy structure - extract from flat dictionary
        private_gdf = network_output.get("tessellation")
        public_gdf = network_output.get("segments")
        private_to_private_gdf = network_output.get("private_to_private")
        public_to_public_gdf = network_output.get("public_to_public")
        private_to_public_gdf = network_output.get("private_to_public")

    # Validate that required data exists
    has_private = private_gdf is not None and not private_gdf.empty
    has_public = public_gdf is not None and not public_gdf.empty

    # Case 1: We have both private and public nodes - create heterogeneous graph
    if has_private and has_public:
        # Create nodes dictionary
        nodes_dict = {"private": private_gdf, "public": public_gdf}

        # Create edges dictionary with edge type tuples
        edges_dict = {
            ("private", "touched_to", "private"): private_to_private_gdf,
            ("private", "faced_to", "public"): private_to_public_gdf,
            ("public", "connected_to", "public"): public_to_public_gdf,
        }

        # Create node ID columns dictionary
        node_id_cols = {"private": private_id_col, "public": public_id_col}

        # Create node feature columns dictionary
        node_feature_cols = {}
        if private_node_feature_cols is not None:
            node_feature_cols["private"] = private_node_feature_cols
        if public_node_feature_cols is not None:
            node_feature_cols["public"] = public_node_feature_cols

        # Prepare edge source/target column mappings based on morphological_graph output columns
        edge_source_cols = {
            ("private", "touched_to", "private"): "from_private_id",
            ("private", "faced_to", "public"): "private_id",
            ("public", "connected_to", "public"): "from_public_id",
        }

        edge_target_cols = {
            ("private", "touched_to", "private"): "to_private_id",
            ("private", "faced_to", "public"): "public_id",
            ("public", "connected_to", "public"): "to_public_id",
        }

        # Create the heterogeneous graph using _build_graph_data
        return _build_graph_data(
            nodes=nodes_dict,
            edges=edges_dict,
            node_id_cols=node_id_cols,
            node_feature_cols=node_feature_cols,
            node_label_cols=None,
            edge_source_cols=edge_source_cols,
            edge_target_cols=edge_target_cols,
            edge_feature_cols=None,
            device=device,
        )

    # Case 2: We only have private nodes - create homogeneous graph
    if has_private:
        nodes_dict = {"node": private_gdf}
        edges_dict = {("node", "edge", "node"): private_to_private_gdf or gpd.GeoDataFrame()}
        node_id_cols_dict = {"node": private_id_col}
        node_feature_cols_dict = {"node": private_node_feature_cols} if private_node_feature_cols else {}
        edge_source_cols_dict = {("node", "edge", "node"): "from_private_id"}
        edge_target_cols_dict = {("node", "edge", "node"): "to_private_id"}

        hetero_data = _build_graph_data(
            nodes=nodes_dict,
            edges=edges_dict,
            node_id_cols=node_id_cols_dict,
            node_feature_cols=node_feature_cols_dict,
            node_label_cols=None,
            edge_source_cols=edge_source_cols_dict,
            edge_target_cols=edge_target_cols_dict,
            edge_feature_cols=None,
            device=device,
        )

        return Data(
            x=hetero_data["node"].x,
            edge_index=hetero_data[("node", "edge", "node")].edge_index,
            edge_attr=hetero_data[("node", "edge", "node")].edge_attr,
            pos=hetero_data["node"].get("pos", None),
            y=hetero_data["node"].get("y", None),
            crs=hetero_data.crs,
        )

    # Case 3: We only have public nodes - create homogeneous graph
    if has_public:
        nodes_dict = {"node": public_gdf}
        edges_dict = {("node", "edge", "node"): public_to_public_gdf or gpd.GeoDataFrame()}
        node_id_cols_dict = {"node": public_id_col}
        node_feature_cols_dict = {"node": public_node_feature_cols} if public_node_feature_cols else {}
        edge_source_cols_dict = {("node", "edge", "node"): "from_public_id"}
        edge_target_cols_dict = {("node", "edge", "node"): "to_public_id"}

        hetero_data = _build_graph_data(
            nodes=nodes_dict,
            edges=edges_dict,
            node_id_cols=node_id_cols_dict,
            node_feature_cols=node_feature_cols_dict,
            node_label_cols=None,
            edge_source_cols=edge_source_cols_dict,
            edge_target_cols=edge_target_cols_dict,
            edge_feature_cols=None,
            device=device,
        )

        return Data(
            x=hetero_data["node"].x,
            edge_index=hetero_data[("node", "edge", "node")].edge_index,
            edge_attr=hetero_data[("node", "edge", "node")].edge_attr,
            pos=hetero_data["node"].get("pos", None),
            y=hetero_data["node"].get("y", None),
            crs=hetero_data.crs,
        )

    # Case 4: No valid nodes - raise an error to prevent unintended empty graphs.
    msg = "No valid node data provided; no nodes found."
    raise ValueError(msg)


def _extract_tensor_features(
    tensor: "torch.Tensor",
    column_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract features from tensor into a dictionary with column names."""
    if tensor is None or tensor.numel() == 0:
        return {}

    # Convert to numpy for faster operations
    features_array = tensor.detach().cpu().numpy()

    if column_names is None:
        # Generate default column names
        num_features = (
            features_array.shape[1] if len(features_array.shape) > 1 else 1
        )
        column_names = [f"feature_{i}" for i in range(num_features)]

    # Create dictionary with vectorized operations
    if len(features_array.shape) == 1:
        return {column_names[0]: features_array}

    return {
        name: features_array[:, i]
        for i, name in enumerate(column_names[: features_array.shape[1]])
    }


def _create_geometries_from_pos(
    pos_tensor: "torch.Tensor",
) -> gpd.array.GeometryArray:
    """Create Point geometries from position tensor using vectorized operations."""
    if pos_tensor is None or pos_tensor.numel() == 0:
        return gpd.array.from_shapely([])

    pos_array = pos_tensor.detach().cpu().numpy()

    # Vectorized Point creation using geopandas
    if len(pos_array.shape) == 2 and pos_array.shape[1] >= 2:
        return gpd.points_from_xy(pos_array[:, 0], pos_array[:, 1])

    return gpd.array.from_shapely([])


def _reconstruct_node_gdf(
    node_type: str,
    data: Data | HeteroData,
    is_hetero: bool = False,
) -> gpd.GeoDataFrame:
    """Reconstruct node GeoDataFrame from PyTorch Geometric data."""
    # Get node data based on graph type
    node_data = data[node_type] if is_hetero else data

    # Initialize data dictionary
    gdf_data = {}

    # Extract node features with proper column names
    if hasattr(node_data, "x") and node_data.x is not None:
        # Get stored feature column names
        feature_cols = getattr(data, "_node_feature_columns", {}).get(node_type, None)
        # If no stored names, try alternative storage location
        if feature_cols is None and hasattr(data, "_node_columns"):
            stored_cols = getattr(data, "_node_columns", {}).get(node_type, [])
            # Filter out non-feature columns
            feature_cols = [col for col in stored_cols if col not in ["geometry", "pos"]]

        features_dict = _extract_tensor_features(node_data.x, feature_cols)
        gdf_data.update(features_dict)

    # Extract node labels with proper column names
    if hasattr(node_data, "y") and node_data.y is not None:
        # Get stored label column names
        label_cols = getattr(data, "_node_label_columns", {}).get(node_type, None)
        labels_dict = _extract_tensor_features(node_data.y, label_cols)
        gdf_data.update(labels_dict)

    # Create geometry from positions
    geometry = None
    if hasattr(node_data, "pos") and node_data.pos is not None:
        geometry = _create_geometries_from_pos(node_data.pos)

    # Create GeoDataFrame
    if gdf_data:
        gdf = gpd.GeoDataFrame(gdf_data, geometry=geometry)
    else:
        # Create minimal GeoDataFrame with geometry if available
        num_nodes = (
            node_data.x.size(0)
            if hasattr(node_data, "x") and node_data.x is not None
            else 0
        )
        if num_nodes == 0 and geometry is not None:
            num_nodes = len(geometry)

        gdf = gpd.GeoDataFrame(
            {"node_id": range(num_nodes)} if num_nodes > 0 else {},
            geometry=geometry,
        )

    # Set CRS if available
    if hasattr(data, "crs") and data.crs:
        gdf.crs = data.crs

    return gdf


def _reconstruct_edge_gdf(
    edge_type: tuple[str, str, str] | str,
    data: Data | HeteroData,
    is_hetero: bool = False,
) -> pd.DataFrame:
    """Reconstruct edge DataFrame from PyTorch Geometric data."""
    # Get edge data based on graph type
    edge_data = data[edge_type] if is_hetero else data

    # Initialize data dictionary
    edge_data_dict = {}

    # Extract edge indices
    if hasattr(edge_data, "edge_index") and edge_data.edge_index is not None:
        edge_index_array = edge_data.edge_index.detach().cpu().numpy()
        if edge_index_array.shape[0] == 2:
            edge_data_dict["source"] = edge_index_array[0]
            edge_data_dict["target"] = edge_index_array[1]

    # Extract edge features with proper column names
    if hasattr(edge_data, "edge_attr") and edge_data.edge_attr is not None:
        # Get stored feature column names
        feature_cols = getattr(data, "_edge_feature_columns", {}).get(edge_type, None)
        # If no stored names, try alternative storage location
        if feature_cols is None and hasattr(data, "_edge_columns"):
            stored_cols = getattr(data, "_edge_columns", {}).get(edge_type, [])
            # Filter out non-feature columns like geometry
            feature_cols = [col for col in stored_cols if col not in ["geometry"]]

        features_dict = _extract_tensor_features(edge_data.edge_attr, feature_cols)
        edge_data_dict.update(features_dict)

    # Create DataFrame (edges typically don't have geometry)
    return pd.DataFrame(edge_data_dict) if edge_data_dict else pd.DataFrame()


def pyg_to_gdf(
    data: Data | HeteroData,
) -> dict[str, dict[str, gpd.GeoDataFrame | pd.DataFrame]] | tuple[
    gpd.GeoDataFrame, pd.DataFrame | None,
]:
    """
    Convert PyTorch Geometric Data or HeteroData to GeoDataFrames and DataFrames.

    Parameters
    ----------
    data : Data or HeteroData
        PyTorch Geometric graph object to convert.

    Returns
    -------
    dict or tuple
        For HeteroData: Returns a dictionary with keys 'nodes' and 'edges',
        where 'nodes' contains GeoDataFrames and 'edges' contains DataFrames.
        For Data: Returns a tuple of (nodes_gdf, edges_df).

    Raises
    ------
    ImportError
        If PyTorch and PyTorch Geometric are not installed
    """
    if not TORCH_AVAILABLE:
        msg = (
            "PyTorch and PyTorch Geometric are required for this function. "
            "Please install them using: poetry install --with torch or "
            "pip install city2graph[torch]"
        )
        raise ImportError(msg)

    # Check if it's heterogeneous data
    is_hetero = hasattr(data, "node_types") and hasattr(data, "edge_types")

    if is_hetero:
        # Handle HeteroData
        nodes_dict = {}
        edges_dict = {}

        # Reconstruct node GeoDataFrames for each node type
        for node_type in data.node_types:
            nodes_dict[node_type] = _reconstruct_node_gdf(
                node_type, data, is_hetero=True,
            )

        # Reconstruct edge DataFrames for each edge type
        for edge_type in data.edge_types:
            edges_dict[edge_type] = _reconstruct_edge_gdf(
                edge_type, data, is_hetero=True,
            )

        return {"nodes": nodes_dict, "edges": edges_dict}

    # Handle homogeneous Data
    nodes_gdf = _reconstruct_node_gdf("node", data, is_hetero=False)

    # Check if edges exist
    if (
        hasattr(data, "edge_index")
        and data.edge_index is not None
        and data.edge_index.numel() > 0
    ):
        edges_df = _reconstruct_edge_gdf("edge", data, is_hetero=False)
    else:
        edges_df = None

    return nodes_gdf, edges_df


def pyg_to_nx(data: Data | HeteroData) -> nx.Graph:
    """
    Convert PyTorch Geometric Data or HeteroData to NetworkX graph.

    Parameters
    ----------
    data : Data or HeteroData
        PyTorch Geometric graph object to convert.

    Returns
    -------
    nx.Graph
        NetworkX graph representation. For HeteroData, returns a MultiDiGraph
        with node and edge type information. For homogeneous Data, returns
        a standard Graph.

    Raises
    ------
    ImportError
        If PyTorch and PyTorch Geometric are not installed
    """
    if not TORCH_AVAILABLE:
        msg = (
            "PyTorch and PyTorch Geometric are required for this function. "
            "Please install them using: poetry install --with torch or "
            "pip install city2graph[torch]"
        )
        raise ImportError(msg)

    # Check if it's heterogeneous data
    is_hetero = hasattr(data, "node_types") and hasattr(data, "edge_types")

    # Get stored attribute column names for reconstruction
    node_feature_cols = getattr(data, "_node_feature_columns", {})
    node_label_cols = getattr(data, "_node_label_columns", {})
    edge_feature_cols = getattr(data, "_edge_feature_columns", {})

    # Determine node and edge attributes to include
    if is_hetero:
        # For heterogeneous data, collect all unique attribute names across node types
        all_node_attrs = set()
        for node_type in data.node_types:
            if node_type in node_feature_cols:
                all_node_attrs.update(node_feature_cols[node_type])
            if node_type in node_label_cols:
                all_node_attrs.update(node_label_cols[node_type])

        # Collect all unique edge attribute names across edge types
        all_edge_attrs = set()
        for edge_type in data.edge_types:
            if edge_type in edge_feature_cols:
                all_edge_attrs.update(edge_feature_cols[edge_type])

        node_attrs = list(all_node_attrs) if all_node_attrs else None
        edge_attrs = list(all_edge_attrs) if all_edge_attrs else None

        # Use to_multi=True for heterogeneous graphs to preserve multiple edges
        graph = pyg_to_networkx(
            data,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            to_undirected=False,
            to_multi=True,
        )
    else:
        # For homogeneous data, get attributes for single node/edge type
        node_attrs = []
        if "node" in node_feature_cols:
            node_attrs.extend(node_feature_cols["node"])
        if "node" in node_label_cols:
            node_attrs.extend(node_label_cols["node"])

        edge_attrs = edge_feature_cols.get(("node", "edge", "node"), [])

        node_attrs = node_attrs if node_attrs else None
        edge_attrs = edge_attrs if edge_attrs else None

        # Use standard conversion for homogeneous graphs
        graph = pyg_to_networkx(
            data,
            node_attrs=node_attrs,
            edge_attrs=edge_attrs,
            to_undirected=False,
        )

    # Preserve global attributes
    if hasattr(data, "crs") and data.crs:
        graph.graph["crs"] = data.crs

    return graph





def _determine_graph_type(
    nodes: dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | gpd.GeoDataFrame | None,
) -> str:
    """Determine if the graph should be homogeneous or heterogeneous."""
    if isinstance(nodes, dict) and len(nodes) > 1:
        return "heterogeneous"
    if isinstance(edges, dict) and len(edges) > 1:
        return "heterogeneous"
    # Check if edges dict has complex edge types
    if isinstance(edges, dict) and edges:
        for edge_type in edges:
            if isinstance(edge_type, tuple) and len(edge_type) == 3:
                src_type, relation, dst_type = edge_type
                if src_type != dst_type or relation != "edge":
                    return "heterogeneous"
    return "homogeneous"


def gdf_to_pyg(  # noqa: PLR0912
    nodes: dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | gpd.GeoDataFrame | None = None,
    node_id_cols: dict[str, str] | str | None = None,
    node_feature_cols: dict[str, list[str]] | list[str] | None = None,
    node_label_cols: dict[str, list[str]] | list[str] | None = None,
    edge_source_cols: dict[tuple[str, str, str], str] | str | None = None,
    edge_target_cols: dict[tuple[str, str, str], str] | str | None = None,
    edge_feature_cols: dict[tuple[str, str, str], list[str]] | list[str] | None = None,
    device: Union[str, "torch.device", None] = None,
) -> Data | HeteroData:
    """
    Convert GeoDataFrames to PyTorch Geometric graph objects with automatic type detection.

    Parameters
    ----------
    nodes : dict or GeoDataFrame
        Dictionary of GeoDataFrames for each node type, or a single GeoDataFrame.
    edges : dict, GeoDataFrame, or None
        Dictionary of GeoDataFrames for each edge type, single GeoDataFrame, or None.
    node_id_cols : dict, str, or None
        Node ID column specification.
    node_feature_cols : dict, list, or None
        Node feature columns specification.
    node_label_cols : dict, list, or None
        Node label columns specification.
    edge_source_cols : dict, str, or None
        Edge source column specification.
    edge_target_cols : dict, str, or None
        Edge target column specification.
    edge_feature_cols : dict, list, or None
        Edge feature columns specification.
    device : torch.device or str, optional
        Device for tensor creation.

    Returns
    -------
    Data or HeteroData
        PyTorch Geometric graph object.

    Raises
    ------
    ImportError
        If PyTorch and PyTorch Geometric are not installed
    """
    if not TORCH_AVAILABLE:
        msg = (
            "PyTorch and PyTorch Geometric are required for this function. "
            "Please install them using: poetry install --with torch or "
            "pip install city2graph[torch]"
        )
        raise ImportError(msg)

    # Validate input data types
    if isinstance(nodes, dict):
        for node_gdf in nodes.values():
            _validate_gdf(node_gdf, None)
    else:
        _validate_gdf(nodes, None)

    if edges is not None:
        if isinstance(edges, dict):
            for edge_gdf in edges.values():
                _validate_gdf(None, edge_gdf)
        else:
            _validate_gdf(None, edges)

    # Determine graph type and delegate to appropriate function
    if _determine_graph_type(nodes, edges) == "heterogeneous":
        # Handle case where nodes_dict is a single GeoDataFrame
        if isinstance(nodes, gpd.GeoDataFrame):
            nodes = _process_single_nodes_gdf(nodes)

        # Handle case where edges_dict is a single GeoDataFrame
        if isinstance(edges, gpd.GeoDataFrame):
            edges = _process_single_edges_gdf(edges)

        # Convert parameters to appropriate dictionaries
        if node_id_cols is None:
            node_id_cols = {}
        if node_feature_cols is None:
            node_feature_cols = {}
        if edge_source_cols is None:
            edge_source_cols = {}
        if edge_target_cols is None:
            edge_target_cols = {}

        return _build_graph_data(
            nodes=nodes,
            edges=edges,
            node_id_cols=node_id_cols,
            node_feature_cols=node_feature_cols,
            node_label_cols=node_label_cols,
            edge_source_cols=edge_source_cols,
            edge_target_cols=edge_target_cols,
            edge_feature_cols=edge_feature_cols,
            device=device,
        )

    # Extract single values for homogeneous case
    nodes_gdf = next(iter(nodes.values())) if isinstance(nodes, dict) else nodes
    edges_gdf = next(iter(edges.values())) if isinstance(edges, dict) and edges else edges

    # Extract single column specifications
    node_id_col = (
        next(iter(node_id_cols.values()))
        if isinstance(node_id_cols, dict)
        else node_id_cols
    )
    node_feature_col_list = (
        next(iter(node_feature_cols.values()))
        if isinstance(node_feature_cols, dict)
        else node_feature_cols
    )
    node_label_col_list = (
        next(iter(node_label_cols.values()))
        if isinstance(node_label_cols, dict)
        else node_label_cols
    )
    edge_source_col = (
        next(iter(edge_source_cols.values()))
        if isinstance(edge_source_cols, dict)
        else edge_source_cols
    )
    edge_target_col = (
        next(iter(edge_target_cols.values()))
        if isinstance(edge_target_cols, dict)
        else edge_target_cols
    )
    edge_feature_col_list = (
        next(iter(edge_feature_cols.values()))
        if isinstance(edge_feature_cols, dict)
        else edge_feature_cols
    )

    # Build homogeneous graph using _build_graph_data
    nodes_dict = {"node": nodes_gdf}
    edges_dict = {
        ("node", "edge", "node"): edges_gdf
        if edges_gdf is not None
        else gpd.GeoDataFrame(),
    }
    node_id_cols_dict = {"node": node_id_col} if node_id_col else {}
    node_feature_cols_dict = {"node": node_feature_col_list} if node_feature_col_list else {}
    node_label_cols_dict = {"node": node_label_col_list} if node_label_col_list else None
    edge_source_cols_dict = {("node", "edge", "node"): edge_source_col}
    edge_target_cols_dict = {("node", "edge", "node"): edge_target_col}
    edge_feature_map = (
        {("node", "edge", "node"): edge_feature_col_list}
        if edge_feature_col_list is not None
        else None
    )

    hetero_data = _build_graph_data(
        nodes=nodes_dict,
        edges=edges_dict,
        node_id_cols=node_id_cols_dict,
        node_feature_cols=node_feature_cols_dict,
        node_label_cols=node_label_cols_dict,
        edge_source_cols=edge_source_cols_dict,
        edge_target_cols=edge_target_cols_dict,
        edge_feature_cols=edge_feature_map,
        device=device,
    )

    # Convert to homogeneous Data object
    data = Data(
        x=hetero_data["node"].x,
        edge_index=hetero_data[("node", "edge", "node")].edge_index,
        edge_attr=hetero_data[("node", "edge", "node")].edge_attr,
        pos=hetero_data["node"].get("pos", None),
    )
    data.y = hetero_data["node"].get("y", None)
    data.crs = hetero_data.crs
    return data


def nx_to_pyg(
    graph: nx.Graph,
    node_feature_attrs: list[str] | None = None,
    node_label_attrs: list[str] | None = None,
    edge_feature_attrs: list[str] | None = None,
    device: Union[str, "torch.device"] | None = None,
) -> Data | HeteroData:
    """
    Convert NetworkX graph to PyTorch Geometric graph with automatic type detection.

    Parameters
    ----------
    graph : networkx.Graph
        NetworkX graph to convert.
    node_feature_attrs : list of str, optional
        List of node attributes to use as features.
    node_label_attrs : list of str, optional
        List of node attributes to use as labels.
    edge_feature_attrs : list of str, optional
        List of edge attributes to use as features.
    device : torch.device or str, optional
        Device for tensor creation.

    Returns
    -------
    Data or HeteroData
        PyTorch Geometric graph object.

    Raises
    ------
    ImportError
        If PyTorch and PyTorch Geometric are not installed
    ValueError
        If the graph is empty
    """
    if not TORCH_AVAILABLE:
        msg = (
            "PyTorch and PyTorch Geometric are required for this function. "
            "Please install them using: poetry install --with torch or "
            "pip install city2graph[torch]"
        )
        raise ImportError(msg)

    # Validate NetworkX graph
    _validate_nx(graph)

    if len(graph.nodes()) == 0:
        msg = "Graph has no nodes"
        raise ValueError(msg)

    # Check if graph has heterogeneous structure using vectorized operations
    node_data_list = list(graph.nodes(data=True))
    edge_data_list = list(graph.edges(data=True))

    # Extract node types efficiently
    node_types = {data.get("node_type", "default") for _, data in node_data_list}

    # Extract edge types efficiently
    edge_types = set()
    for src, dst, edge_data in edge_data_list:
        src_type = graph.nodes[src].get("node_type", "default")
        dst_type = graph.nodes[dst].get("node_type", "default")
        relation = edge_data.get("relation", edge_data.get("edge_type", "edge"))
        edge_types.add((src_type, relation, dst_type))

    # Determine if heterogeneous
    is_hetero = len(node_types) > 1 or len(edge_types) > 1 or any(
        src_type != dst_type or relation != "edge"
        for src_type, relation, dst_type in edge_types
    )

    if is_hetero:
        return _nx_to_hetero_pyg(graph, node_feature_attrs, node_label_attrs, edge_feature_attrs, device)
    return _nx_to_homo_pyg(graph, node_feature_attrs, node_label_attrs, edge_feature_attrs, device)


def _nx_to_homo_pyg(graph: nx.Graph,
                    node_feature_attrs: list[str] | None,
                    node_label_attrs: list[str] | None,
                    edge_feature_attrs: list[str] | None,
                    device: Union[str, "torch.device"] | None) -> Data:
    """Convert NetworkX graph to homogeneous PyTorch Geometric Data object."""
    device = _get_device(device)

    # Create node mapping using vectorized operations
    nodes_list = list(graph.nodes())
    node_mapping = {node: i for i, node in enumerate(nodes_list)}
    num_nodes = len(nodes_list)

    # Vectorized node data extraction
    nodes_data = [graph.nodes[node] for node in nodes_list]

    # Extract node features vectorized
    if node_feature_attrs:
        feature_matrix = np.array([
            [data.get(attr, 0.0) for attr in node_feature_attrs]
            for data in nodes_data
        ], dtype=np.float32)
        x = torch.from_numpy(feature_matrix).to(device=device, dtype=torch.float)
    else:
        x = torch.zeros((num_nodes, 0), dtype=torch.float, device=device)

    # Extract node labels vectorized
    y = None
    if node_label_attrs:
        label_matrix = np.array([
            [data.get(attr, 0.0) for attr in node_label_attrs]
            for data in nodes_data
        ], dtype=np.float32)
        y = torch.from_numpy(label_matrix).to(device=device, dtype=torch.float)

    # Vectorized edge extraction
    edges_data = list(graph.edges(data=True))
    if edges_data:
        # Extract edge indices vectorized
        edge_array = np.array([
            [node_mapping[src], node_mapping[dst]]
            for src, dst, _ in edges_data
        ])
        edge_index = torch.from_numpy(edge_array.T).to(device=device, dtype=torch.long)

        # Extract edge features vectorized
        if edge_feature_attrs:
            edge_feature_matrix = np.array([
                [data.get(attr, 0.0) for attr in edge_feature_attrs]
                for _, _, data in edges_data
            ], dtype=np.float32)
            edge_attr = torch.from_numpy(edge_feature_matrix).to(device=device, dtype=torch.float)
        else:
            edge_attr = torch.zeros((len(edges_data), 0), dtype=torch.float, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.zeros((0, 0), dtype=torch.float, device=device)

    # Extract positional information vectorized
    pos = None
    if all("pos" in data for data in nodes_data):
        pos_matrix = np.array([data["pos"] for data in nodes_data], dtype=np.float32)
        pos = torch.from_numpy(pos_matrix).to(device=device, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)

    # Preserve CRS if available
    if "crs" in graph.graph:
        data.crs = graph.graph["crs"]

    # Store metadata for reconstruction
    _store_nx_metadata(data, graph, node_feature_attrs, node_label_attrs, edge_feature_attrs)

    return data


def _extract_nx_node_data(graph: nx.Graph, node_types: dict,
                          node_feature_attrs: list[str] | None,
                          node_label_attrs: list[str] | None,
                          device: "torch.device") -> tuple[dict, dict]:
    """Extract node features and labels vectorized by type."""
    node_mappings = {}
    data_store = {}

    for node_type, nodes in node_types.items():
        node_mappings[node_type] = {node: i for i, node in enumerate(nodes)}

        if not nodes:
            continue

        # Vectorized node data extraction
        all_node_data = [graph.nodes[node] for node in nodes]

        # Features
        if node_feature_attrs:
            feature_matrix = np.array([
                [data.get(attr, 0.0) for attr in node_feature_attrs]
                for data in all_node_data
            ], dtype=np.float32)
            data_store[f"{node_type}_x"] = torch.from_numpy(feature_matrix).to(device=device,
                                                                               dtype=torch.float)
        else:
            data_store[f"{node_type}_x"] = torch.zeros((len(nodes), 0), dtype=torch.float, device=device)

        # Labels
        if node_label_attrs:
            label_matrix = np.array([
                [data.get(attr, 0.0) for attr in node_label_attrs]
                for data in all_node_data
            ], dtype=np.float32)
            data_store[f"{node_type}_y"] = torch.from_numpy(label_matrix).to(device=device, dtype=torch.float)

        # Positions
        if all("pos" in data for data in all_node_data):
            pos_matrix = np.array([data["pos"] for data in all_node_data], dtype=np.float32)
            data_store[f"{node_type}_pos"] = torch.from_numpy(pos_matrix).to(device=device, dtype=torch.float)

    return node_mappings, data_store


def _extract_nx_edge_data(graph: nx.Graph, node_mappings: dict,
                          edge_feature_attrs: list[str] | None,
                          device: "torch.device") -> dict:
    """Extract edge data vectorized by type."""
    # Group edges by type using vectorized operations
    edge_groups = {}
    edges_list = list(graph.edges(data=True))

    # Vectorized edge type extraction
    for src, dst, edge_data in edges_list:
        src_type = graph.nodes[src].get("node_type", "default")
        dst_type = graph.nodes[dst].get("node_type", "default")
        relation = edge_data.get("relation", edge_data.get("edge_type", "edge"))
        edge_type = (src_type, relation, dst_type)

        if edge_type not in edge_groups:
            edge_groups[edge_type] = []
        edge_groups[edge_type].append((src, dst, edge_data))

    edge_tensors = {}
    for edge_type, edges in edge_groups.items():
        src_type, relation, dst_type = edge_type

        if not edges:
            continue

        # Vectorized edge processing
        edge_indices = np.array([
            [node_mappings[src_type][src], node_mappings[dst_type][dst]]
            for src, dst, _ in edges
        ])
        edge_tensors[edge_type] = {
            "edge_index": torch.from_numpy(edge_indices.T).to(device=device, dtype=torch.long),
        }

        # Edge features vectorized
        if edge_feature_attrs:
            edge_feature_matrix = np.array([
                [data.get(attr, 0.0) for attr in edge_feature_attrs]
                for _, _, data in edges
            ], dtype=np.float32)
            edge_tensors[edge_type]["edge_attr"] = torch.from_numpy(edge_feature_matrix).to(
                device=device, dtype=torch.float,
            )
        else:
            edge_tensors[edge_type]["edge_attr"] = torch.zeros(
                (len(edges), 0), dtype=torch.float, device=device,
            )

    return edge_tensors


def _nx_to_hetero_pyg(graph: nx.Graph,
                      node_feature_attrs: list[str] | None,
                      node_label_attrs: list[str] | None,
                      edge_feature_attrs: list[str] | None,
                      device: Union[str, "torch.device"] | None) -> HeteroData:
    """Convert NetworkX graph to heterogeneous PyTorch Geometric HeteroData object."""
    device = _get_device(device)
    data = HeteroData()

    # Group nodes by type vectorized
    node_types = {}
    for node, node_data in graph.nodes(data=True):
        node_type = node_data.get("node_type", "default")
        if node_type not in node_types:
            node_types[node_type] = []
        node_types[node_type].append(node)

    # Extract node data vectorized
    node_mappings, node_data_store = _extract_nx_node_data(
        graph, node_types, node_feature_attrs, node_label_attrs, device,
    )

    # Assign node data to HeteroData
    for node_type in node_types:
        if f"{node_type}_x" in node_data_store:
            data[node_type].x = node_data_store[f"{node_type}_x"]
        if f"{node_type}_y" in node_data_store:
            data[node_type].y = node_data_store[f"{node_type}_y"]
        if f"{node_type}_pos" in node_data_store:
            data[node_type].pos = node_data_store[f"{node_type}_pos"]

    # Extract edge data vectorized
    edge_data_store = _extract_nx_edge_data(graph, node_mappings, edge_feature_attrs, device)

    # Assign edge data to HeteroData
    for edge_type, edge_data in edge_data_store.items():
        data[edge_type].edge_index = edge_data["edge_index"]
        data[edge_type].edge_attr = edge_data["edge_attr"]

    # Preserve CRS if available
    if "crs" in graph.graph:
        data.crs = graph.graph["crs"]

    # Store metadata for reconstruction
    _store_nx_metadata(data, graph, node_feature_attrs, node_label_attrs, edge_feature_attrs)

    return data


def _collect_nx_node_columns(node_attrs: dict,
                             feature_attrs: list[str] | None,
                             label_attrs: list[str] | None) -> list[str]:
    """Collect all reconstructable attribute names for NetworkX nodes."""
    columns = []

    # Add feature attributes
    if feature_attrs:
        valid_features = [attr for attr in feature_attrs if attr in node_attrs]
        columns.extend(valid_features)

    # Add label attributes
    if label_attrs:
        valid_labels = [attr for attr in label_attrs if attr in node_attrs]
        columns.extend(valid_labels)

    # Add position attribute (stored in data.pos)
    if "pos" in node_attrs:
        columns.append("pos")

    return list(set(columns))


def _collect_nx_edge_columns(edge_attrs: dict,
                             feature_attrs: list[str] | None) -> list[str]:
    """Collect all reconstructable attribute names for NetworkX edges."""
    columns = []

    # Add feature attributes
    if feature_attrs:
        valid_features = [attr for attr in feature_attrs if attr in edge_attrs]
        columns.extend(valid_features)

    return list(set(columns))


def _store_nx_metadata(data: Data | HeteroData,
                       graph: nx.Graph | None,
                       node_feature_attrs: list[str] | None,
                       node_label_attrs: list[str] | None,
                       edge_feature_attrs: list[str] | None) -> None:
    """Store column names from NetworkX graphs that can be reconstructed from tensors."""
    if graph is None:
        return

    # Initialize column storage
    if not hasattr(data, "_node_columns"):
        data._node_columns = {}
    if not hasattr(data, "_edge_columns"):
        data._edge_columns = {}

    # Group nodes by type
    node_types = {}
    for node, node_data in graph.nodes(data=True):
        node_type = node_data.get("node_type", "default")
        if node_type not in node_types:
            node_types[node_type] = []
        node_types[node_type].append(node)

    # Store node attribute names that were used as features/labels
    for node_type, nodes_list in node_types.items():
        if nodes_list:
            sample_attrs = graph.nodes[nodes_list[0]]
            columns = _collect_nx_node_columns(sample_attrs, node_feature_attrs, node_label_attrs)
            if columns:
                data._node_columns[node_type] = columns

    # Store edge attribute names that were used as features
    edge_types = {}
    for src, dst, edge_data in graph.edges(data=True):
        src_type = graph.nodes[src].get("node_type", "default")
        dst_type = graph.nodes[dst].get("node_type", "default")
        relation = edge_data.get("relation", edge_data.get("edge_type", "edge"))
        edge_type = (src_type, relation, dst_type)
        if edge_type not in edge_types:
            edge_types[edge_type] = []
        edge_types[edge_type].append((src, dst, edge_data))

    for edge_type, edges_list in edge_types.items():
        if edges_list:
            sample_edge_data = edges_list[0][2]
            columns = _collect_nx_edge_columns(sample_edge_data, edge_feature_attrs)
            if columns:
                data._edge_columns[edge_type] = columns


def _store_node_gdf_metadata(data: Data | HeteroData,
                            nodes: dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame,
                            node_feature_cols: dict[str, list[str]] | list[str] | None,
                            node_label_cols: dict[str, list[str]] | list[str] | None) -> None:
    """Store node metadata from GeoDataFrames."""
    if isinstance(nodes, dict):
        # Store feature and label column names for each node type
        for node_type, node_gdf in nodes.items():
            reconstructable_cols = []

            # Add feature columns that were used (stored in data.x)
            if isinstance(node_feature_cols, dict) and node_type in node_feature_cols:
                features = node_feature_cols[node_type]
                if features and hasattr(node_gdf, "columns"):
                    valid_features = [col for col in features if col in node_gdf.columns]
                    reconstructable_cols.extend(valid_features)

            # Add label columns that were used (stored in data.y)
            if isinstance(node_label_cols, dict) and node_type in node_label_cols:
                labels = node_label_cols[node_type]
                if labels and hasattr(node_gdf, "columns"):
                    valid_labels = [col for col in labels if col in node_gdf.columns]
                    reconstructable_cols.extend(valid_labels)

            # Add geometry column (stored in data.pos)
            if hasattr(node_gdf, "geometry") and "geometry" in node_gdf.columns:
                reconstructable_cols.append("geometry")

            # Store all reconstructable columns
            if reconstructable_cols:
                if not hasattr(data, "_node_columns"):
                    data._node_columns = {}
                data._node_columns[node_type] = list(set(reconstructable_cols))
    else:
        # Single GeoDataFrame - store for "node" type
        reconstructable_cols = []

        if isinstance(node_feature_cols, list) and hasattr(nodes, "columns"):
            valid_features = [col for col in node_feature_cols if col in nodes.columns]
            reconstructable_cols.extend(valid_features)

        if isinstance(node_label_cols, list) and hasattr(nodes, "columns"):
            valid_labels = [col for col in node_label_cols if col in nodes.columns]
            reconstructable_cols.extend(valid_labels)

        # Add geometry column
        if hasattr(nodes, "geometry") and "geometry" in nodes.columns:
            reconstructable_cols.append("geometry")

        if reconstructable_cols:
            data._node_columns = {"node": list(set(reconstructable_cols))}


def _store_edge_gdf_metadata(data: Data | HeteroData,
                            edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | gpd.GeoDataFrame | None,
                            edge_feature_cols: dict[tuple[str, str, str], list[str]] | list[str] | None) -> None:  # noqa: E501
    """Store edge metadata from GeoDataFrames."""
    if edges is None:
        return

    if isinstance(edges, dict):
        # Store edge feature column names for each edge type
        for edge_type, edge_gdf in edges.items():
            reconstructable_cols = []

            # Add feature columns that were used (stored in edge_attr)
            if isinstance(edge_feature_cols, dict) and edge_type in edge_feature_cols:
                features = edge_feature_cols[edge_type]
                if features and hasattr(edge_gdf, "columns"):
                    valid_features = [col for col in features if col in edge_gdf.columns]
                    reconstructable_cols.extend(valid_features)

            # Add geometry column if present
            if hasattr(edge_gdf, "geometry") and "geometry" in edge_gdf.columns:
                reconstructable_cols.append("geometry")

            if reconstructable_cols:
                if not hasattr(data, "_edge_columns"):
                    data._edge_columns = {}
                data._edge_columns[edge_type] = list(set(reconstructable_cols))
    elif isinstance(edge_feature_cols, list) and hasattr(edges, "columns"):
        # Single GeoDataFrame - store for default edge type
        reconstructable_cols = []

        valid_features = [col for col in edge_feature_cols if col in edges.columns]
        reconstructable_cols.extend(valid_features)

        # Add geometry column
        if hasattr(edges, "geometry") and "geometry" in edges.columns:
            reconstructable_cols.append("geometry")

        if reconstructable_cols:
            data._edge_columns = {("node", "edge", "node"): list(set(reconstructable_cols))}


def _store_gdf_metadata(data: Data | HeteroData,
                       nodes: dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame | None,
                       edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | gpd.GeoDataFrame | None,
                       node_feature_cols: dict[str, list[str]] | list[str] | None,
                       node_label_cols: dict[str, list[str]] | list[str] | None,
                       edge_feature_cols: dict[tuple[str, str, str], list[str]] | list[str] | None) -> None:
    """Store column names from GeoDataFrames that can be reconstructed from tensors."""
    if nodes is not None:
        _store_node_gdf_metadata(data, nodes, node_feature_cols, node_label_cols)

    if edges is not None:
        _store_edge_gdf_metadata(data, edges, edge_feature_cols)


def _store_id_mappings(data: Data | HeteroData,
                       node_id_cols: dict[str, str] | str | None,
                       edge_source_cols: dict[tuple[str, str, str], str] | str | None,
                       edge_target_cols: dict[tuple[str, str, str], str] | str | None) -> None:
    """Store ID and source/target column mappings."""
    if isinstance(node_id_cols, dict):
        data._node_id_cols = node_id_cols
    elif isinstance(node_id_cols, str):
        data._node_id_cols = {"node": node_id_cols}

    if isinstance(edge_source_cols, dict):
        data._edge_source_cols = edge_source_cols
    elif isinstance(edge_source_cols, str):
        data._edge_source_cols = {("node", "edge", "node"): edge_source_cols}

    if isinstance(edge_target_cols, dict):
        data._edge_target_cols = edge_target_cols
    elif isinstance(edge_target_cols, str):
        data._edge_target_cols = {("node", "edge", "node"): edge_target_cols}


def _store_nx_node_metadata(data: Data | HeteroData,
                           graph: nx.Graph,
                           node_feature_cols: dict | list | None,
                           node_label_cols: dict | list | None) -> None:
    """Store node metadata from NetworkX graphs."""
    # Group nodes by type using vectorized operations
    node_types = {}
    for node, node_data in graph.nodes(data=True):
        node_type = node_data.get("node_type", "default")
        if node_type not in node_types:
            node_types[node_type] = []
        node_types[node_type].append(node)

    # Store node attributes
    for node_type, nodes_list in node_types.items():
        if not nodes_list:
            continue

        sample_attrs = graph.nodes[nodes_list[0]]

        # Store feature attributes if they were specified
        if node_feature_cols:
            feature_attrs = (
                node_feature_cols.get(node_type, [])
                if isinstance(node_feature_cols, dict)
                else node_feature_cols if isinstance(node_feature_cols, list) else []
            )
            valid_features = [attr for attr in feature_attrs if attr in sample_attrs]
            if valid_features:
                if not hasattr(data, "_node_feature_columns"):
                    data._node_feature_columns = {}
                data._node_feature_columns[node_type] = valid_features

        # Store label attributes if they were specified
        if node_label_cols:
            label_attrs = (
                node_label_cols.get(node_type, [])
                if isinstance(node_label_cols, dict)
                else node_label_cols if isinstance(node_label_cols, list) else []
            )
            valid_labels = [attr for attr in label_attrs if attr in sample_attrs]
            if valid_labels:
                if not hasattr(data, "_node_label_columns"):
                    data._node_label_columns = {}
                data._node_label_columns[node_type] = valid_labels


def _store_nx_edge_metadata(data: Data | HeteroData,
                           graph: nx.Graph,
                           edge_feature_cols: dict | list | None) -> None:
    """Store edge metadata from NetworkX graphs."""
    if not edge_feature_cols:
        return

    # Store edge attributes
    edge_types = {}
    for src, dst, edge_data in graph.edges(data=True):
        src_type = graph.nodes[src].get("node_type", "default")
        dst_type = graph.nodes[dst].get("node_type", "default")
        relation = edge_data.get("relation", edge_data.get("edge_type", "edge"))
        edge_type = (src_type, relation, dst_type)
        if edge_type not in edge_types:
            edge_types[edge_type] = []
        edge_types[edge_type].append((src, dst, edge_data))

    for edge_type, edges_list in edge_types.items():
        if not edges_list:
            continue

        sample_edge_data = edges_list[0][2]
        feature_attrs = (
            edge_feature_cols.get(edge_type, [])
            if isinstance(edge_feature_cols, dict)
            else edge_feature_cols if isinstance(edge_feature_cols, list) else []
        )
        valid_features = [attr for attr in feature_attrs if attr in sample_edge_data]
        if valid_features:
            if not hasattr(data, "_edge_feature_columns"):
                data._edge_feature_columns = {}
            data._edge_feature_columns[edge_type] = valid_features


def _store_nx_metadata(data: Data | HeteroData,
                      graph: nx.Graph | None,
                      node_feature_cols: dict | list | None,
                      node_label_cols: dict | list | None,
                      edge_feature_cols: dict | list | None) -> None:
    """Store metadata from NetworkX graphs."""
    if graph is None:
        return

    _store_nx_node_metadata(data, graph, node_feature_cols, node_label_cols)
    _store_nx_edge_metadata(data, graph, edge_feature_cols)


def _store_reconstruction_metadata(
    data: Data | HeteroData,
    nodes: dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame | None = None,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | gpd.GeoDataFrame | None = None,
    node_id_cols: dict[str, str] | str | None = None,
    node_feature_cols: dict[str, list[str]] | list[str] | None = None,
    node_label_cols: dict[str, list[str]] | list[str] | None = None,
    edge_source_cols: dict[tuple[str, str, str], str] | str | None = None,
    edge_target_cols: dict[tuple[str, str, str], str] | str | None = None,
    edge_feature_cols: dict[tuple[str, str, str], list[str]] | list[str] | None = None,
    graph: nx.Graph | None = None,
) -> None:
    """Store metadata in Data/HeteroData object for reconstruction purposes."""
    # Initialize metadata storage
    data._node_feature_columns = {}
    data._node_label_columns = {}
    data._edge_feature_columns = {}
    data._node_id_cols = {}
    data._edge_source_cols = {}
    data._edge_target_cols = {}

    # Store feature and label column names based on the inputs
    if node_feature_cols is not None:
        if isinstance(node_feature_cols, dict):
            data._node_feature_columns.update(node_feature_cols)
        elif isinstance(node_feature_cols, list):
            data._node_feature_columns["node"] = node_feature_cols

    if node_label_cols is not None:
        if isinstance(node_label_cols, dict):
            data._node_label_columns.update(node_label_cols)
        elif isinstance(node_label_cols, list):
            data._node_label_columns["node"] = node_label_cols

    if edge_feature_cols is not None:
        if isinstance(edge_feature_cols, dict):
            data._edge_feature_columns.update(edge_feature_cols)
        elif isinstance(edge_feature_cols, list):
            data._edge_feature_columns[("node", "edge", "node")] = edge_feature_cols

    # Store GeoDataFrame metadata
    _store_gdf_metadata(
        data, nodes, edges, node_feature_cols, node_label_cols, edge_feature_cols,
    )

    # Store ID mappings
    _store_id_mappings(data, node_id_cols, edge_source_cols, edge_target_cols)

    # Store NetworkX metadata
    _store_nx_metadata(data, graph, node_feature_cols, node_label_cols, edge_feature_cols)
