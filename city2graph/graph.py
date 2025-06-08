"""
Module for creating heterogeneous graph representations of urban environments.

This module provides comprehensive functionality for converting spatial data (GeoDataFrames)
into PyTorch Geometric graph objects, supporting both homogeneous and heterogeneous graphs.
It handles the complex mapping between geographical coordinates, node/edge features,
and the tensor representations required by graph neural networks.

Key Features:
- Automatic detection of graph structure (homogeneous vs heterogeneous)
- Intelligent column detection for source/target relationships
- Robust type handling and ID mapping
- Preservation of spatial geometry and coordinate reference systems
- Bidirectional conversion between GeoDataFrames and PyTorch Geometric objects
- NetworkX integration for graph analysis workflows

The module supports various spatial data formats and automatically handles:
- Point geometries for nodes (with fallback to centroids)
- LineString geometries for edges (generated from node positions)
- MultiIndex structures for complex edge relationships
- Feature extraction from tabular data
- Label propagation for supervised learning tasks

Main Functions:
    gdf_to_pyg: Convert GeoDataFrames to PyTorch Geometric graphs
    pyg_to_gdf: Convert PyG graphs back to GeoDataFrames
    nx_to_pyg: Convert NetworkX graphs to PyTorch Geometric
    pyg_to_nx: Convert PyG graphs to NetworkX
"""

import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from typing import Any
from shapely.geometry import LineString
from shapely.geometry import Point

from .utils import _validate_gdf
from .utils import _validate_nx

try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.data import HeteroData
    from torch_geometric.utils import to_networkx
    TORCH_AVAILABLE = True

except ImportError:
    TORCH_AVAILABLE = False
    class HeteroData:
        pass
    class Data:
        pass

logger = logging.getLogger(__name__)

__all__ = [
    "gdf_to_pyg",
    "is_torch_available",
    "nx_to_pyg",
    "pyg_to_gdf",
    "pyg_to_nx",
]

# Constants for error messages
TORCH_ERROR_MSG = "PyTorch required. Install with: pip install city2graph[torch]"
DEVICE_ERROR_MSG = "Device must be 'cuda', 'cpu', a torch.device object, or None"
GRAPH_NO_NODES_MSG = "Graph has no nodes"

def is_torch_available() -> bool:
    """Check if PyTorch and PyTorch Geometric are available."""
    return TORCH_AVAILABLE


def _get_device(device: str | torch.device | None = None) -> torch.device:
    """
    Get appropriate torch device.

    Parameters
    ----------
    device : str or torch.device, optional
        Target device specification

    Returns
    -------
    torch.device
        Configured device object

    Raises
    ------
    ImportError
        If PyTorch is not available
    ValueError
        If device specification is invalid
    """
    if not TORCH_AVAILABLE:
        raise ImportError(TORCH_ERROR_MSG)

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    if device in ["cuda", "cpu"]:
        return torch.device(device)
    raise ValueError(DEVICE_ERROR_MSG)

def _detect_edge_columns(
    edge_gdf: gpd.GeoDataFrame,
    id_col: str | None = None,
    source_hints: list[str] | None = None,
    target_hints: list[str] | None = None,
) -> tuple[str | None, str | None]:
    """
    Detect source and target columns in edge GeoDataFrame.

    Automatically identifies which columns contain source and target node IDs
    by checking for common naming patterns or using MultiIndex structure.

    Parameters
    ----------
    edge_gdf : gpd.GeoDataFrame
        GeoDataFrame containing edge data
    id_col : str, optional
        Optional column name hint for ID fields
    source_hints : list[str], optional
        Additional keywords to look for in source column names
    target_hints : list[str], optional
        Additional keywords to look for in target column names

    Returns
    -------
    tuple[str | None, str | None]
        Tuple of (source_column_name, target_column_name) or (None, None) if not found
    """
    # Return early if insufficient data
    if edge_gdf.empty or len(edge_gdf.columns) < 2:
        return None, None

    # Special case: MultiIndex with 2 levels (source, target in index)
    if isinstance(edge_gdf.index, pd.MultiIndex) and edge_gdf.index.nlevels == 2:
        return "source_from_index", "target_from_index"

    # Build keyword lists for column name matching
    source_keywords = ["from", "source", "start", "u"]
    target_keywords = ["to", "target", "end", "v"]

    # Extend with user-provided hints
    if source_hints:
        source_keywords.extend([hint.lower() for hint in source_hints])
    if target_hints:
        target_keywords.extend([hint.lower() for hint in target_hints])
    if id_col:
        source_keywords.append(id_col.lower())
        target_keywords.append(id_col.lower())

    # Find matching columns based on naming patterns
    from_candidates = [col for col in edge_gdf.columns
                      if any(keyword in col.lower() for keyword in source_keywords)]
    to_candidates = [col for col in edge_gdf.columns
                    if any(keyword in col.lower() for keyword in target_keywords)]

    # Return best candidates or fallback to positional detection
    if from_candidates and to_candidates:
        return from_candidates[0], to_candidates[0]

    # Fallback: use first two non-geometry columns
    cols = edge_gdf.columns
    if "geometry" not in cols[:2] and len(cols) >= 2:
        return cols[0], cols[1]
    if "geometry" in cols[:1] and len(cols) >= 3:
        return cols[1], cols[2]

    return None, None

def _create_node_id_mapping(
    node_gdf: gpd.GeoDataFrame, id_col: str | None = None,
) -> tuple[dict[str | int, int], str, list[str | int]]:
    """
    Create mapping from node IDs to sequential integer indices.

    PyTorch Geometric requires nodes to be identified by sequential integers starting from 0.
    This function creates the necessary mapping from original node identifiers to these indices.

    Parameters
    ----------
    node_gdf : gpd.GeoDataFrame
        GeoDataFrame containing node data
    id_col : str, optional
        Column name to use for node IDs (defaults to using the index)

    Returns
    -------
    dict[str | int, int]
        Dictionary mapping original IDs to integer indices
    str
        Name of the ID column used ("index" if using DataFrame index)
    list[str | int]
        List of original IDs in order

    Raises
    ------
    ValueError
        If specified id_col is not found in the GeoDataFrame
    """
    # Check if specified column exists
    if id_col is not None and id_col not in node_gdf.columns:
        error_msg = f"Provided id_col '{id_col}' not found in node GeoDataFrame"
        raise ValueError(error_msg)

    if id_col is None:
        # Use DataFrame index as the node identifier
        original_ids = node_gdf.index.tolist()
        id_mapping = {node_id: i for i, node_id in enumerate(original_ids)}
        return id_mapping, "index", original_ids

    # Use specified column as the node identifier
    original_ids = node_gdf[id_col].tolist()
    id_mapping = {node_id: idx for idx, node_id in enumerate(original_ids)}
    return id_mapping, id_col, original_ids

def _create_node_features(
    node_gdf: gpd.GeoDataFrame,
    feature_cols: list[str] | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Convert node attributes to PyTorch feature tensors.

    Extracts numerical data from specified columns and converts to a tensor suitable
    for graph neural network processing. Handles missing columns gracefully and
    ensures consistent data types.

    Parameters
    ----------
    node_gdf : gpd.GeoDataFrame
        GeoDataFrame containing node data
    feature_cols : list[str], optional
        List of column names to use as features (None creates empty tensor)
    device : str or torch.device, optional
        Target device for tensor creation

    Returns
    -------
    torch.Tensor
        Float tensor of shape (num_nodes, num_features) containing node features
    """
    device = _get_device(device)

    if feature_cols is None:
        # Return empty tensor when no feature columns specified
        return torch.zeros((len(node_gdf), 0), dtype=torch.float, device=device)

    # Find valid columns that exist in the GeoDataFrame
    valid_cols = list(set(feature_cols) & set(node_gdf.columns))
    if valid_cols:
        # Convert to numpy array with consistent float32 type
        features_array = node_gdf[valid_cols].to_numpy().astype(np.float32)
        return torch.from_numpy(features_array).to(device=device, dtype=torch.float)

    # Return empty tensor if no valid columns found
    return torch.zeros((len(node_gdf), 0), dtype=torch.float, device=device)

def _create_edge_features(
    edge_gdf: gpd.GeoDataFrame,
    feature_cols: list[str] | None = None,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """
    Convert edge attributes to PyTorch feature tensors.

    Similar to node features but for edge data. Commonly used for edge weights,
    distances, or other relationship-specific attributes.

    Parameters
    ----------
    edge_gdf : gpd.GeoDataFrame
        GeoDataFrame containing edge data
    feature_cols : list[str], optional
        List of column names to use as features
    device : str or torch.device, optional
        Target device for tensor creation

    Returns
    -------
    torch.Tensor
        Float tensor of shape (num_edges, num_features) containing edge features
    """
    device = _get_device(device)

    if feature_cols is None:
        return torch.empty((edge_gdf.shape[0], 0), dtype=torch.float, device=device)

    valid_cols = list(set(feature_cols) & set(edge_gdf.columns))
    if not valid_cols:
        return torch.empty((edge_gdf.shape[0], 0), dtype=torch.float, device=device)

    features_array = edge_gdf[valid_cols].to_numpy().astype(np.float32)
    return torch.from_numpy(features_array).to(device=device, dtype=torch.float)

def _create_edge_indices(
    edge_gdf: gpd.GeoDataFrame,
    source_mapping: dict[str | int, int],
    target_mapping: dict[str | int, int] | None = None,
    source_col: str | None = None,
    target_col: str | None = None,
) -> list[list[int]]:
    """
    Create edge connectivity matrix from edge data.

    Converts source and target node identifiers to integer indices for PyTorch Geometric.
    This is the core function that establishes graph connectivity by mapping edge
    relationships to the COO (coordinate) format expected by PyG.

    Parameters
    ----------
    edge_gdf : gpd.GeoDataFrame
        GeoDataFrame containing edge data
    source_mapping : dict[str | int, int]
        Dictionary mapping source node IDs to integer indices
    target_mapping : dict[str | int, int], optional
        Dictionary mapping target node IDs to integer indices
    source_col : str, optional
        Column name containing source node IDs
    target_col : str, optional
        Column name containing target node IDs

    Returns
    -------
    list[list[int]]
        List of [source_index, target_index] pairs for all valid edges
        Empty list if no valid edges found
    """
    # Use source mapping for targets if no specific target mapping provided
    target_mapping = target_mapping or source_mapping

    # Handle MultiIndex case - extract source/target from index levels
    if isinstance(edge_gdf.index, pd.MultiIndex) and edge_gdf.index.nlevels == 2:
        source_ids, target_ids = _handle_multiindex_edges(edge_gdf)
    else:
        # For regular DataFrames, detect or use provided column specifications
        extracted_ids = _detect_and_extract_edge_columns(edge_gdf, source_col, target_col)
        if extracted_ids[0] is None or extracted_ids[1] is None:
            return []
        source_ids, target_ids = extracted_ids

    # Attempt type conversion if needed
    source_ids = _attempt_type_conversion(source_ids, source_mapping, "Source")
    target_ids = _attempt_type_conversion(target_ids, target_mapping, "Target")

    # Find edges with valid source and target nodes
    valid_src_mask = source_ids.isin(source_mapping.keys())
    valid_dst_mask = target_ids.isin(target_mapping.keys())
    valid_edges_mask = valid_src_mask & valid_dst_mask

    # Return empty list if no valid edges found
    if not valid_edges_mask.any():
        return []

    # Process valid edges using vectorized operations
    valid_sources = source_ids[valid_edges_mask]
    valid_targets = target_ids[valid_edges_mask]

    # Map original node IDs to integer indices
    from_indices = valid_sources.map(source_mapping).to_numpy()
    to_indices = valid_targets.map(target_mapping).to_numpy()

    # Return as list of [source, target] pairs
    return np.column_stack([from_indices, to_indices]).tolist()

def _handle_multiindex_edges(edge_gdf: gpd.GeoDataFrame) -> tuple[pd.Series, pd.Series]:
    """Extract source and target IDs from MultiIndex DataFrame."""
    return (edge_gdf.index.get_level_values(0),  # First level = source
            edge_gdf.index.get_level_values(1))   # Second level = target


def _detect_and_extract_edge_columns(
    edge_gdf: gpd.GeoDataFrame,
    source_col: str | None,
    target_col: str | None,
) -> tuple[pd.Series | None, pd.Series | None]:
    """Detect and extract source/target columns from regular DataFrame."""
    if source_col is None or target_col is None:
        detected_source, detected_target = _detect_edge_columns(edge_gdf)
        source_col = source_col or detected_source
        target_col = target_col or detected_target

    # Validate that required columns are available
    if source_col is None or target_col is None:
        return None, None

    if source_col not in edge_gdf.columns or target_col not in edge_gdf.columns:
        return None, None

    return edge_gdf[source_col], edge_gdf[target_col]


def _attempt_type_conversion(
    ids: pd.Series | pd.Index,
    mapping: dict[str | int, int],
    id_type: str,
) -> pd.Series:
    """Attempt to convert IDs to match mapping key types."""
    if len(ids) == 0 or not mapping:
        return pd.Series(ids) if isinstance(ids, pd.Index) else ids

    # Convert Index to Series if needed
    if isinstance(ids, pd.Index):
        ids = pd.Series(ids)

    # Check if any IDs match the mapping keys
    if ids.isin(mapping.keys()).sum() > 0:
        return ids

    # Try type conversion
    sample_id = ids.iloc[0] if len(ids) > 0 else None
    sample_key = next(iter(mapping.keys())) if mapping else None

    if (sample_id is not None and sample_key is not None and
        not isinstance(sample_id, type(sample_key))):
        try:
            if isinstance(sample_key, str):
                return ids.astype(str)
            if isinstance(sample_key, int):
                return ids.astype(int)
        except Exception as e:
            logger.debug("%s ID type conversion failed: %s", id_type, e)

    return ids

def _create_node_positions(
    node_gdf: gpd.GeoDataFrame, device: str | torch.device | None = None,
) -> torch.Tensor | None:
    """
    Extract spatial coordinates from node geometries.

    Converts geometric representations to coordinate tensors suitable for
    spatial graph neural networks. Handles various geometry types and
    provides consistent coordinate extraction.

    Args:
        node_gdf: GeoDataFrame with geometry column containing spatial data
        device: Target device for tensor creation

    Returns
    -------
        Float tensor of shape (num_nodes, 2) containing [x, y] coordinates
        None if no geometry column found

    Notes
    -----
        - Point geometries: Uses direct x, y coordinates
        - Other geometries: Uses centroid coordinates
        - Mixed geometries: Handles each type appropriately
        - Coordinates are in the original CRS of the GeoDataFrame
    """
    device = _get_device(device)

    if "geometry" not in node_gdf.columns:
        return None

    geom_series = node_gdf.geometry
    is_point_mask = geom_series.geom_type == "Point"

    if is_point_mask.all():
        # All points - direct extraction
        pos_data = np.column_stack([geom_series.x.to_numpy(), geom_series.y.to_numpy()])
    else:
        # Mixed geometries - use centroids for non-points
        pos_data = np.zeros((len(geom_series), 2))

        if is_point_mask.any():
            point_coords = np.column_stack([
                geom_series[is_point_mask].x.to_numpy(),
                geom_series[is_point_mask].y.to_numpy(),
            ])
            pos_data[is_point_mask] = point_coords

        if (~is_point_mask).any():
            centroids = geom_series[~is_point_mask].centroid
            centroid_coords = np.column_stack([
                centroids.x.to_numpy(),
                centroids.y.to_numpy(),
            ])
            pos_data[~is_point_mask] = centroid_coords

    return torch.tensor(pos_data, dtype=torch.float, device=device)

def _create_linestring_geometries(
    edge_index_array: np.ndarray, src_pos: np.ndarray, dst_pos: np.ndarray,
) -> list[LineString | None]:
    """
    Generate LineString geometries from node positions and edge connectivity.

    Creates geometric representations of edges by connecting source and target
    node coordinates. Useful for visualization and spatial analysis of networks.

    Args:
        edge_index_array: Array of shape (2, num_edges) with source/target indices
        src_pos: Array of source node coordinates
        dst_pos: Array of target node coordinates

    Returns
    -------
        List of LineString objects connecting source to target nodes
        None entries for invalid/out-of-bounds edges

    Notes
    -----
        - Performs bounds checking to avoid index errors
        - Only uses first 2 dimensions of position data (x, y)
        - Returns None for edges with invalid node indices
    """
    if edge_index_array.size == 0:
        return []

    src_indices = edge_index_array[0]
    dst_indices = edge_index_array[1]

    # Vectorized bounds checking
    valid_src_mask = src_indices < len(src_pos)
    valid_dst_mask = dst_indices < len(dst_pos)
    valid_mask = valid_src_mask & valid_dst_mask

    if not valid_mask.any():
        return [None] * len(src_indices)

    # Get valid indices and coordinates
    valid_src_indices = src_indices[valid_mask]
    valid_dst_indices = dst_indices[valid_mask]
    src_coords = src_pos[valid_src_indices][:, :2]
    dst_coords = dst_pos[valid_dst_indices][:, :2]

    # Create LineStrings using vectorized coordinate pairing
    coord_pairs = np.stack([src_coords, dst_coords], axis=1)
    
    # Vectorized LineString creation - use map for better performance
    valid_geometries = list(map(LineString, coord_pairs))

    # Vectorized assignment using fancy indexing
    geometries = np.full(len(src_indices), None, dtype=object)
    geometries[valid_mask] = valid_geometries

    return geometries.tolist()

def _build_homogeneous_graph(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame | None = None,
    node_id_col: str | None = None,
    node_feature_cols: list[str] | None = None,
    node_label_cols: list[str] | None = None,
    edge_source_col: str | None = None,
    edge_target_col: str | None = None,
    edge_feature_cols: list[str] | None = None,
    device: str | torch.device | None = None,
) -> Data:
    """
    Construct a homogeneous PyTorch Geometric Data object.

    Creates a single-type graph where all nodes and edges are treated uniformly.
    This is the most common graph format for simple network analysis tasks.

    Processing Pipeline:
    1. Create node ID mapping (original IDs → integer indices)
    2. Extract node features and positions from geometry
    3. Process node labels if available
    4. Create edge connectivity matrix
    5. Extract edge features
    6. Package everything into PyG Data object
    7. Store metadata for reconstruction

    Args:
        nodes_gdf: GeoDataFrame containing node data
        edges_gdf: GeoDataFrame containing edge data (optional)
        node_id_col: Column for node identification
        node_feature_cols: Columns to use as node features
        node_label_cols: Columns to use as node labels
        edge_source_col: Column containing source node IDs
        edge_target_col: Column containing target node IDs
        edge_feature_cols: Columns to use as edge features
        device: Target device for tensor creation

    Returns
    -------
        PyTorch Geometric Data object with all graph components

    Notes
    -----
        - Preserves original index names and values for reconstruction
        - Stores metadata for bidirectional conversion
        - Handles missing edges gracefully (creates empty edge tensors)
        - Maintains CRS information if available
    """
    device = _get_device(device)

    # Node processing
    id_mapping, id_col_name, original_ids = _create_node_id_mapping(nodes_gdf, node_id_col)

    x = _create_node_features(nodes_gdf, node_feature_cols, device)
    pos = _create_node_positions(nodes_gdf, device)

    # Handle labels
    y = None
    if node_label_cols:
        y = _create_node_features(nodes_gdf, node_label_cols, device)
    elif "y" in nodes_gdf.columns:
        y = torch.tensor(nodes_gdf["y"].to_numpy(), dtype=torch.float, device=device)

    # Edge processing
    edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    edge_attr = torch.empty((0, 0), dtype=torch.float, device=device)

    if edges_gdf is not None and not edges_gdf.empty:
        edge_pairs = _create_edge_indices(
            edges_gdf, id_mapping, id_mapping, edge_source_col, edge_target_col,
        )
        if edge_pairs:
            edge_index = torch.tensor(
                np.array(edge_pairs).T, dtype=torch.long, device=device,
            )
        edge_attr = _create_edge_features(edges_gdf, edge_feature_cols, device)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)

    # Store metadata - use unified _node_mappings structure
    data._node_mappings = {
        "default": {
            "mapping": id_mapping,
            "id_col": id_col_name,
            "original_ids": original_ids,
        }
    }
    data._node_feature_cols = node_feature_cols or []
    data._node_label_cols = node_label_cols or []
    data._edge_feature_cols = edge_feature_cols or []

    # Store index names and values for preservation
    data._node_index_names = nodes_gdf.index.names if hasattr(nodes_gdf.index, "names") else None
    if edges_gdf is not None and hasattr(edges_gdf.index, "names"):
        data._edge_index_names = edges_gdf.index.names
        # Store original edge index values for reconstruction
        if isinstance(edges_gdf.index, pd.MultiIndex):
            data._edge_index_values = [edges_gdf.index.get_level_values(i).tolist()
                                     for i in range(edges_gdf.index.nlevels)]
        else:
            data._edge_index_values = edges_gdf.index.tolist()
    else:
        data._edge_index_names = None
        data._edge_index_values = None

    # Set CRS
    if hasattr(nodes_gdf, "crs") and nodes_gdf.crs:
        data.crs = nodes_gdf.crs

    return data

def _build_heterogeneous_graph(
    nodes_dict: dict[str, gpd.GeoDataFrame],
    edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    node_id_cols: dict[str, str] | None = None,
    node_feature_cols: dict[str, list[str]] | None = None,
    node_label_cols: dict[str, list[str]] | None = None,
    edge_source_cols: dict[tuple[str, str, str], str] | None = None,
    edge_target_cols: dict[tuple[str, str, str], str] | None = None,
    edge_feature_cols: dict[tuple[str, str, str], list[str]] | None = None,
    device: str | torch.device | None = None,
) -> HeteroData:
    """
    Construct a heterogeneous PyTorch Geometric HeteroData object.

    Creates a multi-type graph supporting different node and edge types.
    Essential for complex urban networks with diverse entity types (buildings,
    roads, points of interest, etc.) and relationship types.

    Graph Structure:
    - Node types: Different categories of entities (e.g., 'building', 'road', 'poi')
    - Edge types: Relationships as (source_type, relation, target_type) tuples
    - Type-specific features: Each node/edge type can have different attributes

    Processing Pipeline:
    1. Process each node type separately (features, positions, labels)
    2. Create type-specific ID mappings
    3. Process each edge type with appropriate source/target mappings
    4. Store comprehensive metadata for each type
    5. Validate CRS consistency across all GeoDataFrames

    Args:
        nodes_dict: Dictionary mapping node type names to GeoDataFrames
        edges_dict: Dictionary mapping edge type tuples to GeoDataFrames
        node_id_cols: Per-type node ID column specifications
        node_feature_cols: Per-type node feature column specifications
        node_label_cols: Per-type node label column specifications
        edge_source_cols: Per-type edge source column specifications
        edge_target_cols: Per-type edge target column specifications
        edge_feature_cols: Per-type edge feature column specifications
        device: Target device for tensor creation

    Returns
    -------
        PyTorch Geometric HeteroData object with typed graph components

    Notes
    -----
        - Edge types must reference existing node types
        - Handles type-specific feature dimensions automatically
        - Preserves per-type metadata for reconstruction
        - Validates CRS consistency across all data
    """
    device = _get_device(device)
    data = HeteroData()

    # Default empty dicts
    edges_dict = edges_dict or {}
    node_id_cols = node_id_cols or {}
    node_feature_cols = node_feature_cols or {}
    node_label_cols = node_label_cols or {}
    edge_source_cols = edge_source_cols or {}
    edge_target_cols = edge_target_cols or {}
    edge_feature_cols = edge_feature_cols or {}

    # Store node mappings
    node_mappings = {}

    # Process nodes
    for node_type, node_gdf in nodes_dict.items():
        id_col = node_id_cols.get(node_type)
        id_mapping, id_col_name, original_ids = _create_node_id_mapping(node_gdf, id_col)
        node_mappings[node_type] = id_mapping

        # Features
        feature_cols = node_feature_cols.get(node_type)
        data[node_type].x = _create_node_features(node_gdf, feature_cols, device)

        # Positions
        data[node_type].pos = _create_node_positions(node_gdf, device)

        # Labels
        label_cols = node_label_cols.get(node_type)
        if label_cols:
            data[node_type].y = _create_node_features(node_gdf, label_cols, device)
        elif "y" in node_gdf.columns:
            data[node_type].y = torch.tensor(
                node_gdf["y"].to_numpy(), dtype=torch.float, device=device,
            )

    # Process edges using vectorized operations where possible
    for edge_type, edge_gdf in edges_dict.items():
        if not isinstance(edge_type, tuple) or len(edge_type) != 3:
            continue

        src_type, rel_type, dst_type = edge_type

        if src_type not in node_mappings or dst_type not in node_mappings:
            continue

        src_mapping = node_mappings[src_type]
        dst_mapping = node_mappings[dst_type]
        source_col = edge_source_cols.get(edge_type)
        target_col = edge_target_cols.get(edge_type)

        if edge_gdf is not None and not edge_gdf.empty:
            edge_pairs = _create_edge_indices(
                edge_gdf, src_mapping, dst_mapping, source_col, target_col,
            )
            edge_index = (torch.tensor(np.array(edge_pairs).T, dtype=torch.long, device=device)
                         if edge_pairs else torch.zeros((2, 0), dtype=torch.long, device=device))
            data[edge_type].edge_index = edge_index

            feature_cols = edge_feature_cols.get(edge_type)
            data[edge_type].edge_attr = _create_edge_features(edge_gdf, feature_cols, device)
        else:
            data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            data[edge_type].edge_attr = torch.empty((0, 0), dtype=torch.float, device=device)

    # Store metadata
    data._node_mappings = node_mappings
    data._node_feature_cols = node_feature_cols
    data._node_label_cols = node_label_cols
    data._edge_feature_cols = edge_feature_cols

    # Store index names for preservation
    data._node_index_names = {}
    for node_type, node_gdf in nodes_dict.items():
        if hasattr(node_gdf.index, "names"):
            data._node_index_names[node_type] = node_gdf.index.names

    data._edge_index_names = {}
    data._edge_index_values = {}
    for edge_type, edge_gdf in edges_dict.items():
        if edge_gdf is not None and hasattr(edge_gdf.index, "names"):
            data._edge_index_names[edge_type] = edge_gdf.index.names
            # Store original edge index values for reconstruction
            if isinstance(edge_gdf.index, pd.MultiIndex):
                data._edge_index_values[edge_type] = [edge_gdf.index.get_level_values(i).tolist()
                                                    for i in range(edge_gdf.index.nlevels)]
            else:
                data._edge_index_values[edge_type] = edge_gdf.index.tolist()

    # Set CRS
    crs_values = [gdf.crs for gdf in nodes_dict.values() if hasattr(gdf, "crs") and gdf.crs]
    if crs_values and all(crs == crs_values[0] for crs in crs_values):
        data.crs = crs_values[0]

    return data

def _extract_tensor_data(
    tensor: torch.Tensor | None, column_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract data from tensor with proper column names."""
    if tensor is None or tensor.numel() == 0:
        return {}

    features_array = tensor.detach().cpu().numpy()

    if column_names is None:
        num_features = features_array.shape[1] if len(features_array.shape) > 1 else 1
        column_names = [f"feature_{i}" for i in range(num_features)]

    if len(features_array.shape) == 1:
        return {column_names[0]: features_array}

    num_cols = min(len(column_names), features_array.shape[1])
    return {column_names[i]: features_array[:, i] for i in range(num_cols)}

def _get_node_data_info(
    data: Data | HeteroData, node_type: str, is_hetero: bool,
) -> tuple[Data | Any, int]:
    """Get node data and metadata."""
    node_data = data[node_type] if is_hetero else data

    # Determine number of nodes
    num_nodes = 0
    if hasattr(node_data, "x") and node_data.x is not None:
        num_nodes = node_data.x.size(0)
    elif hasattr(node_data, "pos") and node_data.pos is not None:
        num_nodes = node_data.pos.size(0)

    return node_data, num_nodes

def _get_node_ids(
    data: Data | HeteroData, node_type: str, num_nodes: int, is_hetero: bool,
) -> dict[str, list[str | int]]:
    """Get node IDs from unified _node_mappings structure."""
    gdf_data = {}

    # Use unified _node_mappings structure for both homogeneous and heterogeneous
    if hasattr(data, "_node_mappings"):
        if is_hetero and node_type in data._node_mappings:
            # For heterogeneous graphs, use the specific node type mapping
            mapping = data._node_mappings[node_type]
            if isinstance(mapping, dict) and "mapping" in mapping:
                # New unified structure with metadata
                reverse_mapping = {idx: node_id for node_id, idx in mapping["mapping"].items()}
                original_ids = [reverse_mapping.get(i, i) for i in range(num_nodes)]
                id_col = mapping.get("id_col", "node_id")
                if id_col != "index":
                    gdf_data[id_col] = original_ids
            else:
                # Legacy direct mapping
                reverse_mapping = {idx: node_id for node_id, idx in mapping.items()}
                original_ids = [reverse_mapping.get(i, i) for i in range(num_nodes)]
                gdf_data["node_id"] = original_ids
        elif not is_hetero and "default" in data._node_mappings:
            # For homogeneous graphs, use the "default" mapping
            mapping_info = data._node_mappings["default"]
            mapping = mapping_info.get("mapping", {})
            reverse_mapping = {idx: node_id for node_id, idx in mapping.items()}
            original_ids = [reverse_mapping.get(i, i) for i in range(num_nodes)]
            id_col = mapping_info.get("id_col", "node_id")
            if id_col != "index":
                gdf_data[id_col] = original_ids

    return gdf_data

def _extract_features_and_labels(
    data: Data | HeteroData, node_data: Data | Any, node_type: str, is_hetero: bool,
) -> dict[str, np.ndarray]:
    """Extract features and labels from node data."""
    gdf_data = {}

    # Extract features
    if hasattr(node_data, "x") and node_data.x is not None:
        feature_cols = getattr(data, "_node_feature_cols", {})
        cols = feature_cols.get(node_type, None) if is_hetero else feature_cols
        features_dict = _extract_tensor_data(node_data.x, cols)
        gdf_data.update(features_dict)

    # Extract labels
    if hasattr(node_data, "y") and node_data.y is not None:
        label_cols = getattr(data, "_node_label_cols", {})
        cols = label_cols.get(node_type, None) if is_hetero else label_cols
        labels_dict = _extract_tensor_data(node_data.y, cols)
        gdf_data.update(labels_dict)

    return gdf_data

def _reconstruct_node_gdf(
    data: Data | HeteroData, node_type: str = "node", is_hetero: bool = False,
) -> gpd.GeoDataFrame:
    """
    Reconstruct node GeoDataFrame from PyTorch Geometric data.

    Reverses the graph construction process to recover the original spatial
    data format. Essential for analysis, visualization, and interoperability
    with existing GIS workflows.

    Reconstruction Process:
    1. Extract node count and data structures
    2. Recover original node IDs from stored mappings
    3. Convert feature tensors back to DataFrame columns
    4. Recreate geometry from position tensors
    5. Restore original index structure and names
    6. Apply stored CRS information

    Args:
        data: PyTorch Geometric data object
        node_type: Type of nodes to reconstruct (for heterogeneous graphs)
        is_hetero: Whether this is a heterogeneous graph

    Returns
    -------
        GeoDataFrame with reconstructed node data and spatial information

    Notes
    -----
        - Preserves original column names where possible
        - Generates fallback names for unnamed features
        - Recreates Point geometries from coordinate tensors
        - Maintains index structure from original data
    """
    node_data, num_nodes = _get_node_data_info(data, node_type, is_hetero)

    # Get node IDs
    gdf_data = _get_node_ids(data, node_type, num_nodes, is_hetero)

    # Extract features and labels
    features_labels = _extract_features_and_labels(data, node_data, node_type, is_hetero)
    gdf_data.update(features_labels)

    # Create geometry
    geometry = None
    if hasattr(node_data, "pos") and node_data.pos is not None:
        pos_array = node_data.pos.detach().cpu().numpy()
        if len(pos_array.shape) == 2 and pos_array.shape[1] >= 2:
            geometry = gpd.points_from_xy(pos_array[:, 0], pos_array[:, 1])

    # Create GeoDataFrame
    if not gdf_data and num_nodes > 0:
        gdf_data = {"node_id": range(num_nodes)}

    index_values = None
    index_names = None

    # Determine index values and names
    if (hasattr(data, "_node_mappings") and 
        ((not is_hetero and "default" in data._node_mappings) or 
         (is_hetero and node_type in data._node_mappings))):
        
        mapping_key = "default" if not is_hetero else node_type
        mapping_info = data._node_mappings[mapping_key]
        
        if isinstance(mapping_info, dict) and "id_col" in mapping_info:
            if mapping_info.get("id_col") == "index":
                index_values = mapping_info.get("original_ids", list(range(num_nodes)))[:num_nodes]
        
        # Get stored index names
        if hasattr(data, "_node_index_names") and data._node_index_names:
            if is_hetero and node_type in data._node_index_names:
                index_names = data._node_index_names[node_type]
            elif not is_hetero:
                index_names = data._node_index_names

    gdf = gpd.GeoDataFrame(gdf_data, geometry=geometry, index=index_values)

    # Set index names if available
    if (index_names and hasattr(gdf.index, "names") and
        isinstance(index_names, list) and len(index_names) > 0):
        # For MultiIndex or named single index
        if len(index_names) == 1 and index_names[0] is not None:
            gdf.index.name = index_names[0]
        elif len(index_names) > 1:
            gdf.index.names = index_names

    # Set CRS
    if hasattr(data, "crs") and data.crs:
        gdf.crs = data.crs

    return gdf

def _reconstruct_edge_index(
    data: Data | HeteroData,
    edge_type: str | tuple[str, str, str],
    is_hetero: bool,
    edge_data_dict: dict[str, list | np.ndarray],
) -> pd.Index | pd.MultiIndex | None:
    """Reconstruct edge index from stored values."""
    if not hasattr(data, "_edge_index_values"):
        return None

    stored_values = None
    if is_hetero and edge_type in data._edge_index_values:
        stored_values = data._edge_index_values[edge_type]
    elif not is_hetero and data._edge_index_values:
        stored_values = data._edge_index_values

    if not stored_values:
        return None

    if edge_data_dict:
        num_rows = len(next(iter(edge_data_dict.values())))
    elif isinstance(stored_values, list) and stored_values:
        num_rows = len(stored_values[0])
    else:
        num_rows = 0

    if isinstance(stored_values, list) and len(stored_values) > 0:
        if isinstance(stored_values[0], list):
            # MultiIndex case: stored_values is a list of lists for each level
            if len(stored_values) > 1:
                arrays = [stored_values[i][:num_rows] for i in range(len(stored_values))]
                return pd.MultiIndex.from_arrays(arrays)
            # Single level but stored as list of lists
            return stored_values[0][:num_rows]
        # Regular Index case
        return stored_values[:num_rows]

    return None

def _reconstruct_edge_gdf(
    data: Data | HeteroData, edge_type: str = "edge", is_hetero: bool = False,
) -> gpd.GeoDataFrame:
    """Reconstruct edge GeoDataFrame from PyTorch Geometric data."""
    edge_data = data[edge_type] if is_hetero else data
    edge_data_dict = {}

    # Extract edge features
    if hasattr(edge_data, "edge_attr") and edge_data.edge_attr is not None:
        feature_cols = getattr(data, "_edge_feature_cols", {})
        cols = feature_cols.get(edge_type, None) if is_hetero else feature_cols
        features_dict = _extract_tensor_data(edge_data.edge_attr, cols)
        edge_data_dict.update(features_dict)

    # Create geometries from edge indices and node positions
    geometry = None
    if hasattr(edge_data, "edge_index") and edge_data.edge_index is not None:
        edge_index_array = edge_data.edge_index.detach().cpu().numpy()
        if edge_index_array.shape[0] == 2:
            src_pos = dst_pos = None

            if is_hetero and isinstance(edge_type, tuple) and len(edge_type) == 3:
                src_type, _, dst_type = edge_type
                if hasattr(data[src_type], "pos") and data[src_type].pos is not None:
                    src_pos = data[src_type].pos.detach().cpu().numpy()
                if hasattr(data[dst_type], "pos") and data[dst_type].pos is not None:
                    dst_pos = data[dst_type].pos.detach().cpu().numpy()
            elif hasattr(data, "pos") and data.pos is not None:
                pos = data.pos.detach().cpu().numpy()
                src_pos = dst_pos = pos

            if src_pos is not None and dst_pos is not None:
                geometries = _create_linestring_geometries(edge_index_array, src_pos, dst_pos)
                geometry = gpd.array.from_shapely(geometries)

    # Reconstruct index from stored values
    index_values = _reconstruct_edge_index(data, edge_type, is_hetero, edge_data_dict)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(edge_data_dict, geometry=geometry, index=index_values)

    # Set index names if available
    if hasattr(data, "_edge_index_names"):
        index_names = None
        if is_hetero and edge_type in data._edge_index_names:
            index_names = data._edge_index_names[edge_type]
        elif not is_hetero and data._edge_index_names:
            index_names = data._edge_index_names

        if index_names and hasattr(gdf.index, "names"):
            if isinstance(index_names, list) and len(index_names) > 0:
                # For MultiIndex or named single index
                if len(index_names) == 1 and index_names[0] is not None:
                    gdf.index.name = index_names[0]
                elif len(index_names) > 1:
                    # Only set multiple names if we have a MultiIndex
                    if isinstance(gdf.index, pd.MultiIndex):
                        gdf.index.names = index_names
                    else:
                        # For regular Index with multiple stored names, just use the first one
                        gdf.index.name = index_names[0]

    # Set CRS
    if hasattr(data, "crs") and data.crs:
        gdf.crs = data.crs

    return gdf

# Core API functions

def gdf_to_pyg(
    nodes: dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | gpd.GeoDataFrame | None = None,
    node_id_cols: dict[str, str] | str | None = None,
    node_feature_cols: dict[str, list[str]] | list[str] | None = None,
    node_label_cols: dict[str, list[str]] | list[str] | None = None,
    edge_source_cols: dict[tuple[str, str, str], str] | str | None = None,
    edge_target_cols: dict[tuple[str, str, str], str] | str | None = None,
    edge_feature_cols: dict[tuple[str, str, str], list[str]] | list[str] | None = None,
    device: torch.device | str | None = None,
) -> Data | HeteroData:
    """
    Convert GeoDataFrames to PyTorch Geometric graph objects.

    Main entry point for creating graph neural network representations from
    spatial data. Automatically detects whether to create homogeneous or
    heterogeneous graphs based on input structure.

    Graph Type Detection:
    - Homogeneous: Single node/edge type or simple uniform data
    - Heterogeneous: Multiple types or complex type specifications

    Features:
    - Automatic column detection for source/target relationships
    - Intelligent type conversion and validation
    - Preservation of spatial and tabular metadata
    - Support for various input formats and specifications

    Input Formats:
    1. Single GeoDataFrames: nodes_gdf, edges_gdf → Homogeneous graph
    2. Type dictionaries: {type: gdf} → Heterogeneous graph
    3. Mixed specifications: Automatic format detection and conversion

    Parameters
    ----------
    nodes : dict or gpd.GeoDataFrame
        Node data as single GDF or type-keyed dictionary
    edges : dict or gpd.GeoDataFrame, optional
        Edge data as single GDF, type-keyed dictionary, or None
    node_id_cols : dict or str, optional
        Node ID column specifications (per-type or global)
    node_feature_cols : dict or list[str], optional
        Node feature column specifications
    node_label_cols : dict or list[str], optional
        Node label column specifications
    edge_source_cols : dict or str, optional
        Edge source column specifications
    edge_target_cols : dict or str, optional
        Edge target column specifications
    edge_feature_cols : dict or list[str], optional
        Edge feature column specifications
    device : torch.device or str, optional
        Target device for tensor creation

    Returns
    -------
    Data or HeteroData
        Data object for homogeneous graphs
        HeteroData object for heterogeneous graphs

    Raises
    ------
    ImportError
        If PyTorch/PyG not available
    ValueError
        If input data validation fails

    Examples
    --------
    Simple homogeneous graph:
    >>> data = gdf_to_pyg(nodes_gdf, edges_gdf)

    Heterogeneous urban network:
    >>> nodes_dict = {'building': buildings_gdf, 'road': roads_gdf}
    >>> edges_dict = {('building', 'connects', 'road'): connections_gdf}
    >>> data = gdf_to_pyg(nodes_dict, edges_dict)
    """
    if not TORCH_AVAILABLE:
        raise ImportError(TORCH_ERROR_MSG)

    # Validate inputs
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

    # Determine if heterogeneous
    is_hetero = ((isinstance(nodes, dict) and len(nodes) > 1) or
                (isinstance(edges, dict) and len(edges) > 1))

    if not is_hetero and isinstance(edges, dict) and edges:
        # Vectorized heterogeneous check for edge types
        edge_type_tuples = [edge_type for edge_type in edges
                           if isinstance(edge_type, tuple) and len(edge_type) == 3]
        if edge_type_tuples:
            is_hetero = any(src_type != dst_type or relation != "edge"
                           for src_type, relation, dst_type in edge_type_tuples)

    if is_hetero:
        # Convert single GDFs to dicts if needed
        if isinstance(nodes, gpd.GeoDataFrame):
            if "type" not in nodes.columns:
                nodes = {"default": nodes}
            else:
                # Vectorized node type grouping
                unique_types = nodes["type"].unique()
                nodes = {node_type: nodes[nodes["type"] == node_type].copy()
                        for node_type in unique_types}

        if isinstance(edges, gpd.GeoDataFrame):
            if "edge_type" not in edges.columns:
                edges = {("default", "edge", "default"): edges}
            else:
                # Vectorized edge type grouping
                unique_edge_types = edges["edge_type"].unique()
                edges_dict = {}
                for edge_type_str in unique_edge_types:
                    parts = str(edge_type_str).split("_", 2)
                    edge_key = tuple(parts) if len(parts) == 3 else ("default", "edge", "default")
                    edges_dict[edge_key] = edges[edges["edge_type"] == edge_type_str].copy()
                edges = edges_dict

        return _build_heterogeneous_graph(
            nodes, edges, node_id_cols, node_feature_cols, node_label_cols,
            edge_source_cols, edge_target_cols, edge_feature_cols, device,
        )

    # Homogeneous case
    nodes_gdf = next(iter(nodes.values())) if isinstance(nodes, dict) else nodes
    edges_gdf = next(iter(edges.values())) if isinstance(edges, dict) and edges else edges

    # Extract single values for homogeneous case
    node_id_col = next(iter(node_id_cols.values())) if isinstance(node_id_cols, dict) else node_id_cols
    node_feature_cols_list = (
        next(iter(node_feature_cols.values())) if isinstance(node_feature_cols, dict)
        else node_feature_cols
    )
    node_label_cols_list = (
        next(iter(node_label_cols.values())) if isinstance(node_label_cols, dict)
        else node_label_cols
    )
    edge_source_col = (
        next(iter(edge_source_cols.values())) if isinstance(edge_source_cols, dict)
        else edge_source_cols
    )
    edge_target_col = (
        next(iter(edge_target_cols.values())) if isinstance(edge_target_cols, dict)
        else edge_target_cols
    )
    edge_feature_cols_list = (
        next(iter(edge_feature_cols.values())) if isinstance(edge_feature_cols, dict)
        else edge_feature_cols
    )

    data = _build_homogeneous_graph(
        nodes_gdf, edges_gdf, node_id_col, node_feature_cols_list,
        node_label_cols_list, edge_source_col, edge_target_col,
        edge_feature_cols_list, device,
    )

    # Validate the created PyG object
    _validate_pyg(data)
    return data

def pyg_to_gdf(
    data: Data | HeteroData,
) -> (tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]] |
      tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]):
    """
    Convert PyTorch Geometric Data or HeteroData to GeoDataFrames.

    Parameters
    ----------
    data : Data or HeteroData
        PyTorch Geometric graph object to convert.

    Returns
    -------
    tuple
        For HeteroData: (nodes_dict, edges_dict) where nodes_dict maps node types to
        GeoDataFrames and edges_dict maps edge type tuples to GeoDataFrames.
        For Data: (nodes_gdf, edges_gdf) as single GeoDataFrames.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(TORCH_ERROR_MSG)

    # Validate the input PyG object
    _validate_pyg(data)

    # Check if heterogeneous
    is_hetero = hasattr(data, "node_types") and hasattr(data, "edge_types")

    if is_hetero:
        # Handle HeteroData
        nodes_dict = {node_type: _reconstruct_node_gdf(data, node_type, is_hetero=True)
                     for node_type in data.node_types}
        edges_dict = {edge_type: _reconstruct_edge_gdf(data, edge_type, is_hetero=True)
                     for edge_type in data.edge_types}
        return nodes_dict, edges_dict

    # Handle homogeneous Data
    nodes_gdf = _reconstruct_node_gdf(data, "node", is_hetero=False)

    edges_gdf = None
    if (hasattr(data, "edge_index") and data.edge_index is not None and
        data.edge_index.numel() > 0):
        edges_gdf = _reconstruct_edge_gdf(data, "edge", is_hetero=False)

    return nodes_gdf, edges_gdf

def _add_node_geometry(graph: nx.Graph, data: Data | HeteroData, is_hetero: bool) -> None:
    """Add geometry from positions to NetworkX graph nodes."""
    if is_hetero:
        for node_type in data.node_types:
            if hasattr(data[node_type], "pos") and data[node_type].pos is not None:
                pos_data = data[node_type].pos.detach().cpu().numpy()
                # Get all nodes of this type using vectorized filtering
                nodes_with_data = [(node, node_data) for node, node_data in graph.nodes(data=True)]
                type_nodes = [node for node, node_data in nodes_with_data
                             if node_data.get("node_type", "default") == node_type]

                # Vectorized geometry creation and batch assignment
                if len(pos_data) >= 2 and pos_data.shape[1] >= 2:
                    valid_count = min(len(type_nodes), len(pos_data))
                    geometries = [Point(pos[0], pos[1]) for pos in pos_data[:valid_count]]
                    node_geom_pairs = list(zip(type_nodes[:valid_count], geometries, strict=False))
                    
                    # Batch update using dictionary comprehension
                    geom_updates = {node: {"geometry": geom} for node, geom in node_geom_pairs}
                    nx.set_node_attributes(graph, geom_updates)
    elif hasattr(data, "pos") and data.pos is not None:
        pos_data = data.pos.detach().cpu().numpy()
        nodes_list = list(graph.nodes())

        # Vectorized geometry creation for valid positions
        if len(pos_data) >= 2 and pos_data.shape[1] >= 2:
            valid_count = min(len(nodes_list), len(pos_data))
            geometries = [Point(pos[0], pos[1]) for pos in pos_data[:valid_count]]
            node_geom_pairs = list(zip(nodes_list[:valid_count], geometries, strict=False))
            
            # Batch update using dictionary comprehension
            geom_updates = {node: {"geometry": geom} for node, geom in node_geom_pairs}
            nx.set_node_attributes(graph, geom_updates)

def _restore_node_ids(graph: nx.Graph, data: Data | HeteroData) -> nx.Graph:
    """Restore original node IDs if available."""
    if hasattr(data, "_node_mappings"):
        # Process all node types at once
        all_node_mappings = {}
        nodes_with_data = list(graph.nodes(data=True))

        for node_type in data._node_mappings:
            mapping_info = data._node_mappings[node_type]
            
            # Handle both new unified structure and legacy direct mapping
            if isinstance(mapping_info, dict) and "mapping" in mapping_info:
                # New unified structure
                mapping = mapping_info["mapping"]
            else:
                # Legacy direct mapping
                mapping = mapping_info

            reverse_mapping = {idx: node_id for node_id, idx in mapping.items()}

            # Vectorized node mapping creation
            if node_type == "default":
                # For homogeneous graphs stored as "default"
                type_nodes = [(node, reverse_mapping.get(node, node))
                             for node, node_data in nodes_with_data]
            else:
                # For heterogeneous graphs
                type_nodes = [(node, reverse_mapping.get(node, node))
                             for node, node_data in nodes_with_data
                             if node_data.get("node_type", "default") == node_type]

            all_node_mappings.update(dict(type_nodes))

        if all_node_mappings:
            graph = nx.relabel_nodes(graph, all_node_mappings)
    return graph

def nx_to_pyg(
    graph: nx.Graph,
    node_feature_cols: list[str] | None = None,
    node_label_cols: list[str] | None = None,
    edge_feature_cols: list[str] | None = None,
    device: torch.device | str | None = None,
) -> Data:
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph to convert
    node_feature_cols : list[str], optional
        Node attribute names to use as features
    node_label_cols : list[str], optional
        Node attribute names to use as labels
    edge_feature_cols : list[str], optional
        Edge attribute names to use as features
    device : torch.device or str, optional
        Target device for tensor creation

    Returns
    -------
    Data
        PyTorch Geometric Data object
    """
    if not TORCH_AVAILABLE:
        raise ImportError(TORCH_ERROR_MSG)

    # Validate input graph
    graph = _validate_nx(graph)

    # Convert to GeoDataFrames first using existing utils functions
    from .utils import nx_to_gdf
    
    # Get nodes and edges GeoDataFrames
    nodes_gdf, edges_gdf = nx_to_gdf(graph, nodes=True, edges=True)
    
    # Convert to PyG using existing function
    return gdf_to_pyg(
        nodes=nodes_gdf,
        edges=edges_gdf,
        node_feature_cols=node_feature_cols,
        node_label_cols=node_label_cols,
        edge_feature_cols=edge_feature_cols,
        device=device,
    )

def _add_hetero_nodes_to_graph(graph: nx.Graph, data: HeteroData) -> dict[str, int]:
    """Add heterogeneous nodes to NetworkX graph and return node offsets."""
    node_offset = {}
    current_offset = 0
    
    for node_type in data.node_types:
        node_offset[node_type] = current_offset
        num_nodes = data[node_type].num_nodes
        
        # Get feature and label column names for this node type
        node_feature_cols = None
        if hasattr(data, "_node_feature_cols") and data._node_feature_cols:
            node_feature_cols = data._node_feature_cols.get(node_type)
        
        node_label_cols = None
        if hasattr(data, "_node_label_cols") and data._node_label_cols:
            node_label_cols = data._node_label_cols.get(node_type)
        
        # Add nodes with type information
        for i in range(num_nodes):
            node_id = current_offset + i
            attrs = {"node_type": node_type}
            
            # Add position if available
            if hasattr(data[node_type], "pos") and data[node_type].pos is not None:
                pos = data[node_type].pos[i].detach().cpu().numpy()
                attrs["pos"] = tuple(float(p) for p in pos)
            
            # Add features with original column names if available
            if hasattr(data[node_type], "x") and data[node_type].x is not None:
                x = data[node_type].x[i].detach().cpu().numpy()
                if node_feature_cols and len(node_feature_cols) > 0:
                    # Use original column names
                    for j, col_name in enumerate(node_feature_cols):
                        if j < len(x):
                            attrs[col_name] = float(x[j])
                else:
                    # Fallback to generic names
                    for j, feat in enumerate(x):
                        attrs[f"feat_{j}"] = float(feat)
            
            # Add labels with original column names if available
            if hasattr(data[node_type], "y") and data[node_type].y is not None:
                y = data[node_type].y[i].detach().cpu().numpy()
                if node_label_cols and len(node_label_cols) > 0:
                    # Use original column names
                    for j, col_name in enumerate(node_label_cols):
                        if j < len(y):
                            attrs[col_name] = float(y[j])
                else:
                    # Fallback to generic names
                    for j, label in enumerate(y):
                        attrs[f"label_{j}"] = float(label)
            
            graph.add_node(node_id, **attrs)
        
        current_offset += num_nodes
    
    return node_offset


def _add_hetero_edges_to_graph(graph: nx.Graph, data: HeteroData, node_offset: dict[str, int]) -> None:
    """Add heterogeneous edges to NetworkX graph."""
    for edge_type in data.edge_types:
        src_type, rel_type, dst_type = edge_type
        
        if hasattr(data[edge_type], "edge_index") and data[edge_type].edge_index is not None:
            edge_index = data[edge_type].edge_index.detach().cpu().numpy()
            
            # Get edge feature column names for this edge type
            edge_feature_cols = None
            if hasattr(data, "_edge_feature_cols") and data._edge_feature_cols:
                edge_feature_cols = data._edge_feature_cols.get(edge_type)
            
            for i in range(edge_index.shape[1]):
                src_idx = int(edge_index[0, i]) + node_offset[src_type]
                dst_idx = int(edge_index[1, i]) + node_offset[dst_type]
                
                edge_attrs = {"edge_type": rel_type}
                
                # Add edge features with original column names if available
                if hasattr(data[edge_type], "edge_attr") and data[edge_type].edge_attr is not None:
                    edge_attr = data[edge_type].edge_attr[i].detach().cpu().numpy()
                    if edge_feature_cols and len(edge_feature_cols) > 0:
                        # Use original column names
                        for j, col_name in enumerate(edge_feature_cols):
                            if j < len(edge_attr):
                                edge_attrs[col_name] = float(edge_attr[j])
                    else:
                        # Fallback to generic names
                        for j, feat in enumerate(edge_attr):
                            edge_attrs[f"edge_feat_{j}"] = float(feat)
                
                graph.add_edge(src_idx, dst_idx, **edge_attrs)


def _convert_hetero_pyg_to_nx(data: HeteroData) -> nx.Graph:
    """Convert heterogeneous PyG data to NetworkX graph."""
    graph = nx.Graph()
    
    # Add metadata to indicate this is a heterogeneous graph
    graph.graph["crs"] = getattr(data, "crs", None)
    graph.graph["is_heterogeneous"] = True
    
    # Store type information for reconstruction
    graph.graph["node_types"] = list(data.node_types)
    graph.graph["edge_types"] = list(data.edge_types)
    
    # Store original mappings and metadata for reconstruction
    if hasattr(data, "_node_mappings"):
        graph.graph["_node_mappings"] = data._node_mappings
    if hasattr(data, "_node_feature_cols"):
        graph.graph["_node_feature_cols"] = data._node_feature_cols
    if hasattr(data, "_node_label_cols"):
        graph.graph["_node_label_cols"] = data._node_label_cols
    if hasattr(data, "_edge_feature_cols"):
        graph.graph["_edge_feature_cols"] = data._edge_feature_cols
    if hasattr(data, "_node_index_names"):
        graph.graph["_node_index_names"] = data._node_index_names
    if hasattr(data, "_edge_index_names"):
        graph.graph["_edge_index_names"] = data._edge_index_names
    if hasattr(data, "_edge_index_values"):
        graph.graph["_edge_index_values"] = data._edge_index_values
    
    # Add nodes and get offsets
    node_offset = _add_hetero_nodes_to_graph(graph, data)
    
    # Add edges
    _add_hetero_edges_to_graph(graph, data, node_offset)
    
    # Store node offset for reconstruction
    graph.graph["node_offset"] = node_offset
    
    return graph


def _add_homo_nodes_to_graph(graph: nx.Graph, data: Data) -> None:
    """Add homogeneous nodes to NetworkX graph."""
    node_feature_cols = getattr(data, "_node_feature_cols", None)
    node_label_cols = getattr(data, "_node_label_cols", None)
    
    # Determine number of nodes properly
    num_nodes = 0
    if hasattr(data, "x") and data.x is not None:
        num_nodes = data.x.size(0)
    elif hasattr(data, "pos") and data.pos is not None:
        num_nodes = data.pos.size(0)
    elif hasattr(data, "y") and data.y is not None:
        num_nodes = data.y.size(0)
    
    # Add nodes with preserved attribute names
    for i in range(num_nodes):
        attrs = {}
        
        # Add position if available
        if hasattr(data, "pos") and data.pos is not None and i < data.pos.size(0):
            pos = data.pos[i].detach().cpu().numpy()
            attrs["pos"] = tuple(float(p) for p in pos)
        
        # Add features with original column names if available
        if hasattr(data, "x") and data.x is not None and i < data.x.size(0):
            x = data.x[i].detach().cpu().numpy()
            if node_feature_cols and len(node_feature_cols) > 0:
                # Use original column names
                for j, col_name in enumerate(node_feature_cols):
                    if j < len(x):
                        attrs[col_name] = float(x[j])
            else:
                # Fallback to generic names
                for j, feat in enumerate(x):
                    attrs[f"feat_{j}"] = float(feat)
        
        # Add labels with original column names if available
        if hasattr(data, "y") and data.y is not None and i < data.y.size(0):
            y = data.y[i].detach().cpu().numpy()
            if node_label_cols and len(node_label_cols) > 0:
                # Use original column names
                for j, col_name in enumerate(node_label_cols):
                    if j < len(y):
                        attrs[col_name] = float(y[j])
            else:
                # Fallback to generic names
                for j, label in enumerate(y):
                    attrs[f"label_{j}"] = float(label)
        
        graph.add_node(i, **attrs)


def _add_homo_edges_to_graph(graph: nx.Graph, data: Data) -> None:
    """Add homogeneous edges to NetworkX graph."""
    edge_feature_cols = getattr(data, "_edge_feature_cols", None)
    
    if hasattr(data, "edge_index") and data.edge_index is not None:
        edge_index = data.edge_index.detach().cpu().numpy()
        
        for i in range(edge_index.shape[1]):
            src_idx = int(edge_index[0, i])
            dst_idx = int(edge_index[1, i])
            
            edge_attrs = {}
            
            # Add edge features with original column names if available
            if hasattr(data, "edge_attr") and data.edge_attr is not None and i < data.edge_attr.size(0):
                edge_attr = data.edge_attr[i].detach().cpu().numpy()
                if edge_feature_cols and len(edge_feature_cols) > 0:
                    # Use original column names
                    for j, col_name in enumerate(edge_feature_cols):
                        if j < len(edge_attr):
                            edge_attrs[col_name] = float(edge_attr[j])
                else:
                    # Fallback to generic names
                    for j, feat in enumerate(edge_attr):
                        edge_attrs[f"edge_feat_{j}"] = float(feat)
            
            graph.add_edge(src_idx, dst_idx, **edge_attrs)


def _convert_homo_pyg_to_nx(data: Data) -> nx.Graph:
    """Convert homogeneous PyG data to NetworkX graph."""
    graph = nx.Graph()
    
    # Add metadata
    graph.graph["crs"] = getattr(data, "crs", None)
    
    # Add nodes and edges
    _add_homo_nodes_to_graph(graph, data)
    _add_homo_edges_to_graph(graph, data)
    
    # Store index information for reconstruction
    if hasattr(data, "_node_index_names"):
        graph.graph["node_index_names"] = data._node_index_names
    if hasattr(data, "_edge_index_names"):
        graph.graph["edge_index_names"] = data._edge_index_names
    
    return graph


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
        NetworkX graph representation.
    """
    if not TORCH_AVAILABLE:
        raise ImportError(TORCH_ERROR_MSG)

    _validate_pyg(data)

    is_hetero = hasattr(data, "node_types") and hasattr(data, "edge_types")

    if is_hetero:
        return _convert_hetero_pyg_to_nx(data)
    return _convert_homo_pyg_to_nx(data)


def _validate_pyg(data: Data | HeteroData) -> None:
    """
    Validate PyTorch Geometric Data or HeteroData objects.

    Parameters
    ----------
    data : Data or HeteroData
        PyTorch Geometric graph object to validate.

    Raises
    ------
    ValueError
        If required attributes are missing or inconsistent.
    """
    if not hasattr(data, "edge_index"):
        is_hetero = hasattr(data, "node_types") and hasattr(data, "edge_types")
        if not is_hetero:
            msg = "PyG object must have edge_index attribute"
            raise ValueError(msg)







