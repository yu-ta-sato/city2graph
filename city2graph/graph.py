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
"""

from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from city2graph.utils import _validate_gdf
from city2graph.utils import _validate_nx
from city2graph.utils import nx_to_gdf

# Try to import the PyTorch Geometric packages. If unavailable, issue a gentle warning.
try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.data import HeteroData

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - makes life easier for docs build.
    TORCH_AVAILABLE = False

    class HeteroData:
        """Fallback stub when torch is unavailable."""


    class Data:
        """Fallback stub when torch is unavailable."""


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


# ============================================================================
# GRAPH CONVERSION FUNCTIONS
# ============================================================================

def gdf_to_pyg(
    nodes: dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame,
    edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | gpd.GeoDataFrame | None = None,
    node_feature_cols: dict[str, list[str]] | list[str] | None = None,
    node_label_cols: dict[str, list[str]] | list[str] | None = None,
    edge_feature_cols: dict[str, list[str]] | list[str] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Data | HeteroData:
    """Convert GeoDataFrames (nodes/edges) to a PyTorch Geometric object.

    This function serves as the main entry point for converting spatial data into 
    PyTorch Geometric graph objects. It automatically detects whether to create 
    homogeneous or heterogeneous graphs based on input structure. Node identifiers
    are taken from the GeoDataFrame index. Edge relationships are defined by a
    MultiIndex on the edge GeoDataFrame (source ID, target ID).

    Parameters
    ----------
    nodes : dict[str, gpd.GeoDataFrame] or gpd.GeoDataFrame
        Node data. For homogeneous graphs, provide a single GeoDataFrame.
        For heterogeneous graphs, provide a dictionary mapping node type names 
        to their respective GeoDataFrames. The index of these GeoDataFrames
        will be used as node identifiers.
    edges : dict[tuple[str, str, str], gpd.GeoDataFrame] or gpd.GeoDataFrame, optional
        Edge data. For homogeneous graphs, provide a single GeoDataFrame.
        For heterogeneous graphs, provide a dictionary mapping edge type tuples
        (source_type, relation_type, target_type) to their GeoDataFrames.
        The GeoDataFrame must have a MultiIndex where the first level represents
        source node IDs and the second level represents target node IDs.
    node_feature_cols : dict[str, list[str]] or list[str], optional
        Column names to use as node features. For heterogeneous graphs,
        provide a dictionary mapping node types to their feature columns.
    node_label_cols : dict[str, list[str]] or list[str], optional
        Column names to use as node labels for supervised learning tasks.
        For heterogeneous graphs, provide a dictionary mapping node types
        to their label columns.
    edge_feature_cols : dict[str, list[str]] or list[str], optional
        Column names to use as edge features. For heterogeneous graphs,
        provide a dictionary mapping relation types to their feature columns.
    device : str or torch.device, optional
        Target device for tensor placement ('cpu', 'cuda', or torch.device).
        If None, automatically selects CUDA if available, otherwise CPU.
    dtype : torch.dtype, optional
        Data type for float tensors (e.g., torch.float32, torch.float16).
        If None, uses torch.float32 (default PyTorch float type).

    Returns
    -------
    Data or HeteroData
        PyTorch Geometric Data object for homogeneous graphs or HeteroData 
        object for heterogeneous graphs. The returned object contains:
        - Node features (x), positions (pos), and labels (y) if available
        - Edge connectivity (edge_index) and features (edge_attr) if available
        - Metadata for reconstruction including ID mappings and column names

    Raises
    ------
    ImportError
        If PyTorch Geometric is not installed
    ValueError
        If input GeoDataFrames are invalid or incompatible

    Examples
    --------
    >>> # Homogeneous graph from single GeoDataFrames
    >>> nodes_gdf = gpd.read_file("nodes.geojson").set_index("node_id")
    >>> edges_gdf = gpd.read_file("edges.geojson").set_index(["source_id", "target_id"])
    >>> data = gdf_to_pyg(nodes_gdf, edges_gdf, 
    ...                   node_feature_cols=['population', 'area'])

    >>> # Heterogeneous graph from dictionaries
    >>> buildings_gdf = buildings_gdf.set_index("building_id")
    >>> roads_gdf = roads_gdf.set_index("road_id")
    >>> connections_gdf = connections_gdf.set_index(["building_id", "road_id"])
    >>> nodes_dict = {'building': buildings_gdf, 'road': roads_gdf}
    >>> edges_dict = {('building', 'connects', 'road'): connections_gdf}
    >>> data = gdf_to_pyg(nodes_dict, edges_dict)

    Notes
    -----
    - Preserves original coordinate reference systems (CRS)
    - Maintains index structure for bidirectional conversion
    - Handles both Point and non-Point geometries (using centroids)
    - Creates empty tensors for missing features/edges
    """
    # ------------------------------------------------------------------
    # 0. Input validation & dispatch
    # ------------------------------------------------------------------
    if not TORCH_AVAILABLE:
        raise ImportError(TORCH_ERROR_MSG)

    # Validate input GeoDataFrames
    is_hetero = isinstance(nodes, dict)
    if is_hetero:
        [_validate_gdf(node_gdf, None) for node_gdf in nodes.values()]
        if edges:
            [_validate_gdf(None, edge_gdf) for edge_gdf in edges.values() if edge_gdf is not None]
    else:
        _validate_gdf(nodes, edges)

    device = _get_device(device)

    is_hetero = isinstance(nodes, dict)
    if is_hetero:
        data = _build_heterogeneous_graph(
            nodes, edges or {}, node_feature_cols, node_label_cols,
            edge_feature_cols or {}, device, dtype,
        )
    else:
        nodes_gdf: gpd.GeoDataFrame = nodes
        edges_gdf: gpd.GeoDataFrame | None = edges
        node_feature_cols_list: list[str] | None = node_feature_cols
        node_label_cols_list: list[str] | None = node_label_cols
        edge_feature_cols_list: list[str] | None = edge_feature_cols

        # Create a homogeneous Data object
        data = _build_homogeneous_graph(
            nodes_gdf, edges_gdf, node_feature_cols_list,
            node_label_cols_list,
            edge_feature_cols_list, device, dtype,
        )

    # Validate the created PyG object
    _validate_pyg(data)
    return data


def pyg_to_gdf(
    data: Data | HeteroData,
    node_types: str | list[str] | None = None,
    edge_types: str | list[tuple[str, str, str]] | None = None,
) -> (
    tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
    | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]
):
    """Convert PyTorch Geometric data to GeoDataFrames.

    Reconstructs the original GeoDataFrame structure from PyTorch Geometric 
    Data or HeteroData objects. This function provides bidirectional conversion
    capability, preserving spatial information, feature data, and metadata.

    Parameters
    ----------
    data : Data or HeteroData
        PyTorch Geometric data object to convert back to GeoDataFrames
    node_types : str or list[str], optional
        For heterogeneous graphs, specify which node types to reconstruct.
        If None, reconstructs all available node types.
    edge_types : str or list[tuple[str, str, str]], optional
        For heterogeneous graphs, specify which edge types to reconstruct.
        Edge types are specified as (source_type, relation_type, target_type) tuples.
        If None, reconstructs all available edge types.

    Returns
    -------
    For HeteroData:
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
            First element: dictionary mapping node type names to node GeoDataFrames
            Second element: dictionary mapping edge type tuples to edge GeoDataFrames
    For Data:
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame | None]
            First element: nodes GeoDataFrame
            Second element: edges GeoDataFrame (None if no edges)

    Notes
    -----
    - Preserves original index structure and names when available
    - Reconstructs geometry from stored position tensors
    - Maintains coordinate reference system (CRS) information
    - Converts feature tensors back to named DataFrame columns
    - Handles both homogeneous and heterogeneous graph structures
    """
    metadata = _validate_pyg(data)

    if metadata["is_hetero"]:
        # ------------------------------------------------------------------
        # HeteroData → pandas
        # ------------------------------------------------------------------
        node_types_to_process = node_types or metadata["node_types"]
        edge_types_to_process = edge_types or metadata["edge_types"]

        node_gdfs = {
            nt: _reconstruct_node_gdf(data, nt, metadata) for nt in node_types_to_process
        }
        edge_gdfs = {
            et: _reconstruct_edge_gdf(data, et, metadata) for et in edge_types_to_process
        }
        return node_gdfs, edge_gdfs

    # ------------------------------------------------------------------
    # Data → pandas
    # ------------------------------------------------------------------
    nodes_gdf = _reconstruct_node_gdf(data, None, metadata)
    edges_gdf = _reconstruct_edge_gdf(data, None, metadata)
    return nodes_gdf, edges_gdf


# ============================================================================
# NETWORKX CONVERSION FUNCTIONS
# ============================================================================

def pyg_to_nx(data: Data | HeteroData) -> nx.Graph:
    """Convert a PyTorch Geometric object to a NetworkX graph.

    Converts PyTorch Geometric Data or HeteroData objects to NetworkX graphs,
    preserving node and edge features as graph attributes. This enables 
    compatibility with the extensive NetworkX ecosystem for graph analysis.

    Parameters
    ----------
    data : Data or HeteroData
        PyTorch Geometric data object to convert

    Returns
    -------
    nx.Graph
        NetworkX graph with node and edge attributes from the PyG object.
        For heterogeneous graphs, node and edge types are stored as attributes.

    Raises
    ------
    ImportError
        If PyTorch Geometric is not installed

    Notes
    -----
    - Node features, positions, and labels are stored as node attributes
    - Edge features are stored as edge attributes
    - For heterogeneous graphs, type information is preserved
    - Geometry information is converted from tensor positions
    """
    if not TORCH_AVAILABLE:
        raise ImportError(TORCH_ERROR_MSG)

    metadata = _validate_pyg(data)

    if metadata["is_hetero"]:
        return _convert_hetero_pyg_to_nx(data)
    return _convert_homo_pyg_to_nx(data)


def nx_to_pyg(
    graph: nx.Graph,
    node_feature_cols: list[str] | None = None,
    node_label_cols: list[str] | None = None,
    edge_feature_cols: list[str] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Data:
    """Convert NetworkX graph to PyTorch Geometric Data object.

    Converts a NetworkX graph to a PyTorch Geometric Data object by first
    converting to GeoDataFrames then using the main conversion pipeline.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph to convert
    node_feature_cols : list[str], optional
        List of node attribute names to use as features
    node_label_cols : list[str], optional
        List of node attribute names to use as labels
    edge_feature_cols : list[str], optional
        List of edge attribute names to use as features
    device : torch.device or str, optional
        Target device for tensor placement
    dtype : torch.dtype, optional
        Data type for float tensors (e.g., torch.float32, torch.float16).
        If None, uses torch.float32 (default PyTorch float type).

    Returns
    -------
    Data
        PyTorch Geometric Data object

    Raises
    ------
    ImportError
        If PyTorch Geometric is not installed
    ValueError
        If the NetworkX graph is invalid or empty

    Notes
    -----
    - Uses intermediate GeoDataFrame conversion for consistency
    - Preserves all graph attributes and metadata
    - Handles spatial coordinates if present in node attributes
    """
    if not TORCH_AVAILABLE:
        raise ImportError(TORCH_ERROR_MSG)

    # Validate NetworkX graph
    _validate_nx(graph)

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
        dtype=dtype,
    )


# ============================================================================
# TORCH UTILITIES FUNCTIONS
# ============================================================================

def is_torch_available() -> bool:
    """Check if PyTorch Geometric is available.

    Returns
    -------
    bool
        True if PyTorch Geometric can be imported, False otherwise
    """
    return TORCH_AVAILABLE


def _get_device(device: str | torch.device | None = None) -> torch.device:
    """Normalize the device argument and return a torch.device instance.

    Parameters
    ----------
    device : str, torch.device, or None
        Device specification. Can be 'cpu', 'cuda', a torch.device object, or None.
        If None, automatically selects CUDA if available, otherwise CPU.

    Returns
    -------
    torch.device
        Normalized torch.device object

    Raises
    ------
    ImportError
        If PyTorch is not available
    ValueError
        If device string is not 'cpu' or 'cuda', or if 'cuda' is selected but not available.
    TypeError
        If device is not a valid type
    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, str):
        device_lower = device.lower()
        if device_lower not in {"cpu", "cuda"}:
            raise ValueError(DEVICE_ERROR_MSG)
        if device_lower == "cuda" and not torch.cuda.is_available():
            # Raise ValueError consistent with the test's expectation
            msg = f"CUDA selected, but not available. {DEVICE_ERROR_MSG}"
            raise ValueError(msg)
        return torch.device(device_lower)
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            # Also handle cases where a torch.device("cuda") object is passed
            # when CUDA is not available.
            msg = f"CUDA selected, but not available. {DEVICE_ERROR_MSG}"
            raise ValueError(msg)
        return device
    raise TypeError(DEVICE_ERROR_MSG)


# ============================================================================
# EDGE COLUMN DETECTION FUNCTIONS
# ============================================================================

# Removed: _get_source_target_keywords, _find_column_candidates, _fallback_column_detection, _detect_edge_columns
# These functions are no longer needed as edge relationships are derived from MultiIndex.


# ============================================================================
# NODE PREPARATION FUNCTIONS
# ============================================================================

def _create_node_id_mapping(
    node_gdf: gpd.GeoDataFrame,
) -> tuple[dict[str | int, int], str, list[str | int]]:
    """Create mapping from node IDs (from index) to sequential integer indices.

    PyTorch Geometric requires nodes to be identified by sequential integers starting from 0.
    This function creates the necessary mapping from original node identifiers (taken from
    the GeoDataFrame index) to these indices.

    Parameters
    ----------
    node_gdf : gpd.GeoDataFrame
        GeoDataFrame containing node data. The index is used for node IDs.

    Returns
    -------
    dict[str | int, int]
        Dictionary mapping original IDs to integer indices
    str
        Always "index", indicating the DataFrame index was used.
    list[str | int]
        List of original IDs in order
    """
    # Use DataFrame index as the node identifier
    original_ids = node_gdf.index.tolist()
    id_mapping = {node_id: i for i, node_id in enumerate(original_ids)}
    return id_mapping, "index", original_ids


def _create_node_features(
    node_gdf: gpd.GeoDataFrame,
    feature_cols: list[str] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert node attributes to PyTorch feature tensors.

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
    dtype = dtype or torch.float

    if feature_cols is None:
        # Return empty tensor when no feature columns specified
        return torch.zeros((len(node_gdf), 0), dtype=dtype, device=device)

    # Find valid columns that exist in the GeoDataFrame
    valid_cols = list(set(feature_cols) & set(node_gdf.columns))
    if valid_cols:
        # Convert to numpy array with consistent float32 type
        features_array = node_gdf[valid_cols].to_numpy().astype(np.float32)
        return torch.from_numpy(features_array).to(device=device, dtype=dtype)

    # Return empty tensor if no valid columns found
    return torch.zeros((len(node_gdf), 0), dtype=dtype, device=device)


def _create_node_positions(
    node_gdf: gpd.GeoDataFrame, device: str | torch.device | None = None,
) -> torch.Tensor | None:
    """Extract spatial coordinates from node geometries.

    Converts geometric representations to coordinate tensors suitable for
    spatial graph neural networks. Handles various geometry types and
    provides consistent coordinate extraction.

    Parameters
    ----------
    node_gdf : gpd.GeoDataFrame
        GeoDataFrame with geometry column containing spatial data
    device : str or torch.device, optional
        Target device for tensor creation

    Returns
    -------
    torch.Tensor or None
        Float tensor of shape (num_nodes, 2) containing [x, y] coordinates
        None if no geometry column found

    Notes
    -----
    - Uses centroid coordinates for all geometry types.
    - Coordinates are in the original CRS of the GeoDataFrame
    """
    # Get the device for tensor creation
    device = _get_device(device)

    # Get the geometry column
    geom_series = node_gdf.geometry

    # Get centroids of geometries
    centroids = geom_series.centroid
    pos_data = np.column_stack([
        centroids.x.to_numpy(),
        centroids.y.to_numpy(),
    ])

    return torch.tensor(pos_data, dtype=torch.float, device=device)


# ============================================================================
# EDGE PREPARATION FUNCTIONS
# ============================================================================

def _create_edge_features(
    edge_gdf: gpd.GeoDataFrame,
    feature_cols: list[str] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Convert edge attributes to PyTorch feature tensors.

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
    dtype = dtype or torch.float

    # If no feature columns specified, return empty tensor
    if feature_cols is None:
        return torch.empty((edge_gdf.shape[0], 0), dtype=dtype, device=device)

    # Find valid columns that exist in the GeoDataFrame
    valid_cols = list(set(feature_cols) & set(edge_gdf.columns))
    if not valid_cols:
        return torch.empty((edge_gdf.shape[0], 0), dtype=dtype, device=device)

    # Select only numeric columns from valid_cols to prevent conversion errors
    numeric_cols = edge_gdf[valid_cols].select_dtypes(include=np.number).columns.tolist()

    # Convert to numpy array with consistent float32 type
    features_array = edge_gdf[numeric_cols].to_numpy().astype(np.float32)
    return torch.from_numpy(features_array).to(device=device, dtype=dtype)


def _create_edge_indices(
    edge_gdf: gpd.GeoDataFrame,
    source_mapping: dict[str | int, int],
    target_mapping: dict[str | int, int] | None = None,
) -> list[list[int]]:
    """Create edge connectivity matrix from edge data using MultiIndex."""
    target_mapping = target_mapping or source_mapping

    # Extract source and target IDs from MultiIndex
    source_ids, target_ids = _extract_edge_ids(edge_gdf)

    # Convert types if needed and validate
    source_ids = pd.Series(source_ids) if isinstance(source_ids, pd.Index) else source_ids
    target_ids = pd.Series(target_ids) if isinstance(target_ids, pd.Index) else target_ids

    return _map_edge_ids_to_indices(source_ids, target_ids, source_mapping, target_mapping)


def _extract_edge_ids(edge_gdf: gpd.GeoDataFrame) -> tuple[pd.Series, pd.Series]:
    """Extract source and target IDs from MultiIndex DataFrame."""
    return (edge_gdf.index.get_level_values(0),  # First level = source
            edge_gdf.index.get_level_values(1))   # Second level = target


def _map_edge_ids_to_indices(
    source_ids: pd.Series,
    target_ids: pd.Series,
    source_mapping: dict[str | int, int],
    target_mapping: dict[str | int, int],
) -> list[list[int]]:
    """Map edge IDs to indices."""
    # Find edges with valid source and target nodes
    valid_src_mask = source_ids.isin(source_mapping.keys())
    valid_dst_mask = target_ids.isin(target_mapping.keys())
    valid_edges_mask = valid_src_mask & valid_dst_mask

    # Process valid edges using vectorized operations
    valid_sources = source_ids[valid_edges_mask]
    valid_targets = target_ids[valid_edges_mask]

    # Map original node IDs to integer indices
    from_indices = valid_sources.map(source_mapping).to_numpy()
    to_indices = valid_targets.map(target_mapping).to_numpy()

    return np.column_stack([from_indices, to_indices]).tolist()


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


# ============================================================================
# GRAPH BUILDING FUNCTIONS
# ============================================================================

def _build_homogeneous_graph(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame | None = None,
    node_feature_cols: list[str] | None = None,
    node_label_cols: list[str] | None = None,
    edge_feature_cols: list[str] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Data:
    """
    Construct a homogeneous PyTorch Geometric Data object.

    Creates a single-type graph where all nodes and edges are treated uniformly.
    Node IDs are taken from the nodes_gdf index. Edge relationships are taken
    from the edges_gdf MultiIndex (source_id, target_id).

    Processing Pipeline:
    1. Create node ID mapping (original IDs from index → integer indices)
    2. Extract node features and positions from geometry
    3. Process node labels if available
    4. Create edge connectivity matrix
    5. Extract edge features
    6. Package everything into PyG Data object
    7. Store metadata for bidirectional conversion

    Args:
        nodes_gdf: GeoDataFrame containing node data (index used for IDs)
        edges_gdf: GeoDataFrame containing edge data (MultiIndex used for relationships)
        node_feature_cols: Columns to use as node features
        node_label_cols: Columns to use as node labels
        edge_feature_cols: Columns to use as edge features
        device: Target device for tensor creation
        dtype: Data type for float tensors

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
    id_mapping, id_col_name, original_ids = _create_node_id_mapping(nodes_gdf)

    x = _create_node_features(nodes_gdf, node_feature_cols, device, dtype)
    pos = _create_node_positions(nodes_gdf, device)

    # Handle labels
    y = None
    if node_label_cols:
        y = _create_node_features(nodes_gdf, node_label_cols, device, dtype)

    # Edge processing
    edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    edge_attr = torch.empty((0, 0), dtype=dtype or torch.float, device=device)

    if edges_gdf is not None and not edges_gdf.empty:
        edge_pairs = _create_edge_indices(
            edges_gdf, id_mapping, id_mapping,
        )
        if edge_pairs:
            edge_index = torch.tensor(
                np.array(edge_pairs).T, dtype=torch.long, device=device,
            )
        edge_attr = _create_edge_features(edges_gdf, edge_feature_cols, device, dtype)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)

    # Store metadata - use unified _node_mappings structure
    data._node_mappings = {
        "default": {
            "mapping": id_mapping,
            "id_col": id_col_name,
            "original_ids": original_ids,
        },
    }
    data._node_feature_cols = node_feature_cols or []
    data._node_label_cols = node_label_cols or []
    data._edge_feature_cols = edge_feature_cols or []

    # Store index names and values for preservation
    data._node_index_names = nodes_gdf.index.names if hasattr(nodes_gdf.index, "names") else None
    if edges_gdf is not None and hasattr(edges_gdf.index, "names"):
        data._edge_index_names = edges_gdf.index.names

        # Store original edge index values for reconstruction
        data._edge_index_values = [edges_gdf.index.get_level_values(i).tolist()
                                    for i in range(edges_gdf.index.nlevels)]
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
    node_feature_cols: dict[str, list[str]] | None = None,
    node_label_cols: dict[str, list[str]] | None = None,
    edge_feature_cols: dict[str, list[str]] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> HeteroData:
    """Build heterogeneous PyTorch Geometric HeteroData object."""
    device = _get_device(device)
    data = HeteroData()

    # Default empty dicts
    edges_dict = edges_dict or {}

    # Process nodes and get mappings
    node_mappings = _process_hetero_nodes(
        data, nodes_dict, node_feature_cols, node_label_cols, device, dtype,
    )

    # Process edges
    _process_hetero_edges(
        data, edges_dict, node_mappings, edge_feature_cols, device, dtype,
    )

    # Store metadata
    _store_hetero_metadata(
        data, node_mappings, nodes_dict, edges_dict, node_feature_cols, node_label_cols, edge_feature_cols,
    )

    return data


def _process_hetero_nodes(
    data: HeteroData,
    nodes_dict: dict[str, gpd.GeoDataFrame],
    node_feature_cols: dict[str, list[str]] | None,
    node_label_cols: dict[str, list[str]] | None,
    device: str | torch.device | None,
    dtype: torch.dtype | None,
) -> dict[str, dict]:
    """Process all node types for heterogeneous graph."""
    node_mappings = {}
    device = _get_device(device)

    for node_type, node_gdf in nodes_dict.items():
        id_mapping, id_col_name, original_ids = _create_node_id_mapping(node_gdf)

        # Store mapping with metadata in unified structure
        node_mappings[node_type] = {
            "mapping": id_mapping,
            "id_col": id_col_name,
            "original_ids": original_ids,
        }

        # Features
        feature_cols = node_feature_cols.get(node_type) if node_feature_cols else None
        data[node_type].x = _create_node_features(node_gdf, feature_cols, device, dtype)

        # Positions
        data[node_type].pos = _create_node_positions(node_gdf, device)

        # Labels
        label_cols = node_label_cols.get(node_type) if node_label_cols else None
        if label_cols:
            data[node_type].y = _create_node_features(node_gdf, label_cols, device, dtype)

    return node_mappings


def _process_hetero_edges(
    data: HeteroData,
    edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    node_mappings: dict[str, dict],
    edge_feature_cols: dict[str, list[str]] | None,
    device: str | torch.device | None,
    dtype: torch.dtype | None,
) -> None:
    """Process all edge types for heterogeneous graph."""
    device = _get_device(device)

    for edge_type, edge_gdf in edges_dict.items():
        # Extract source, relation, and destination types from edge_type tuple
        src_type, rel_type, dst_type = edge_type

        # Get the mapping dictionaries (not the full metadata)
        src_mapping = node_mappings[src_type]["mapping"]
        dst_mapping = node_mappings[dst_type]["mapping"]

        if edge_gdf is not None and not edge_gdf.empty:
            edge_pairs = _create_edge_indices(
                edge_gdf, src_mapping, dst_mapping,
            )
            edge_index = (torch.tensor(np.array(edge_pairs).T, dtype=torch.long, device=device)
                         if edge_pairs else torch.zeros((2, 0), dtype=torch.long, device=device))
            data[edge_type].edge_index = edge_index

            feature_cols = edge_feature_cols.get(rel_type) if edge_feature_cols else None
            data[edge_type].edge_attr = _create_edge_features(edge_gdf, feature_cols, device, dtype)
        else:
            data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            data[edge_type].edge_attr = torch.empty((0, 0), dtype=dtype or torch.float, device=device)


def _store_hetero_metadata(
    data: HeteroData,
    node_mappings: dict[str, dict],
    nodes_dict: dict[str, gpd.GeoDataFrame],
    edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    node_feature_cols: dict[str, list[str]] | None,
    node_label_cols: dict[str, list[str]] | None,
    edge_feature_cols: dict[str, list[str]] | None,
) -> None:
    """Store metadata for heterogeneous graph."""
    # Store mappings and column metadata
    data._node_mappings = node_mappings
    data._node_feature_cols = node_feature_cols or {}
    data._node_label_cols = node_label_cols or {}
    data._edge_feature_cols = edge_feature_cols or {}

    # Store index names for reconstruction
    data._node_index_names = {}
    for node_type, node_gdf in nodes_dict.items():
        if hasattr(node_gdf.index, "names"):
            data._node_index_names[node_type] = node_gdf.index.names

    # Store edge index names and values for reconstruction
    data._edge_index_names = {}
    data._edge_index_values = {}
    for edge_type, edge_gdf in edges_dict.items():
        if edge_gdf is not None and hasattr(edge_gdf.index, "names"):
            # Store edge index names
            data._edge_index_names[edge_type] = edge_gdf.index.names

            # Store original edge index values for reconstruction
            data._edge_index_values[edge_type] = [edge_gdf.index.get_level_values(i).tolist()
                                                for i in range(edge_gdf.index.nlevels)]

    # Set CRS
    crs_values = [gdf.crs for gdf in nodes_dict.values() if hasattr(gdf, "crs") and gdf.crs]
    if crs_values and all(crs == crs_values[0] for crs in crs_values):
        data.crs = crs_values[0]


# ============================================================================
# GRAPH VALIDATION FUNCTIONS
# ============================================================================

def _validate_pyg(data: Data | HeteroData) -> dict[str, Any]:
    """
    Validate PyTorch Geometric Data or HeteroData objects and return metadata.

    This centralized validation function checks all necessary attributes and
    returns comprehensive metadata to eliminate redundant hasattr() checks
    throughout the codebase.
    """
    if not TORCH_AVAILABLE:
        msg = "PyTorch required. Install with: pip install city2graph[torch]"
        raise ImportError(msg)

    is_hetero = isinstance(data, HeteroData)
    metadata = {
        "is_hetero": is_hetero,
        "has_node_mappings": hasattr(data, "_node_mappings"),
        "has_node_feature_cols": hasattr(data, "_node_feature_cols"),
        "has_node_label_cols": hasattr(data, "_node_label_cols"),
        "has_edge_feature_cols": hasattr(data, "_edge_feature_cols"),
        "has_node_index_names": hasattr(data, "_node_index_names"),
        "has_edge_index_names": hasattr(data, "_edge_index_names"),
        "has_edge_index_values": hasattr(data, "_edge_index_values"),
        "has_crs": hasattr(data, "crs"),
    }

    if is_hetero:
        node_types = list(data.node_types) if hasattr(data, "node_types") else []
        edge_types = list(data.edge_types) if hasattr(data, "edge_types") else []
        metadata.update({"node_types": node_types, "edge_types": edge_types})

        # Check each node type
        for node_type in node_types:
            node_data = data[node_type]
            has_x = hasattr(node_data, "x") and node_data.x is not None
            has_pos = hasattr(node_data, "pos") and node_data.pos is not None
            has_y = hasattr(node_data, "y") and node_data.y is not None
            metadata.update({
                f"{node_type}_has_x": has_x,
                f"{node_type}_has_pos": has_pos,
                f"{node_type}_has_y": has_y,
            })

        # Check each edge type
        for edge_type in edge_types:
            edge_data = data[edge_type]
            has_edge_index = hasattr(edge_data, "edge_index") and edge_data.edge_index is not None
            has_edge_attr = hasattr(edge_data, "edge_attr") and edge_data.edge_attr is not None
            metadata.update({
                f"{edge_type}_has_edge_index": has_edge_index,
                f"{edge_type}_has_edge_attr": has_edge_attr,
            })
    else:
        has_x = hasattr(data, "x") and data.x is not None
        has_pos = hasattr(data, "pos") and data.pos is not None
        has_y = hasattr(data, "y") and data.y is not None
        has_edge_index = hasattr(data, "edge_index") and data.edge_index is not None
        has_edge_attr = hasattr(data, "edge_attr") and data.edge_attr is not None

        metadata.update({
            "has_x": has_x,
            "has_pos": has_pos,
            "has_y": has_y,
            "has_edge_index": has_edge_index,
            "has_edge_attr": has_edge_attr,
        })

    return metadata


# ============================================================================
# GRAPH RECONSTRUCTION FUNCTIONS
# ============================================================================

def _extract_tensor_data(
    tensor: torch.Tensor | None, column_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract data from tensor with proper column names."""
    if tensor is None or tensor.numel() == 0:
        return {}

    features_array = tensor.detach().cpu().numpy()

    num_cols = min(len(column_names), features_array.shape[1])
    return {column_names[i]: features_array[:, i] for i in range(num_cols)}


def _get_node_data_info(data: Data | HeteroData, node_type: str | None, metadata: dict[str, Any]) -> tuple[Data, int]:
    """Get node data and number of nodes."""
    node_data = data[node_type] if metadata["is_hetero"] and node_type else data
    return node_data, int(node_data.num_nodes)


def _get_mapping_info(data: Data | HeteroData, node_type: str | None, metadata: dict[str, Any]) -> dict | None:
    """Get mapping info for the given node type."""
    mapping_key = "default" if not metadata["is_hetero"] or not node_type else node_type
    return data._node_mappings.get(mapping_key)


def _extract_index_values(mapping_info: dict, num_nodes: int) -> list | None:
    """Extract index values from mapping info."""
    original_ids = mapping_info.get("original_ids", list(range(num_nodes)))
    return original_ids[:num_nodes]


def _create_geometry_from_positions(node_data: Data) -> gpd.array.GeometryArray | None:
    """Create geometry from node positions."""
    pos_array = node_data.pos.detach().cpu().numpy()
    return gpd.points_from_xy(pos_array[:, 0], pos_array[:, 1])


def _extract_node_features_and_labels(
    data: Data | HeteroData, node_data: Data, node_type: str | None, metadata: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Extract features and labels from node data."""
    gdf_data = {}
    is_hetero = metadata["is_hetero"]

    # Extract features
    has_x_key = f"{node_type}_has_x" if is_hetero and node_type else "has_x"

    if metadata.get(has_x_key, False) and metadata["has_node_feature_cols"]:
        feature_cols = data._node_feature_cols
        cols = feature_cols.get(node_type) if is_hetero and node_type else feature_cols
        features_dict = _extract_tensor_data(node_data.x, cols)
        gdf_data.update(features_dict)

    # Extract labels
    has_y_key = f"{node_type}_has_y" if is_hetero and node_type else "has_y"

    if metadata.get(has_y_key, False) and metadata["has_node_label_cols"]:
        label_cols = data._node_label_cols
        cols = label_cols.get(node_type) if is_hetero and node_type else label_cols
        labels_dict = _extract_tensor_data(node_data.y, cols)
        gdf_data.update(labels_dict)

    return gdf_data


def _set_gdf_index_and_crs(
    gdf: gpd.GeoDataFrame, data: Data | HeteroData, node_type: str | None, metadata: dict[str, Any],
) -> None:
    """Set index names and CRS on GeoDataFrame."""
    # Set index names
    if metadata["has_node_index_names"]:
        index_names = None
        # Get index names based on heterogeneity and node type
        if metadata["is_hetero"] and node_type and node_type in data._node_index_names:
            index_names = data._node_index_names[node_type]
        elif not metadata["is_hetero"]:
            index_names = data._node_index_names

        # Set index name if available
        if (index_names and hasattr(gdf.index, "names") and
            isinstance(index_names, list) and len(index_names) > 0):
                gdf.index.name = index_names[0]

    # Set CRS
    if metadata["has_crs"]:
        gdf.crs = data.crs


def _reconstruct_node_gdf(
    data: Data | HeteroData, node_type: str | None = None, metadata: dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    """Reconstruct node GeoDataFrame from PyTorch Geometric data."""
    node_data, num_nodes = _get_node_data_info(data, node_type, metadata)
    mapping_info = _get_mapping_info(data, node_type, metadata)

    # Extract node IDs and features/labels
    gdf_data = {}
    features_labels = _extract_node_features_and_labels(data, node_data, node_type, metadata)
    gdf_data.update(features_labels)

    # Create geometry and index
    geometry = _create_geometry_from_positions(node_data)
    index_values = _extract_index_values(mapping_info, num_nodes) if mapping_info else None

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(gdf_data, geometry=geometry, index=index_values)
    _set_gdf_index_and_crs(gdf, data, node_type, metadata)

    return gdf


def _reconstruct_edge_index(
    data: Data | HeteroData,
    edge_type: str | tuple[str, str, str] | None,
    is_hetero: bool,
    edge_data_dict: dict[str, list | np.ndarray],
) -> pd.Index | pd.MultiIndex | None:
    """Reconstruct edge index from stored values."""
    stored_values = None
    if is_hetero and edge_type and edge_type in data._edge_index_values:
        stored_values = data._edge_index_values[edge_type]
    elif not is_hetero and data._edge_index_values:
        stored_values = data._edge_index_values

    if not stored_values:
        return None

    # Determine number of rows based on edge data or stored values
    num_rows = len(next(iter(edge_data_dict.values()))) if edge_data_dict else len(stored_values[0])

    # Handle MultiIndex case
    arrays = [stored_values[i][:num_rows] for i in range(len(stored_values))]
    return pd.MultiIndex.from_arrays(arrays)


def _extract_edge_features(
    edge_data: Data, edge_type: str | tuple | None, is_hetero: bool, data: Data | HeteroData,
) -> dict[str, np.ndarray]:
    """Extract edge features from edge data."""
    edge_data_dict = {}
    if hasattr(edge_data, "edge_attr") and edge_data.edge_attr is not None:
        feature_cols = getattr(data, "_edge_feature_cols", {})
        if is_hetero and edge_type:
                rel_type = edge_type[1]
                cols = feature_cols.get(rel_type)
        else:
            cols = feature_cols
        features_dict = _extract_tensor_data(edge_data.edge_attr, cols)
        edge_data_dict.update(features_dict)
    return edge_data_dict


def _create_edge_geometries(
    edge_data: Data, edge_type: str | tuple | None, is_hetero: bool, data: Data | HeteroData,
) -> gpd.array.GeometryArray | None:
    """Create edge geometries from edge indices and node positions."""
    # Get edge index array
    edge_index_array = edge_data.edge_index.detach().cpu().numpy()

    # Set default positions as None
    src_pos = dst_pos = None

    # If hetero and specific edge type, get source and destination positions
    if is_hetero and isinstance(edge_type, tuple) and len(edge_type) == 3:
        src_type, _, dst_type = edge_type
        if hasattr(data[src_type], "pos") and data[src_type].pos is not None:
            src_pos = data[src_type].pos.detach().cpu().numpy()
        if hasattr(data[dst_type], "pos") and data[dst_type].pos is not None:
            dst_pos = data[dst_type].pos.detach().cpu().numpy()

    # If not hetero or no specific edge type, use default positions
    elif hasattr(data, "pos") and data.pos is not None:
        pos = data.pos.detach().cpu().numpy()
        src_pos = dst_pos = pos

    geometries = _create_linestring_geometries(edge_index_array, src_pos, dst_pos)
    return gpd.array.from_shapely(geometries)


def _set_edge_index_names(
    gdf: gpd.GeoDataFrame, data: Data | HeteroData, edge_type: str | tuple | None, is_hetero: bool,
) -> None:
    """Set index names on edge GeoDataFrame."""
    index_names = None
    if is_hetero and edge_type and edge_type in data._edge_index_names:
        index_names = data._edge_index_names[edge_type]
    elif not is_hetero and data._edge_index_names:
        index_names = data._edge_index_names

    if (hasattr(gdf.index, "names") and isinstance(index_names, list) and
        len(index_names) > 1 and isinstance(gdf.index, pd.MultiIndex)):
        gdf.index.names = index_names

def _reconstruct_edge_gdf(
    data: Data | HeteroData, edge_type: str | tuple[str, str, str] | None = None, metadata: dict[str, Any] | None = None,
) -> gpd.GeoDataFrame:
    """Reconstruct edge GeoDataFrame from PyTorch Geometric data."""
    is_hetero = metadata["is_hetero"]

    edge_data = data[edge_type] if is_hetero and edge_type else data

    # Extract edge features
    edge_data_dict = _extract_edge_features(edge_data, edge_type, is_hetero, data)

    # Create geometries from edge indices and node positions
    geometry = _create_edge_geometries(edge_data, edge_type, is_hetero, data)

    # Reconstruct index from stored values
    index_values = _reconstruct_edge_index(data, edge_type, is_hetero, edge_data_dict)

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(edge_data_dict, geometry=geometry, index=index_values)

    # Set index names if available
    _set_edge_index_names(gdf, data, edge_type, is_hetero)

    # Set CRS
    if hasattr(data, "crs") and data.crs:
        gdf.crs = data.crs

    return gdf


# ============================================================================
# NETWORKX CONVERSION HELPERS
# ============================================================================

def _add_features_to_attrs(
    tensor_data: np.ndarray,
    column_names: list[str] | None,
    attrs: dict[str, Any],
    prefix: str,
) -> None:
    """Add tensor data to attribute dictionary with proper column names."""
    if tensor_data is None or tensor_data.size == 0:
        return

    if column_names and len(column_names) > 0:
        # Use original column names
        # Ensure we don't go out of bounds for tensor_data
        num_elements_to_process = min(len(column_names), len(tensor_data))
        update_dict = {
            column_names[j]: float(tensor_data[j])
            for j in range(num_elements_to_process)
        }
        attrs.update(update_dict)
    else:
        # Fallback to generic names
        update_dict = {
            f"{prefix}_{j}": float(value)
            for j, value in enumerate(tensor_data)
        }
        attrs.update(update_dict)


def _add_node_attributes(
    node_data: Data,
    node_id: int,
    attrs: dict[str, Any],
    feature_cols: list[str] | None,
    label_cols: list[str] | None,
) -> None:
    """Add node attributes (position, features, labels) to attribute dictionary."""
    # Add position if available
    if (hasattr(node_data, "pos") and node_data.pos is not None and
        node_id < node_data.pos.size(0)):
        pos = node_data.pos[node_id].detach().cpu().numpy()
        attrs["pos"] = tuple(float(p) for p in pos)

    # Add features
    if (hasattr(node_data, "x") and node_data.x is not None and
        node_id < node_data.x.size(0)):
        x = node_data.x[node_id].detach().cpu().numpy()
        _add_features_to_attrs(x, feature_cols, attrs, "feat")

    # Add labels
    if (hasattr(node_data, "y") and node_data.y is not None and
        node_id < node_data.y.size(0)):
        y = node_data.y[node_id].detach().cpu().numpy()
        _add_features_to_attrs(y, label_cols, attrs, "label")


def _add_edge_attributes(
    edge_data: Data, edge_id: int, attrs: dict[str, Any], feature_cols: list[str] | None,
) -> None:
    """Add edge attributes to attribute dictionary."""
    if (hasattr(edge_data, "edge_attr") and edge_data.edge_attr is not None and
        edge_id < edge_data.edge_attr.size(0)):
        edge_attr = edge_data.edge_attr[edge_id].detach().cpu().numpy()
        _add_features_to_attrs(edge_attr, feature_cols, attrs, "edge_feat")


def _add_hetero_nodes_to_graph(graph: nx.Graph, data: HeteroData) -> dict[str, int]:
    """Add heterogeneous nodes to NetworkX graph and return node offsets."""
    node_offset = {}
    current_offset = 0

    for node_type in data.node_types:
        node_offset[node_type] = current_offset
        num_nodes = data[node_type].num_nodes

        # Get feature and label column names for this node type
        node_feature_cols = getattr(data, "_node_feature_cols", {}).get(node_type)
        node_label_cols = getattr(data, "_node_label_cols", {}).get(node_type)

        # Add nodes with type information
        for i in range(num_nodes):
            node_id = current_offset + i
            attrs = {"node_type": node_type}
            _add_node_attributes(
                data[node_type], i, attrs, node_feature_cols, node_label_cols,
            )
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
            edge_feature_cols = getattr(data, "_edge_feature_cols", {}).get(rel_type)

            # Add edges using comprehension
            [graph.add_edge(
                int(edge_index[0, i]) + node_offset[src_type],
                int(edge_index[1, i]) + node_offset[dst_type],
                edge_type=rel_type,
                **{k: v for attrs in [{}] for k, v in (
                    _add_edge_attributes(data[edge_type], i, attrs, edge_feature_cols) or attrs
                ).items()},
            ) for i in range(edge_index.shape[1])]


def _add_homo_nodes_to_graph(graph: nx.Graph, data: Data) -> None:
    """Add homogeneous nodes to NetworkX graph."""
    node_feature_cols = getattr(data, "_node_feature_cols", None)
    node_label_cols = getattr(data, "_node_label_cols", None)

    # Determine number of nodes
    num_nodes = data.x.size(0)

    # Add nodes with preserved attribute names using comprehension
    [graph.add_node(i, **{
        k: v for attrs in [{}]
        for k, v in (_add_node_attributes(data, i, attrs, node_feature_cols, node_label_cols) or attrs).items()
    }) for i in range(num_nodes)]


def _add_homo_edges_to_graph(graph: nx.Graph, data: Data) -> None:
    """Add homogeneous edges to NetworkX graph."""
    edge_feature_cols = getattr(data, "_edge_feature_cols", None)

    if hasattr(data, "edge_index") and data.edge_index is not None:
        edge_index = data.edge_index.detach().cpu().numpy()

        # Add edges using comprehension
        [(lambda i: (
            lambda src_idx, dst_idx, edge_attrs: graph.add_edge(src_idx, dst_idx, **edge_attrs)
        )(
            int(edge_index[0, i]),
            int(edge_index[1, i]),
            {k: v for attrs in [{}] for k, v in (
                _add_edge_attributes(data, i, attrs, edge_feature_cols) or attrs
            ).items()},
        ))(i) for i in range(edge_index.shape[1])]


def _convert_homo_pyg_to_nx(data: Data) -> nx.Graph:
    """Convert homogeneous PyG data to NetworkX graph."""
    graph = nx.Graph()

    # Add metadata
    graph.graph["crs"] = getattr(data, "crs", None)
    graph.graph["is_hetero"] = False

    # Add nodes and edges
    _add_homo_nodes_to_graph(graph, data)
    _add_homo_edges_to_graph(graph, data)

    # Store index information for reconstruction
    if hasattr(data, "_node_index_names"):
        graph.graph["node_index_names"] = data._node_index_names
    if hasattr(data, "_edge_index_names"):
        graph.graph["edge_index_names"] = data._edge_index_names

    return graph


def _convert_hetero_pyg_to_nx(data: HeteroData) -> nx.Graph:
    """Convert heterogeneous PyG data to NetworkX graph."""
    graph = nx.Graph()

    # Add metadata
    graph.graph["crs"] = getattr(data, "crs", None)
    graph.graph["is_hetero"] = True
    graph.graph["node_types"] = list(data.node_types)
    graph.graph["edge_types"] = list(data.edge_types)

    # Store metadata for reconstruction
    for attr_name in ["_node_mappings", "_node_feature_cols", "_node_label_cols",
                      "_edge_feature_cols", "_node_index_names", "_edge_index_names", "_edge_index_values"]:
        if hasattr(data, attr_name):
            graph.graph[attr_name] = getattr(data, attr_name)

    # Add nodes and edges
    node_offset = _add_hetero_nodes_to_graph(graph, data)
    _add_hetero_edges_to_graph(graph, data, node_offset)
    graph.graph["node_offset"] = node_offset

    return graph
