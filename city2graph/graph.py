"""
Module for creating heterogeneous graph representations of urban environments.

This module provides comprehensive functionality for converting spatial data
(GeoDataFrames and NetworkX objects) into PyTorch Geometric Data and HeteroData objects,
supporting both homogeneous and heterogeneous graphs. It handles the complex mapping between
geographical coordinates, node/edge features, and the tensor representations
required by graph neural networks.

The module serves as a bridge between geospatial data analysis tools and deep
learning frameworks, enabling seamless integration of spatial urban data with
Graph Neural Networks (GNNs) for tasks such as urban modeling, traffic prediction,
and spatial analysis.
"""

# Future annotations for type hints
from __future__ import annotations

# Standard library imports
import logging
from typing import TYPE_CHECKING

# Third-party imports
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString

# Local imports
from city2graph.utils import GraphMetadata
from city2graph.utils import nx_to_gdf
from city2graph.utils import validate_gdf
from city2graph.utils import validate_nx

# PyTorch Geometric imports with availability checking
try:
    import torch
    from torch_geometric.data import Data
    from torch_geometric.data import HeteroData

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - makes life easier for docs build.
    TORCH_AVAILABLE = False

    # Create stubs for documentation and fallback functionality
    if TYPE_CHECKING:
        from torch_geometric.data import Data
        from torch_geometric.data import HeteroData
    else:
        torch = None

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
    "validate_pyg",
]

# Constants for error messages
TORCH_ERROR_MSG = "PyTorch and PyTorch Geometric required for graph conversion functionality."
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
    """
    Convert GeoDataFrames (nodes/edges) to a PyTorch Geometric object.

    This function serves as the main entry point for converting spatial data into
    PyTorch Geometric graph objects. It automatically detects whether to create
    homogeneous or heterogeneous graphs based on input structure. Node identifiers
    are taken from the GeoDataFrame index. Edge relationships are defined by a
    MultiIndex on the edge GeoDataFrame (source ID, target ID).

    Parameters
    ----------
    nodes : dict[str, geopandas.GeoDataFrame] or geopandas.GeoDataFrame
        Node data. For homogeneous graphs, provide a single GeoDataFrame.
        For heterogeneous graphs, provide a dictionary mapping node type names
        to their respective GeoDataFrames. The index of these GeoDataFrames
        will be used as node identifiers.
    edges : dict[tuple[str, str, str], geopandas.GeoDataFrame] or geopandas.GeoDataFrame, optional
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
    torch_geometric.data.Data or torch_geometric.data.HeteroData
        PyTorch Geometric Data object for homogeneous graphs or HeteroData
        object for heterogeneous graphs. The returned object contains:

        - Node features (x), positions (pos), and labels (y) if available
        - Edge connectivity (edge_index) and features (edge_attr) if available
        - Metadata for reconstruction including ID mappings and column names

    Raises
    ------
    ImportError
        If PyTorch Geometric is not installed.
    ValueError
        If input GeoDataFrames are invalid or incompatible.

    See Also
    --------
    pyg_to_gdf : Convert PyTorch Geometric data back to GeoDataFrames.
    nx_to_pyg : Convert NetworkX graph to PyTorch Geometric object.
    city2graph.utils.validate_gdf : Validate GeoDataFrame structure.

    Notes
    -----
    This function automatically detects the graph type based on input structure.
    For heterogeneous graphs, provide dictionaries mapping types to GeoDataFrames.
    Node positions are automatically extracted from geometry centroids when available.
    - Preserves original coordinate reference systems (CRS)
    - Maintains index structure for bidirectional conversion
    - Handles both Point and non-Point geometries (using centroids)
    - Creates empty tensors for missing features/edges
    - For heterogeneous graphs, ensures consistent node/edge type mapping

    Examples
    --------
    Create a homogeneous graph from single GeoDataFrames:

    >>> import geopandas as gpd
    >>> from city2graph.graph import gdf_to_pyg
    >>>
    >>> # Load and prepare node data
    >>> nodes_gdf = gpd.read_file("nodes.geojson").set_index("node_id")
    >>> edges_gdf = gpd.read_file("edges.geojson").set_index(["source_id", "target_id"])
    >>>
    >>> # Convert to PyTorch Geometric
    >>> data = gdf_to_pyg(nodes_gdf, edges_gdf,
    ...                   node_feature_cols=['population', 'area'])

    Create a heterogeneous graph from dictionaries:

    >>> # Prepare heterogeneous data
    >>> buildings_gdf = buildings_gdf.set_index("building_id")
    >>> roads_gdf = roads_gdf.set_index("road_id")
    >>> connections_gdf = connections_gdf.set_index(["building_id", "road_id"])
    >>>
    >>> # Define node and edge types
    >>> nodes_dict = {'building': buildings_gdf, 'road': roads_gdf}
    >>> edges_dict = {('building', 'connects', 'road'): connections_gdf}
    >>>
    >>> # Convert to heterogeneous graph with labels
    >>> data = gdf_to_pyg(nodes_dict, edges_dict,
    ...                   node_label_cols={'building': ['type'], 'road': ['category']})
    """
    # ------------------------------------------------------------------
    # 0. Input validation & dispatch
    # ------------------------------------------------------------------
    if not TORCH_AVAILABLE:
        raise ImportError(TORCH_ERROR_MSG)

    # Validate input GeoDataFrames and get type information
    nodes, edges, is_hetero = validate_gdf(nodes_gdf=nodes, edges_gdf=edges)

    device = _get_device(device)

    if is_hetero:
        # Type assertions for heterogeneous graphs
        assert isinstance(nodes, dict)
        assert edges is None or isinstance(edges, dict)

        # Type narrowing for heterogeneous graphs
        if isinstance(node_feature_cols, dict) or node_feature_cols is None:
            node_feature_cols_hetero: dict[str, list[str]] | None = node_feature_cols
        else:
            msg = "node_feature_cols must be a dict for heterogeneous graphs"
            raise TypeError(msg)

        if isinstance(node_label_cols, dict) or node_label_cols is None:
            node_label_cols_hetero: dict[str, list[str]] | None = node_label_cols
        else:
            msg = "node_label_cols must be a dict for heterogeneous graphs"
            raise TypeError(msg)

        if isinstance(edge_feature_cols, dict) or edge_feature_cols is None:
            edge_feature_cols_hetero: dict[str, list[str]] | None = edge_feature_cols
        else:
            msg = "edge_feature_cols must be a dict for heterogeneous graphs"
            raise TypeError(msg)

        data = _build_heterogeneous_graph(
            nodes,
            edges,
            node_feature_cols_hetero,
            node_label_cols_hetero,
            edge_feature_cols_hetero,
            device,
            dtype,
        )
    else:
        # Type assertions for homogeneous graphs
        assert isinstance(nodes, gpd.GeoDataFrame) or nodes is None
        assert isinstance(edges, gpd.GeoDataFrame) or edges is None

        # Type narrowing for homogeneous graphs
        if isinstance(node_feature_cols, list) or node_feature_cols is None:
            node_feature_cols_homo: list[str] | None = node_feature_cols
        else:
            msg = "node_feature_cols must be a list for homogeneous graphs"
            raise TypeError(msg)

        if isinstance(node_label_cols, list) or node_label_cols is None:
            node_label_cols_homo: list[str] | None = node_label_cols
        else:
            msg = "node_label_cols must be a list for homogeneous graphs"
            raise TypeError(msg)

        if isinstance(edge_feature_cols, list) or edge_feature_cols is None:
            edge_feature_cols_homo: list[str] | None = edge_feature_cols
        else:
            msg = "edge_feature_cols must be a list for homogeneous graphs"
            raise TypeError(msg)

        # Create a homogeneous Data object
        data = _build_homogeneous_graph(
            nodes,
            edges,
            node_feature_cols_homo,
            node_label_cols_homo,
            edge_feature_cols_homo,
            device,
            dtype,
        )

    # Validate the created PyG object
    validate_pyg(data)
    return data


def pyg_to_gdf(
    data: Data | HeteroData,
    node_types: str | list[str] | None = None,
    edge_types: str | list[tuple[str, str, str]] | None = None,
) -> (
    tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
    | tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None]
):
    """
    Convert PyTorch Geometric data to GeoDataFrames.

    Reconstructs the original GeoDataFrame structure from PyTorch Geometric
    Data or HeteroData objects. This function provides bidirectional conversion
    capability, preserving spatial information, feature data, and metadata.

    Parameters
    ----------
    data : torch_geometric.data.Data or torch_geometric.data.HeteroData
        PyTorch Geometric data object to convert back to GeoDataFrames.
    node_types : str or list[str], optional
        For heterogeneous graphs, specify which node types to reconstruct.
        If None, reconstructs all available node types.
    edge_types : str or list[tuple[str, str, str]], optional
        For heterogeneous graphs, specify which edge types to reconstruct.
        Edge types are specified as (source_type, relation_type, target_type) tuples.
        If None, reconstructs all available edge types.

    Returns
    -------
    tuple
        **For HeteroData input:** Returns a tuple containing:
            - First element: dict[str, geopandas.GeoDataFrame] mapping node type names to
              GeoDataFrames
            - Second element: dict[tuple[str, str, str], geopandas.GeoDataFrame] mapping
              edge types to GeoDataFrames

        **For Data input:** Returns a tuple containing:
            - First element: geopandas.GeoDataFrame containing nodes
            - Second element: geopandas.GeoDataFrame containing edges (or None if no edges)

    See Also
    --------
    gdf_to_pyg : Convert GeoDataFrames to PyTorch Geometric object.
    pyg_to_nx : Convert PyTorch Geometric data to NetworkX graph.

    Notes
    -----
    - Preserves original index structure and names when available
    - Reconstructs geometry from stored position tensors
    - Maintains coordinate reference system (CRS) information
    - Converts feature tensors back to named DataFrame columns
    - Handles both homogeneous and heterogeneous graph structures

    Examples
    --------
    Convert homogeneous PyTorch Geometric data back to GeoDataFrames:

    >>> from city2graph.graph import pyg_to_gdf
    >>>
    >>> # Convert back to GeoDataFrames
    >>> nodes_gdf, edges_gdf = pyg_to_gdf(data)

    Convert heterogeneous data with specific node types:

    >>> # Convert only specific node types
    >>> node_gdfs, edge_gdfs = pyg_to_gdf(hetero_data,
    ...                                   node_types=['building', 'road'])
    """
    metadata = validate_pyg(data)

    if metadata.is_hetero:
        # ------------------------------------------------------------------
        # HeteroData → pandas
        # ------------------------------------------------------------------
        node_types_to_process = node_types or metadata.node_types
        edge_types_to_process = edge_types or metadata.edge_types

        node_gdfs = {
            nt: _reconstruct_node_gdf(data, metadata, node_type=nt) for nt in node_types_to_process
        }
        edge_gdfs = {et: _reconstruct_edge_gdf(data, metadata, et) for et in edge_types_to_process}
        return node_gdfs, edge_gdfs

    # ------------------------------------------------------------------
    # Data → pandas
    # ------------------------------------------------------------------
    nodes_gdf = _reconstruct_node_gdf(data, metadata, None)
    edges_gdf = _reconstruct_edge_gdf(data, metadata, None)
    return nodes_gdf, edges_gdf


# ============================================================================
# NETWORKX CONVERSION FUNCTIONS
# ============================================================================


def pyg_to_nx(data: Data | HeteroData) -> nx.Graph:
    """
    Convert a PyTorch Geometric object to a NetworkX graph.

    Converts PyTorch Geometric Data or HeteroData objects to NetworkX graphs,
    preserving node and edge features as graph attributes. This enables
    compatibility with the extensive NetworkX ecosystem for graph analysis.

    Parameters
    ----------
    data : torch_geometric.data.Data or torch_geometric.data.HeteroData
        PyTorch Geometric data object to convert.

    Returns
    -------
    networkx.Graph
        NetworkX graph with node and edge attributes from the PyG object.
        For heterogeneous graphs, node and edge types are stored as attributes.

    Raises
    ------
    ImportError
        If PyTorch Geometric is not installed.

    See Also
    --------
    nx_to_pyg : Convert NetworkX graph to PyTorch Geometric object.
    pyg_to_gdf : Convert PyTorch Geometric data to GeoDataFrames.

    Notes
    -----
    - Node features, positions, and labels are stored as node attributes
    - Edge features are stored as edge attributes
    - For heterogeneous graphs, type information is preserved
    - Geometry information is converted from tensor positions
    - Maintains compatibility with NetworkX analysis algorithms

    Examples
    --------
    Convert PyTorch Geometric data to NetworkX:

    >>> from city2graph.graph import pyg_to_nx
    >>> import networkx as nx
    >>>
    >>> # Convert to NetworkX graph
    >>> nx_graph = pyg_to_nx(data)
    >>>
    >>> # Use NetworkX algorithms
    >>> centrality = nx.betweenness_centrality(nx_graph)
    >>> communities = nx.community.greedy_modularity_communities(nx_graph)
    """
    metadata = validate_pyg(data)

    if metadata.is_hetero:
        return _convert_hetero_pyg_to_nx(data, metadata)
    return _convert_homo_pyg_to_nx(data, metadata)


def nx_to_pyg(
    graph: nx.Graph,
    node_feature_cols: list[str] | None = None,
    node_label_cols: list[str] | None = None,
    edge_feature_cols: list[str] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Data | HeteroData:
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Converts a NetworkX Graph to a PyTorch Geometric Data object by first
    converting to GeoDataFrames then using the main conversion pipeline. This
    provides a bridge between NetworkX's rich graph analysis tools and PyTorch
    Geometric's deep learning capabilities.

    Parameters
    ----------
    graph : networkx.Graph
        NetworkX graph to convert.
    node_feature_cols : list[str], optional
        List of node attribute names to use as features.
    node_label_cols : list[str], optional
        List of node attribute names to use as labels.
    edge_feature_cols : list[str], optional
        List of edge attribute names to use as features.
    device : torch.device or str, optional
        Target device for tensor placement ('cpu', 'cuda', or torch.device).
        If None, automatically selects CUDA if available, otherwise CPU.
    dtype : torch.dtype, optional
        Data type for float tensors (e.g., torch.float32, torch.float16).
        If None, uses torch.float32 (default PyTorch float type).

    Returns
    -------
    torch_geometric.data.Data or torch_geometric.data.HeteroData
        PyTorch Geometric Data object for homogeneous graphs or HeteroData
        object for heterogeneous graphs. The returned object contains:

        - Node features (x), positions (pos), and labels (y) if available
        - Edge connectivity (edge_index) and features (edge_attr) if available
        - Metadata for reconstruction including ID mappings and column names

    Raises
    ------
    ImportError
        If PyTorch Geometric is not installed.
    ValueError
        If the NetworkX graph is invalid or empty.

    See Also
    --------
    pyg_to_nx : Convert PyTorch Geometric data to NetworkX graph.
    gdf_to_pyg : Convert GeoDataFrames to PyTorch Geometric object.
    city2graph.utils.nx_to_gdf : Convert NetworkX graph to GeoDataFrames.

    Notes
    -----
    - Uses intermediate GeoDataFrame conversion for consistency
    - Preserves all graph attributes and metadata
    - Handles spatial coordinates if present in node attributes
    - Maintains compatibility with existing city2graph workflows
    - Automatically creates geometry from 'x', 'y' coordinates if available

    Examples
    --------
    Convert a NetworkX graph with spatial data:

    >>> import networkx as nx
    >>> from city2graph.graph import nx_to_pyg
    >>>
    >>> # Create NetworkX graph with spatial attributes
    >>> G = nx.Graph()
    >>> G.add_node(0, x=0.0, y=0.0, population=1000)
    >>> G.add_node(1, x=1.0, y=1.0, population=1500)
    >>> G.add_edge(0, 1, weight=0.5, road_type='primary')
    >>>
    >>> # Convert to PyTorch Geometric
    >>> data = nx_to_pyg(G,
    ...                  node_feature_cols=['population'],
    ...                  edge_feature_cols=['weight'])

    Convert from graph analysis results:

    >>> # Use NetworkX for analysis, then convert for ML
    >>> communities = nx.community.greedy_modularity_communities(G)
    >>> # Add community labels to nodes
    >>> for i, community in enumerate(communities):
    ...     for node in community:
    ...         G.nodes[node]['community'] = i
    >>>
    >>> # Convert with community labels
    >>> data = nx_to_pyg(G, node_label_cols=['community'])
    """
    # Validate NetworkX graph (includes type checking)
    validate_nx(graph)

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
    """
    Check if PyTorch Geometric is available.

    This utility function checks whether the required PyTorch and PyTorch Geometric
    packages are installed and can be imported. It's useful for conditional
    functionality and providing helpful error messages.

    Returns
    -------
    bool
        True if PyTorch Geometric can be imported, False otherwise.

    See Also
    --------
    gdf_to_pyg : Convert GeoDataFrames to PyTorch Geometric (requires torch).
    pyg_to_gdf : Convert PyTorch Geometric to GeoDataFrames (requires torch).

    Notes
    -----
    - Returns False if either PyTorch or PyTorch Geometric is missing
    - Used internally by torch-dependent functions to provide helpful error messages

    Examples
    --------
    Check availability before using torch-dependent functions:

    >>> from city2graph.graph import is_torch_available
    >>>
    >>> if is_torch_available():
    ...     from city2graph.graph import gdf_to_pyg
    ...     data = gdf_to_pyg(nodes_gdf, edges_gdf)
    ... else:
    ...     print("PyTorch Geometric not available.")
    """
    return TORCH_AVAILABLE


def _get_device(device: str | torch.device | None) -> torch.device:
    """
    Normalize the device argument and return a torch.device instance.

    This function provides a consistent interface for device specification across
    the library, handling automatic device selection and validation.

    Parameters
    ----------
    device : str, torch.device, or None
        Device specification. Can be 'cpu', 'cuda', a torch.device object, or None.
        If None, automatically selects CUDA if available, otherwise CPU.

    Returns
    -------
    torch.device
        Normalized torch.device object.

    Raises
    ------
    ImportError
        If PyTorch is not available.
    ValueError
        If device string is not 'cpu' or 'cuda', or if 'cuda' is selected but not available.
    TypeError
        If device is not a valid type.

    See Also
    --------
    torch.device : PyTorch device specification.

    Examples
    --------
    >>> device = _normalize_device("cuda")
    >>> device = _normalize_device(None)  # Auto-selects best available
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Check for invalid types first
    if not isinstance(device, (str, torch.device)):
        raise TypeError(DEVICE_ERROR_MSG)

    try:
        result_device = torch.device(device)
    except RuntimeError as e:
        # Convert RuntimeError from torch.device() to ValueError for consistency
        raise ValueError(DEVICE_ERROR_MSG) from e

    if result_device.type == "cuda" and not torch.cuda.is_available():
        msg = f"CUDA selected, but not available. {DEVICE_ERROR_MSG}"
        raise ValueError(msg)

    return result_device


# ============================================================================
# EDGE COLUMN DETECTION FUNCTIONS
# ============================================================================

# Removed: _get_source_target_keywords, _find_column_candidates,
# _fallback_column_detection, _detect_edge_columns
# These functions are no longer needed as edge relationships are derived from MultiIndex.


# ============================================================================
# NODE PREPARATION FUNCTIONS
# ============================================================================


def _create_node_id_mapping(
    node_gdf: gpd.GeoDataFrame,
) -> tuple[dict[str | int, int], str, list[str | int]]:
    """
    Create mapping from node IDs (from index) to sequential integer indices.

    PyTorch Geometric requires nodes to be identified by sequential integers starting from 0.
    This function creates the necessary mapping from original node identifiers (taken from
    the GeoDataFrame index) to these indices.

    Parameters
    ----------
    node_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing node data. The index is used for node IDs.

    Returns
    -------
    dict[str | int, int]
        Dictionary mapping original IDs to integer indices.
    str
        Always "index", indicating the DataFrame index was used.
    list[str | int]
        List of original IDs in order.

    See Also
    --------
    _create_node_features : Convert node attributes to tensors.

    Examples
    --------
    >>> import geopandas as gpd
    >>> gdf = gpd.GeoDataFrame({'id': [1, 2, 3]})
    >>> mapping, node_type, ids = _create_node_mapping(gdf)
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
    """
    Convert node attributes to PyTorch feature tensors.

    Extracts numerical data from specified columns and converts to a tensor suitable
    for graph neural network processing. Handles missing columns gracefully and
    ensures consistent data types.

    Parameters
    ----------
    node_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing node data.
    feature_cols : list[str], optional
        List of column names to use as features (None creates empty tensor).
    device : str or torch.device, optional
        Target device for tensor creation.
    dtype : torch.dtype, optional
        Data type for the tensor.

    Returns
    -------
    torch.Tensor
        Float tensor of shape (num_nodes, num_features) containing node features.

    See Also
    --------
    _create_node_positions : Extract spatial coordinates from geometries.

    Examples
    --------
    >>> import geopandas as gpd
    >>> gdf = gpd.GeoDataFrame({'feature1': [1, 2], 'feature2': [3, 4]})
    >>> tensor = _create_node_features(gdf, ['feature1', 'feature2'])
    """
    device = _get_device(device)
    dtype = dtype or torch.float32

    if feature_cols is None:
        # Return empty tensor when no feature columns specified
        return torch.zeros((len(node_gdf), 0), dtype=dtype, device=device)

    # Find valid columns that exist in the GeoDataFrame
    valid_cols = list(set(feature_cols) & set(node_gdf.columns))
    if valid_cols:
        # Map torch dtype to numpy dtype for consistency
        numpy_dtype = torch.tensor(0, dtype=dtype).numpy().dtype
        features_array = node_gdf[valid_cols].to_numpy().astype(numpy_dtype)
        return torch.from_numpy(features_array).to(device=device, dtype=dtype)

    # Return empty tensor if no valid columns found
    return torch.zeros((len(node_gdf), 0), dtype=dtype, device=device)


def _create_node_positions(
    node_gdf: gpd.GeoDataFrame,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | None:
    """
    Extract spatial coordinates from node geometries.

    Converts geometric representations to coordinate tensors suitable for
    spatial graph neural networks. Handles various geometry types and
    provides consistent coordinate extraction.

    Parameters
    ----------
    node_gdf : geopandas.GeoDataFrame
        GeoDataFrame with geometry column containing spatial data.
    device : str or torch.device, optional
        Target device for tensor creation.
    dtype : torch.dtype, optional
        Data type for position tensors. If None, uses torch.float32.

    Returns
    -------
    torch.Tensor or None
        Float tensor of shape (num_nodes, 2) containing [x, y] coordinates.
        None if no geometry column found.

    See Also
    --------
    _create_node_features : Convert node attributes to tensors.

    Notes
    -----
    - Uses centroid coordinates for all geometry types.
    - Coordinates are in the original CRS of the GeoDataFrame.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point
    >>> gdf = gpd.GeoDataFrame(geometry=[Point(0, 0), Point(1, 1)])
    >>> coords = _create_node_positions(gdf)
    """
    # Get the device for tensor creation
    device = _get_device(device)
    dtype = dtype or torch.float32

    # Get the geometry column
    geom_series = node_gdf.geometry

    # Get centroids of geometries
    centroids = geom_series.centroid

    # Map torch dtype to numpy dtype for consistency
    numpy_dtype = torch.tensor(0, dtype=dtype).numpy().dtype
    pos_data = np.column_stack(
        [
            centroids.x.to_numpy(),
            centroids.y.to_numpy(),
        ],
    ).astype(numpy_dtype)

    return torch.tensor(pos_data, dtype=dtype, device=device)


# ============================================================================
# EDGE PREPARATION FUNCTIONS
# ============================================================================


def _create_edge_features(
    edge_gdf: gpd.GeoDataFrame,
    feature_cols: list[str] | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Convert edge attributes to PyTorch feature tensors.

    Similar to node features but for edge data. Commonly used for edge weights,
    distances, or other relationship-specific attributes.

    Parameters
    ----------
    edge_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing edge data.
    feature_cols : list[str], optional
        List of column names to use as features.
    device : str or torch.device, optional
        Target device for tensor creation.
    dtype : torch.dtype, optional
        Data type for the tensor.

    Returns
    -------
    torch.Tensor
        Float tensor of shape (num_edges, num_features) containing edge features.

    See Also
    --------
    _create_node_features : Convert node attributes to tensors.

    Examples
    --------
    >>> import geopandas as gpd
    >>> gdf = gpd.GeoDataFrame({'weight': [1.0, 2.0]})
    >>> tensor = _create_edge_features(gdf, ['weight'])
    """
    device = _get_device(device)
    dtype = dtype or torch.float32

    # If no feature columns specified, return empty tensor
    if feature_cols is None:
        return torch.empty((edge_gdf.shape[0], 0), dtype=dtype, device=device)

    # Find valid columns that exist in the GeoDataFrame
    valid_cols = list(set(feature_cols) & set(edge_gdf.columns))
    if not valid_cols:
        return torch.empty((edge_gdf.shape[0], 0), dtype=dtype, device=device)

    # Select only numeric columns from valid_cols to prevent conversion errors
    numeric_cols = edge_gdf[valid_cols].select_dtypes(include=np.number).columns.tolist()

    # Map torch dtype to numpy dtype for consistency
    numpy_dtype = torch.tensor(0, dtype=dtype).numpy().dtype
    features_array = edge_gdf[numeric_cols].to_numpy().astype(numpy_dtype)
    return torch.from_numpy(features_array).to(device=device, dtype=dtype)


def _create_edge_indices(
    edge_gdf: gpd.GeoDataFrame,
    source_mapping: dict[str | int, int],
    target_mapping: dict[str | int, int] | None = None,
) -> list[list[int]]:
    """
    Create edge connectivity matrix from edge data using MultiIndex.

    Extracts source and target node IDs from the MultiIndex of the edge GeoDataFrame
    and maps them to sequential integer indices required by PyTorch Geometric.

    Parameters
    ----------
    edge_gdf : geopandas.GeoDataFrame
        GeoDataFrame with MultiIndex containing (source_id, target_id) pairs.
    source_mapping : dict[str | int, int]
        Mapping from original source node IDs to integer indices.
    target_mapping : dict[str | int, int], optional
        Mapping from original target node IDs to integer indices.
        If None, uses source_mapping.

    Returns
    -------
    list[list[int]]
        Edge connectivity matrix as [source_indices, target_indices].

    See Also
    --------
    _create_node_mapping : Create node ID mappings.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_tuples([(0, 1), (1, 2)])
    >>> gdf = gpd.GeoDataFrame(index=idx)
    >>> mapping = {0: 0, 1: 1, 2: 2}
    >>> edges = _create_edge_index(gdf, mapping)
    """
    target_mapping = target_mapping or source_mapping

    # Extract source and target IDs from MultiIndex
    source_ids, target_ids = _extract_edge_ids(edge_gdf)

    # Convert types if needed and validate
    source_ids = pd.Series(source_ids) if isinstance(source_ids, pd.Index) else source_ids
    target_ids = pd.Series(target_ids) if isinstance(target_ids, pd.Index) else target_ids

    return _map_edge_ids_to_indices(source_ids, target_ids, source_mapping, target_mapping)


def _extract_edge_ids(edge_gdf: gpd.GeoDataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Extract source and target IDs from MultiIndex DataFrame.

    This helper function extracts the source and target node identifiers from
    the two levels of a MultiIndex, which represent edge relationships.

    Parameters
    ----------
    edge_gdf : geopandas.GeoDataFrame
        GeoDataFrame with MultiIndex containing (source_id, target_id) pairs.

    Returns
    -------
    tuple[pd.Series, pd.Series]
        Source IDs and target IDs from the MultiIndex levels.

    See Also
    --------
    _create_edge_index : Create edge connectivity matrix.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> idx = pd.MultiIndex.from_tuples([(0, 1), (1, 2)])
    >>> gdf = gpd.GeoDataFrame(index=idx)
    >>> src, tgt = _extract_edge_ids(gdf)
    """
    return (
        edge_gdf.index.get_level_values(0),  # First level = source
        edge_gdf.index.get_level_values(1),
    )  # Second level = target


def _map_edge_ids_to_indices(
    source_ids: pd.Series,
    target_ids: pd.Series,
    source_mapping: dict[str | int, int],
    target_mapping: dict[str | int, int],
) -> list[list[int]]:
    """
    Map edge IDs to indices.

    This function converts original edge node IDs to sequential integer indices
    required by PyTorch Geometric, filtering out invalid edges in the process.

    Parameters
    ----------
    source_ids : pd.Series
        Series of source node IDs.
    target_ids : pd.Series
        Series of target node IDs.
    source_mapping : dict[str | int, int]
        Mapping from source node IDs to indices.
    target_mapping : dict[str | int, int]
        Mapping from target node IDs to indices.

    Returns
    -------
    list[list[int]]
        Edge connectivity matrix as [source_indices, target_indices].

    See Also
    --------
    _extract_edge_ids : Extract IDs from MultiIndex.

    Examples
    --------
    >>> import pandas as pd
    >>> src = pd.Series([0, 1])
    >>> tgt = pd.Series([1, 2])
    >>> mapping = {0: 0, 1: 1, 2: 2}
    >>> edges = _map_edge_ids_to_indices(src, tgt, mapping, mapping)
    """
    # Find edges with valid source and target nodes
    valid_src_mask = source_ids.isin(source_mapping.keys())
    valid_dst_mask = target_ids.isin(target_mapping.keys())
    valid_edges_mask = valid_src_mask & valid_dst_mask

    # Process valid edges using vectorized operations
    valid_sources = source_ids[valid_edges_mask]
    valid_targets = target_ids[valid_edges_mask]

    # Map original node IDs to integer indices
    from_indices: np.ndarray[tuple[int, ...], np.dtype[np.int64]] = valid_sources.map(
        source_mapping,
    ).to_numpy(dtype=int)
    to_indices: np.ndarray[tuple[int, ...], np.dtype[np.int64]] = valid_targets.map(
        target_mapping,
    ).to_numpy(dtype=int)

    combined_array = np.column_stack([from_indices, to_indices]).astype(int)
    result: list[list[int]] = combined_array.tolist()
    return result


def _create_linestring_geometries(
    edge_index_array: np.ndarray[tuple[int, ...], np.dtype[np.int64]],
    src_pos: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    dst_pos: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
) -> list[LineString | None]:
    """
    Generate LineString geometries from node positions and edge connectivity.

    Creates geometric representations of edges by connecting source and target
    node coordinates. Useful for visualization and spatial analysis of networks.

    Parameters
    ----------
    edge_index_array : np.ndarray
        Array of shape (2, num_edges) with source/target indices.
    src_pos : np.ndarray
        Array of source node coordinates.
    dst_pos : np.ndarray
        Array of target node coordinates.

    Returns
    -------
    list[LineString | None]
        List of LineString objects connecting source to target nodes.
        None entries for invalid/out-of-bounds edges.

    See Also
    --------
    _create_edge_index : Create edge connectivity matrix.

    Notes
    -----
    - Performs bounds checking to avoid index errors.
    - Only uses first 2 dimensions of position data (x, y).
    - Returns None for edges with invalid node indices.

    Examples
    --------
    >>> import numpy as np
    >>> edge_index = np.array([[0, 1], [1, 2]])
    >>> src_pos = np.array([[0, 0], [1, 1]])
    >>> dst_pos = np.array([[1, 1], [2, 2]])
    >>> lines = _create_edge_geometries(edge_index, src_pos, dst_pos)
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

    Parameters
    ----------
    nodes_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing node data (index used for IDs).
    edges_gdf : geopandas.GeoDataFrame, optional
        GeoDataFrame containing edge data (MultiIndex used for relationships).
    node_feature_cols : list[str], optional
        Columns to use as node features.
    node_label_cols : list[str], optional
        Columns to use as node labels.
    edge_feature_cols : list[str], optional
        Columns to use as edge features.
    device : str or torch.device, optional
        Target device for tensor creation.
    dtype : torch.dtype, optional
        Data type for float tensors.

    Returns
    -------
    Data
        PyTorch Geometric Data object with all graph components.

    See Also
    --------
    create_heterogeneous_graph : Create multi-type graphs.

    Notes
    -----
    - Preserves original index names and values for reconstruction.
    - Stores metadata for bidirectional conversion.
    - Handles missing edges gracefully (creates empty edge tensors).
    - Maintains CRS information if available.

    Examples
    --------
    >>> import geopandas as gpd
    >>> nodes = gpd.GeoDataFrame({'feature': [1, 2]})
    >>> data = create_homogeneous_graph(nodes)
    """
    device = _get_device(device)

    # Node processing
    id_mapping, id_col_name, original_ids = _create_node_id_mapping(nodes_gdf)

    x = _create_node_features(nodes_gdf, node_feature_cols, device, dtype)
    pos = _create_node_positions(nodes_gdf, device, dtype)

    # Handle labels
    y = None
    if node_label_cols:
        y = _create_node_features(nodes_gdf, node_label_cols, device, dtype)

    # Edge processing
    edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
    edge_attr = torch.empty((0, 0), dtype=dtype or torch.float32, device=device)

    if edges_gdf is not None and not edges_gdf.empty:
        edge_pairs = _create_edge_indices(
            edges_gdf,
            id_mapping,
            id_mapping,
        )
        if edge_pairs:
            edge_index = torch.tensor(
                np.array(edge_pairs).T,
                dtype=torch.long,
                device=device,
            )
        edge_attr = _create_edge_features(edges_gdf, edge_feature_cols, device, dtype)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)

    # Store metadata
    metadata = GraphMetadata(is_hetero=False)
    metadata.node_mappings = {
        "default": {
            "mapping": id_mapping,
            "id_col": id_col_name,
            "original_ids": original_ids,
        },
    }
    metadata.node_feature_cols = node_feature_cols or []
    metadata.node_label_cols = node_label_cols or []
    metadata.edge_feature_cols = edge_feature_cols or []

    # Store index names and values for preservation
    metadata.node_index_names = nodes_gdf.index.names if hasattr(nodes_gdf.index, "names") else None
    if edges_gdf is not None and hasattr(edges_gdf.index, "names"):
        metadata.edge_index_names = edges_gdf.index.names

        # Store original edge index values for reconstruction
        metadata.edge_index_values = [
            edges_gdf.index.get_level_values(i).tolist() for i in range(edges_gdf.index.nlevels)
        ]
    else:
        metadata.edge_index_names = None
        metadata.edge_index_values = None

    # Set CRS
    if hasattr(nodes_gdf, "crs") and nodes_gdf.crs:
        metadata.crs = nodes_gdf.crs
        data.crs = metadata.crs

    data.graph_metadata = metadata
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
    """
    Build heterogeneous PyTorch Geometric HeteroData object.

    Creates a multi-type graph where nodes and edges can have different types and
    schemas. Each node type can have different features and each edge type can
    connect different node types with different relationship semantics.

    Parameters
    ----------
    nodes_dict : dict[str, geopandas.GeoDataFrame]
        Dictionary mapping node type names to their corresponding GeoDataFrames.
    edges_dict : dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        Dictionary mapping edge type tuples (source_type, relation, target_type)
        to their corresponding GeoDataFrames.
    node_feature_cols : dict[str, list[str]], optional
        Dictionary mapping node types to lists of feature column names.
    node_label_cols : dict[str, list[str]], optional
        Dictionary mapping node types to lists of label column names.
    edge_feature_cols : dict[str, list[str]], optional
        Dictionary mapping edge types to lists of feature column names.
    device : str or torch.device, optional
        Target device for tensor creation.
    dtype : torch.dtype, optional
        Data type for float tensors.

    Returns
    -------
    HeteroData
        PyTorch Geometric HeteroData object with all graph components.

    See Also
    --------
    create_homogeneous_graph : Create single-type graphs.

    Examples
    --------
    >>> import geopandas as gpd
    >>> nodes = {'buildings': gpd.GeoDataFrame({'area': [100, 200]})}
    >>> data = create_heterogeneous_graph(nodes)
    """
    device = _get_device(device)
    data = HeteroData()

    # Default empty dicts
    edges_dict = edges_dict or {}

    # Process nodes and get mappings
    node_mappings = _process_hetero_nodes(
        data,
        nodes_dict,
        node_feature_cols,
        node_label_cols,
        device,
        dtype,
    )

    # Process edges
    _process_hetero_edges(
        data,
        edges_dict,
        node_mappings,
        edge_feature_cols,
        device,
        dtype,
    )

    # Store metadata
    _store_hetero_metadata(
        data,
        node_mappings,
        nodes_dict,
        edges_dict,
        node_feature_cols,
        node_label_cols,
        edge_feature_cols,
    )

    return data


def _process_hetero_nodes(
    data: HeteroData,
    nodes_dict: dict[str, gpd.GeoDataFrame],
    node_feature_cols: dict[str, list[str]] | None,
    node_label_cols: dict[str, list[str]] | None,
    device: str | torch.device | None,
    dtype: torch.dtype | None,
) -> dict[str, dict[str, dict[str | int, int] | str | list[str | int]]]:
    """
    Process all node types for heterogeneous graph.

    Handles node processing for each node type in a heterogeneous graph, creating
    mappings, features, and labels for each type independently.

    Parameters
    ----------
    data : HeteroData
        The HeteroData object to populate with node information.
    nodes_dict : dict[str, geopandas.GeoDataFrame]
        Dictionary mapping node type names to their GeoDataFrames.
    node_feature_cols : dict[str, list[str]], optional
        Dictionary mapping node types to feature column lists.
    node_label_cols : dict[str, list[str]], optional
        Dictionary mapping node types to label column lists.
    device : str or torch.device, optional
        Target device for tensor creation.
    dtype : torch.dtype, optional
        Data type for float tensors.

    Returns
    -------
    dict[str, dict[str, dict[str | int, int] | str | list[str | int]]]
        Dictionary containing node mappings and metadata for each node type.

    See Also
    --------
    _process_hetero_edges : Process edge types for heterogeneous graphs.

    Examples
    --------
    >>> data = HeteroData()
    >>> nodes = {'buildings': gpd.GeoDataFrame()}
    >>> mappings = _process_hetero_nodes(data, nodes, None, None, 'cpu', torch.float32)
    """
    node_mappings: dict[str, dict[str, dict[str | int, int] | str | list[str | int]]] = {}
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
        data[node_type].pos = _create_node_positions(node_gdf, device, dtype)

        # Labels
        label_cols = node_label_cols.get(node_type) if node_label_cols else None
        if label_cols:
            data[node_type].y = _create_node_features(node_gdf, label_cols, device, dtype)

    return node_mappings


def _process_hetero_edges(
    data: HeteroData,
    edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    node_mappings: dict[str, dict[str, dict[str | int, int] | str | list[str | int]]],
    edge_feature_cols: dict[str, list[str]] | None,
    device: str | torch.device | None,
    dtype: torch.dtype | None,
) -> None:
    """
    Process all edge types for heterogeneous graph.

    Handles edge processing for each edge type in a heterogeneous graph, creating
    connectivity matrices and features for relationships between different node types.

    Parameters
    ----------
    data : HeteroData
        The HeteroData object to populate with edge information.
    edges_dict : dict[tuple[str, str, str], geopandas.GeoDataFrame]
        Dictionary mapping edge type tuples to their GeoDataFrames.
    node_mappings : dict[str, dict[str, dict[str | int, int] | str | list[str | int]]]
        Node mappings and metadata from node processing.
    edge_feature_cols : dict[str, list[str]], optional
        Dictionary mapping edge types to feature column lists.
    device : str or torch.device, optional
        Target device for tensor creation.
    dtype : torch.dtype, optional
        Data type for float tensors.

    See Also
    --------
    _process_hetero_nodes : Process node types for heterogeneous graphs.

    Examples
    --------
    >>> data = HeteroData()
    >>> edges = {('building', 'near', 'building'): gpd.GeoDataFrame()}
    >>> _process_hetero_edges(data, edges, node_mappings, None, 'cpu', torch.float32)
    """
    device = _get_device(device)

    for edge_type, edge_gdf in edges_dict.items():
        # Extract source, relation, and destination types from edge_type tuple
        src_type, rel_type, dst_type = edge_type

        # Get the mapping dictionaries (not the full metadata)
        # The type system guarantees these are dictionaries based on _process_hetero_nodes
        src_mapping_raw = node_mappings[src_type]["mapping"]
        dst_mapping_raw = node_mappings[dst_type]["mapping"]

        # Type assertion for mypy - these are guaranteed to be dicts by construction
        assert isinstance(src_mapping_raw, dict), f"Expected dict mapping for {src_type}"
        assert isinstance(dst_mapping_raw, dict), f"Expected dict mapping for {dst_type}"

        src_mapping: dict[str | int, int] = src_mapping_raw
        dst_mapping: dict[str | int, int] = dst_mapping_raw

        if edge_gdf is not None and not edge_gdf.empty:
            edge_pairs = _create_edge_indices(
                edge_gdf,
                src_mapping,
                dst_mapping,
            )
            edge_index = (
                torch.tensor(np.array(edge_pairs).T, dtype=torch.long, device=device)
                if edge_pairs
                else torch.zeros((2, 0), dtype=torch.long, device=device)
            )
            data[edge_type].edge_index = edge_index

            feature_cols = edge_feature_cols.get(rel_type) if edge_feature_cols else None
            data[edge_type].edge_attr = _create_edge_features(edge_gdf, feature_cols, device, dtype)
        else:
            data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
            data[edge_type].edge_attr = torch.empty(
                (0, 0),
                dtype=dtype or torch.float32,
                device=device,
            )


def _store_hetero_metadata(
    data: HeteroData,
    node_mappings: dict[str, dict[str, dict[str | int, int] | str | list[str | int]]],
    nodes_dict: dict[str, gpd.GeoDataFrame],
    edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
    node_feature_cols: dict[str, list[str]] | None,
    node_label_cols: dict[str, list[str]] | None,
    edge_feature_cols: dict[str, list[str]] | None,
) -> None:
    """
    Store metadata for heterogeneous graph.

    Stores all necessary metadata for bidirectional conversion between HeteroData
    and GeoDataFrames, including mappings, column information, and graph structure.

    Parameters
    ----------
    data : HeteroData
        The HeteroData object to store metadata in.
    node_mappings : dict[str, dict[str, dict[str | int, int] | str | list[str | int]]]
        Node mappings and metadata from node processing.
    nodes_dict : dict[str, geopandas.GeoDataFrame]
        Dictionary mapping node type names to their GeoDataFrames.
    edges_dict : dict[tuple[str, str, str], geopandas.GeoDataFrame]
        Dictionary mapping edge type tuples to their GeoDataFrames.
    node_feature_cols : dict[str, list[str]], optional
        Dictionary mapping node types to feature column lists.
    node_label_cols : dict[str, list[str]], optional
        Dictionary mapping node types to label column lists.
    edge_feature_cols : dict[str, list[str]], optional
        Dictionary mapping edge types to feature column lists.

    See Also
    --------
    _process_hetero_nodes : Process node types for heterogeneous graphs.
    _process_hetero_edges : Process edge types for heterogeneous graphs.

    Examples
    --------
    >>> data = HeteroData()
    >>> _store_hetero_metadata(data, mappings, nodes, edges, None, None, None)
    """
    # Store mappings and column metadata
    metadata = GraphMetadata(is_hetero=True)
    metadata.node_types = list(nodes_dict.keys())
    metadata.edge_types = list(edges_dict.keys())
    metadata.node_mappings = node_mappings
    metadata.node_feature_cols = node_feature_cols or {}
    metadata.node_label_cols = node_label_cols or {}
    metadata.edge_feature_cols = edge_feature_cols or {}

    # Store index names for reconstruction
    metadata.node_index_names = {}
    for node_type, node_gdf in nodes_dict.items():
        if hasattr(node_gdf.index, "names"):
            metadata.node_index_names[node_type] = node_gdf.index.names

    # Store edge index names and values for reconstruction
    metadata.edge_index_names = {}
    metadata.edge_index_values = {}
    for edge_type, edge_gdf in edges_dict.items():
        if edge_gdf is not None and hasattr(edge_gdf.index, "names"):
            # Store edge index names
            metadata.edge_index_names[edge_type] = edge_gdf.index.names

            # Store original edge index values for reconstruction
            metadata.edge_index_values[edge_type] = [
                edge_gdf.index.get_level_values(i).tolist() for i in range(edge_gdf.index.nlevels)
            ]

    # Set CRS
    crs_values = [gdf.crs for gdf in nodes_dict.values() if hasattr(gdf, "crs") and gdf.crs]
    if crs_values and all(crs == crs_values[0] for crs in crs_values):
        metadata.crs = crs_values[0]
        data.crs = metadata.crs

    data.graph_metadata = metadata


# ============================================================================
# GRAPH VALIDATION FUNCTIONS
# ============================================================================


def validate_pyg(data: Data | HeteroData) -> GraphMetadata:
    """
    Validate PyTorch Geometric Data or HeteroData objects and return metadata.

    This centralized validation function performs comprehensive validation of PyG objects,
    including type checking, metadata validation, and structural consistency checks.
    It serves as the single point of validation for all PyG objects in city2graph.

    Parameters
    ----------
    data : torch_geometric.data.Data or torch_geometric.data.HeteroData
        PyTorch Geometric data object to validate.

    Returns
    -------
    GraphMetadata
        Metadata object containing graph information for reconstruction.

    Raises
    ------
    ImportError
        If PyTorch Geometric is not installed.
    TypeError
        If data is not a valid PyTorch Geometric object.
    ValueError
        If the data object is missing required metadata or is inconsistent.

    See Also
    --------
    pyg_to_gdf : Convert PyG objects to GeoDataFrames.
    pyg_to_nx : Convert PyG objects to NetworkX graphs.

    Examples
    --------
    >>> data = create_homogeneous_graph(nodes_gdf)
    >>> metadata = _validate_pyg_data(data)
    """
    # Check PyTorch availability first
    if not TORCH_AVAILABLE:
        raise ImportError(TORCH_ERROR_MSG)

    # Comprehensive type checking for PyG objects
    if not isinstance(data, (Data, HeteroData)):
        # Provide detailed error message based on the actual type
        actual_type = type(data).__name__
        msg = (
            f"Input must be a PyTorch Geometric Data or HeteroData object, "
            f"got {actual_type}. Ensure you have PyTorch Geometric installed "
            f"and are passing a valid PyG object."
        )
        raise TypeError(msg)

    # Validate metadata presence and type
    if not hasattr(data, "graph_metadata"):
        msg = (
            "PyG object is missing 'graph_metadata' attribute. "
            "This object may not have been created by city2graph. "
            "Use city2graph.graph.gdf_to_pyg() or city2graph.graph.nx_to_pyg() "
            "to create compatible PyG objects."
        )
        raise ValueError(msg)

    if not isinstance(data.graph_metadata, GraphMetadata):
        actual_metadata_type = type(data.graph_metadata).__name__
        msg = (
            f"PyG object has 'graph_metadata' of incorrect type: {actual_metadata_type}. "
            f"Expected GraphMetadata. This object may not have been created by city2graph."
        )
        raise TypeError(msg)

    metadata = data.graph_metadata
    is_hetero = isinstance(data, HeteroData)

    # Validate consistency between PyG object type and metadata
    if is_hetero and not metadata.is_hetero:
        msg = (
            "Inconsistency detected: PyG object is HeteroData but metadata.is_hetero is False. "
            "This indicates corrupted metadata or an incorrectly constructed object."
        )
        raise ValueError(msg)
    if not is_hetero and metadata.is_hetero:
        msg = (
            "Inconsistency detected: PyG object is Data but metadata.is_hetero is True. "
            "This indicates corrupted metadata or an incorrectly constructed object."
        )
        raise ValueError(msg)

    # Additional structural validation for heterogeneous graphs
    if is_hetero:
        _validate_hetero_structure(data, metadata)
    else:
        _validate_homo_structure(data, metadata)

    return metadata


def _validate_hetero_structure(data: HeteroData, metadata: GraphMetadata) -> None:
    """
    Validate structural consistency of heterogeneous PyG data.

    Performs comprehensive validation of heterogeneous graph structure, ensuring
    that node types, edge types, and tensor dimensions are consistent.

    Parameters
    ----------
    data : HeteroData
        The heterogeneous PyTorch Geometric data object to validate.
    metadata : GraphMetadata
        Metadata containing expected graph structure information.

    See Also
    --------
    _validate_homo_structure : Validate homogeneous graph structure.

    Examples
    --------
    >>> data = create_heterogeneous_graph(nodes_dict)
    >>> metadata = data._metadata
    >>> _validate_hetero_structure(data, metadata)
    """
    # Check that node types in metadata match actual node types in data
    if metadata.node_types:
        actual_node_types = set(data.node_types)
        expected_node_types = set(metadata.node_types)
        if actual_node_types != expected_node_types:
            msg = (
                f"Node types mismatch: metadata expects {expected_node_types}, "
                f"but PyG object has {actual_node_types}"
            )
            raise ValueError(msg)

    # Check that edge types in metadata match actual edge types in data
    if metadata.edge_types:
        actual_edge_types = set(data.edge_types)
        expected_edge_types = set(metadata.edge_types)
        if actual_edge_types != expected_edge_types:
            msg = (
                f"Edge types mismatch: metadata expects {expected_edge_types}, "
                f"but PyG object has {actual_edge_types}"
            )
            raise ValueError(msg)

    # Validate tensor shape consistency for each node type
    for node_type in data.node_types:
        node_data = data[node_type]
        if hasattr(node_data, "x") and node_data.x is not None:
            num_nodes = node_data.x.size(0)

            # Check position tensor consistency
            if (
                hasattr(node_data, "pos")
                and node_data.pos is not None
                and node_data.pos.size(0) != num_nodes
            ):
                msg = (
                    f"Node type '{node_type}': position tensor size ({node_data.pos.size(0)}) "
                    f"doesn't match node feature tensor size ({num_nodes})"
                )
                raise ValueError(msg)

            # Check label tensor consistency
            if (
                hasattr(node_data, "y")
                and node_data.y is not None
                and node_data.y.size(0) != num_nodes
            ):
                msg = (
                    f"Node type '{node_type}': label tensor size ({node_data.y.size(0)}) "
                    f"doesn't match node feature tensor size ({num_nodes})"
                )
                raise ValueError(msg)


def _validate_homo_structure(data: Data, metadata: GraphMetadata) -> None:
    """
    Validate structural consistency of homogeneous PyG data.

    Performs comprehensive validation of homogeneous graph structure, ensuring
    that tensor dimensions are consistent and metadata is properly structured.

    Parameters
    ----------
    data : Data
        The homogeneous PyTorch Geometric data object to validate.
    metadata : GraphMetadata
        Metadata containing expected graph structure information.

    See Also
    --------
    _validate_hetero_structure : Validate heterogeneous graph structure.

    Examples
    --------
    >>> data = create_homogeneous_graph(nodes_gdf)
    >>> metadata = data._metadata
    >>> _validate_homo_structure(data, metadata)
    """
    # Validate that metadata has the expected structure for homogeneous graphs
    if metadata.node_types and len(metadata.node_types) > 0:
        msg = "Homogeneous graph metadata should not have node_types specified"
        raise ValueError(msg)

    if metadata.edge_types and len(metadata.edge_types) > 0:
        msg = "Homogeneous graph metadata should not have edge_types specified"
        raise ValueError(msg)

    # Validate that node mappings use the "default" key for homogeneous graphs
    if metadata.node_mappings and "default" not in metadata.node_mappings:
        msg = "Homogeneous graph metadata should use 'default' key in node_mappings"
        raise ValueError(msg)

    # Validate that feature/label columns are lists, not dicts
    if metadata.node_feature_cols and not isinstance(metadata.node_feature_cols, list):
        msg = "Homogeneous graph metadata should have node_feature_cols as list, not dict"
        raise ValueError(msg)

    if metadata.node_label_cols and not isinstance(metadata.node_label_cols, list):
        msg = "Homogeneous graph metadata should have node_label_cols as list, not dict"
        raise ValueError(msg)

    if metadata.edge_feature_cols and not isinstance(metadata.edge_feature_cols, list):
        msg = "Homogeneous graph metadata should have edge_feature_cols as list, not dict"
        raise ValueError(msg)

    # Validate tensor shape consistency
    if hasattr(data, "x") and data.x is not None:
        num_nodes = data.x.size(0)

        # Check position tensor consistency
        if hasattr(data, "pos") and data.pos is not None and data.pos.size(0) != num_nodes:
            msg = (
                f"Node position tensor size ({data.pos.size(0)}) "
                f"doesn't match node feature tensor size ({num_nodes})"
            )
            raise ValueError(msg)

        # Check label tensor consistency
        if hasattr(data, "y") and data.y is not None and data.y.size(0) != num_nodes:
            msg = (
                f"Node label tensor size ({data.y.size(0)}) "
                f"doesn't match node feature tensor size ({num_nodes})"
            )
            raise ValueError(msg)

    # Validate edge tensor consistency
    if hasattr(data, "edge_index") and data.edge_index is not None:
        num_edges = data.edge_index.size(1)

        # Check edge attribute tensor consistency
        if (
            hasattr(data, "edge_attr")
            and data.edge_attr is not None
            and data.edge_attr.size(0) != num_edges
        ):
            msg = (
                f"Edge attribute tensor size ({data.edge_attr.size(0)}) "
                f"doesn't match number of edges ({num_edges})"
            )
            raise ValueError(msg)


# ============================================================================
# GRAPH RECONSTRUCTION FUNCTIONS
# ============================================================================


def _extract_tensor_data(
    tensor: torch.Tensor | None,
    column_names: list[str] | None = None,
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
    """
    Extract data from tensor with proper column names.

    Converts PyTorch tensors to numpy arrays and maps them to column names
    for reconstruction of GeoDataFrame columns.

    Parameters
    ----------
    tensor : torch.Tensor, optional
        Input tensor containing feature data.
    column_names : list[str], optional
        List of column names to map tensor columns to.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping column names to numpy arrays.

    See Also
    --------
    _get_node_data_info : Get node data and count information.

    Examples
    --------
    >>> tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> cols = ['feature1', 'feature2']
    >>> data = _extract_tensor_data(tensor, cols)
    """
    if tensor is None or tensor.numel() == 0 or column_names is None:
        return {}

    features_array = tensor.detach().cpu().numpy()
    num_cols = min(len(column_names), features_array.shape[1])
    return {column_names[i]: features_array[:, i] for i in range(num_cols)}


def _get_node_data_info(
    data: Data | HeteroData,
    node_type: str | None,
    metadata: GraphMetadata,
) -> tuple[Data | HeteroData, int]:
    """
    Get node data and number of nodes.

    Extracts node-specific data from PyG objects, handling both homogeneous
    and heterogeneous graphs appropriately.

    Parameters
    ----------
    data : Data or HeteroData
        PyTorch Geometric data object.
    node_type : str, optional
        Node type for heterogeneous graphs.
    metadata : GraphMetadata
        Metadata containing graph structure information.

    Returns
    -------
    tuple[Data | HeteroData, int]
        Node data object and number of nodes.

    See Also
    --------
    _extract_tensor_data : Extract data from tensors.

    Examples
    --------
    >>> node_data, num_nodes = _get_node_data_info(data, 'building', metadata)
    """
    node_data = data[node_type] if metadata.is_hetero and node_type else data
    return node_data, int(node_data.num_nodes)


def _get_mapping_info(
    node_type: str | None,
    metadata: GraphMetadata,
) -> dict[str, dict[str | int, int] | str | list[str | int]] | None:
    """
    Get mapping info for the given node type.

    This function retrieves mapping information from the metadata for a specific
    node type, handling both homogeneous and heterogeneous graphs.

    Parameters
    ----------
    node_type : str or None
        The type of node to get mapping info for. If None, uses default mapping.
    metadata : GraphMetadata
        The graph metadata containing node mappings.

    Returns
    -------
    dict or None
        Dictionary containing mapping information with keys like 'original_ids',
        or None if no mapping exists for the given node type.

    See Also
    --------
    _extract_index_values : Extract index values from mapping info.

    Examples
    --------
    >>> metadata = GraphMetadata(is_hetero=True, node_mappings={'building': {...}})
    >>> mapping = _get_mapping_info('building', metadata)
    """
    mapping_key = "default" if not metadata.is_hetero or not node_type else node_type
    return metadata.node_mappings.get(mapping_key)


def _extract_index_values(
    mapping_info: dict[str, dict[str | int, int] | str | list[str | int]],
    num_nodes: int,
) -> list[str | int]:
    """
    Extract index values from mapping info.

    This function extracts the original node IDs from mapping information,
    ensuring the returned list has the correct length.

    Parameters
    ----------
    mapping_info : dict
        Dictionary containing mapping information with 'original_ids' key.
    num_nodes : int
        Number of nodes to extract IDs for.

    Returns
    -------
    list of str or int
        List of original node IDs, truncated to num_nodes length.

    See Also
    --------
    _get_mapping_info : Get mapping info for a given node type.

    Examples
    --------
    >>> mapping_info = {'original_ids': ['a', 'b', 'c', 'd']}
    >>> ids = _extract_index_values(mapping_info, 3)
    >>> print(ids)  # ['a', 'b', 'c']
    """
    original_ids = mapping_info.get("original_ids", list(range(num_nodes)))

    # Convert to list if not already, then slice to num_nodes
    ids_list = original_ids if isinstance(original_ids, list) else list(range(num_nodes))
    return ids_list[:num_nodes]


def _create_geometry_from_positions(node_data: Data | HeteroData) -> gpd.array.GeometryArray | None:
    """
    Create geometry from node positions.

    This function converts node position tensors into GeoPandas geometry objects
    for spatial analysis and visualization.

    Parameters
    ----------
    node_data : Data or HeteroData
        PyTorch Geometric data object containing node positions.

    Returns
    -------
    gpd.array.GeometryArray or None
        Array of Point geometries created from node positions, or None if
        no position data is available.

    See Also
    --------
    _extract_node_features_and_labels : Extract features and labels from node data.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> data = Data(pos=torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
    >>> geom = _create_geometry_from_positions(data)
    """
    if not hasattr(node_data, "pos") or node_data.pos is None:
        return None
    pos_array: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = (
        node_data.pos.detach().cpu().numpy()
    )
    return gpd.points_from_xy(pos_array[:, 0], pos_array[:, 1])


def _extract_node_features_and_labels(
    node_data: Data | HeteroData,
    node_type: str | None,
    metadata: GraphMetadata,
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
    """
    Extract features and labels from node data.

    This function extracts node features and labels from PyTorch Geometric data
    objects, handling both homogeneous and heterogeneous graphs.

    Parameters
    ----------
    node_data : Data or HeteroData
        PyTorch Geometric data object containing node features and labels.
    node_type : str or None
        The type of nodes to extract data for. Required for heterogeneous graphs.
    metadata : GraphMetadata
        Graph metadata containing information about feature and label mappings.

    Returns
    -------
    dict
        Dictionary mapping column names to numpy arrays containing the extracted
        features and labels.

    See Also
    --------
    _create_geometry_from_positions : Create geometry from node positions.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> data = Data(x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    >>> features = _extract_node_features_and_labels(data, None, metadata)
    """
    gdf_data = {}
    is_hetero = metadata.is_hetero

    # Extract features
    if hasattr(node_data, "x") and node_data.x is not None and metadata.node_feature_cols:
        feature_cols = metadata.node_feature_cols
        feature_cols_list: list[str] | None = None
        if is_hetero and node_type and isinstance(feature_cols, dict):
            feature_cols_list = feature_cols.get(node_type)
        elif not is_hetero and isinstance(feature_cols, list):
            feature_cols_list = feature_cols
        features_dict = _extract_tensor_data(node_data.x, feature_cols_list)
        gdf_data.update(features_dict)

    # Extract labels
    if hasattr(node_data, "y") and node_data.y is not None and metadata.node_label_cols:
        label_cols = metadata.node_label_cols
        label_cols_list: list[str] | None = None
        if is_hetero and node_type and isinstance(label_cols, dict):
            label_cols_list = label_cols.get(node_type)
        elif not is_hetero and isinstance(label_cols, list):
            label_cols_list = label_cols
        labels_dict = _extract_tensor_data(node_data.y, label_cols_list)
        gdf_data.update(labels_dict)

    return gdf_data


def _set_gdf_index_and_crs(
    gdf: gpd.GeoDataFrame,
    node_type: str | None,
    metadata: GraphMetadata,
) -> None:
    """
    Set index names and CRS on GeoDataFrame.

    This function configures the index names and coordinate reference system
    for a GeoDataFrame based on metadata information.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The GeoDataFrame to configure.
    node_type : str or None
        The type of nodes in the GeoDataFrame.
    metadata : GraphMetadata
        Graph metadata containing index names and CRS information.

    See Also
    --------
    _reconstruct_node_gdf : Reconstruct node GeoDataFrame from PyTorch data.

    Examples
    --------
    >>> import geopandas as gpd
    >>> gdf = gpd.GeoDataFrame({'col1': [1, 2]})
    >>> _set_gdf_index_and_crs(gdf, 'building', metadata)
    """
    # Set index names
    if metadata.node_index_names:
        index_names: list[str] | None = None
        # Get index names based on heterogeneity and node type
        if metadata.is_hetero and node_type and isinstance(metadata.node_index_names, dict):
            index_names = metadata.node_index_names.get(node_type)
        elif not metadata.is_hetero and isinstance(metadata.node_index_names, list):
            index_names = metadata.node_index_names

        # Set index name if available
        if (
            index_names
            and hasattr(gdf.index, "names")
            and isinstance(index_names, list)
            and len(index_names) > 0
        ):
            gdf.index.name = index_names[0]

    # Set CRS
    if metadata.crs and hasattr(gdf, "geometry") and gdf.geometry is not None:
        # Check if the geometry column is truly empty or all null
        if gdf.empty or gdf.geometry.isna().all():
            gdf.crs = metadata.crs
        else:
            # Use set_crs for non-empty geometries
            gdf.set_crs(metadata.crs, allow_override=True, inplace=True)
    # If no geometry column, we can't set CRS - skip silently


def _reconstruct_node_gdf(
    data: Data | HeteroData,
    metadata: GraphMetadata,
    node_type: str | None = None,
) -> gpd.GeoDataFrame:
    """
    Reconstruct node GeoDataFrame from PyTorch Geometric data.

    This function reconstructs a GeoDataFrame containing node information
    from PyTorch Geometric data objects and metadata.

    Parameters
    ----------
    data : Data or HeteroData
        PyTorch Geometric data object containing node information.
    metadata : GraphMetadata
        Graph metadata with mapping and feature information.
    node_type : str, optional
        The type of nodes to reconstruct. Required for heterogeneous graphs.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing reconstructed node data with geometry,
        features, and proper indexing.

    See Also
    --------
    _extract_node_features_and_labels : Extract features and labels from node data.
    _create_geometry_from_positions : Create geometry from node positions.

    Examples
    --------
    >>> from torch_geometric.data import Data
    >>> import torch
    >>> data = Data(x=torch.tensor([[1.0, 2.0]]), pos=torch.tensor([[0.0, 1.0]]))
    >>> gdf = _reconstruct_node_gdf(data, metadata)
    """
    node_data, num_nodes = _get_node_data_info(data, node_type, metadata)
    mapping_info = _get_mapping_info(node_type, metadata)

    # Extract node IDs and features/labels
    gdf_data = {}
    features_labels = _extract_node_features_and_labels(node_data, node_type, metadata)
    gdf_data.update(features_labels)

    # Create geometry
    geometry = _create_geometry_from_positions(node_data)
    index_values = _extract_index_values(mapping_info, num_nodes) if mapping_info else None

    gdf = gpd.GeoDataFrame(gdf_data, geometry=geometry, index=index_values)

    _set_gdf_index_and_crs(gdf, node_type, metadata)

    return gdf


def _reconstruct_edge_index(
    edge_type: str | tuple[str, str, str] | None,
    is_hetero: bool,
    edge_data_dict: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]],
    metadata: GraphMetadata,
) -> pd.Index | pd.MultiIndex | None:
    """
    Reconstruct edge index from stored values.

    This function reconstructs pandas Index or MultiIndex objects for edges
    from stored values in the metadata.

    Parameters
    ----------
    edge_type : str, tuple, or None
        The type of edges to reconstruct index for.
    is_hetero : bool
        Whether the graph is heterogeneous.
    edge_data_dict : dict
        Dictionary containing edge data arrays.
    metadata : GraphMetadata
        Graph metadata containing stored edge index values.

    Returns
    -------
    pd.Index, pd.MultiIndex, or None
        Reconstructed index for the edges, or None if no stored values exist.

    See Also
    --------
    _extract_edge_features : Extract edge features from data.

    Examples
    --------
    >>> edge_data = {'feature1': np.array([1, 2, 3])}
    >>> index = _reconstruct_edge_index('road', False, edge_data, metadata)
    """
    stored_values: list[list[str | int]] | None = None
    if is_hetero and edge_type and isinstance(metadata.edge_index_values, dict):
        if isinstance(edge_type, tuple):
            stored_values = metadata.edge_index_values.get(edge_type)
    elif not is_hetero and isinstance(metadata.edge_index_values, list):
        stored_values = metadata.edge_index_values

    if not stored_values:
        return None

    # Determine number of rows based on edge data or stored values
    num_rows = len(next(iter(edge_data_dict.values()))) if edge_data_dict else len(stored_values[0])

    # Handle MultiIndex case
    arrays = [stored_values[i][:num_rows] for i in range(len(stored_values))]
    return pd.MultiIndex.from_arrays(arrays)


def _extract_edge_features(
    edge_data: Data | HeteroData,
    edge_type: str | tuple[str, str, str] | None,
    is_hetero: bool,
    metadata: GraphMetadata,
) -> dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]]:
    """
    Extract edge features from edge data.

    This function extracts edge features from PyTorch Geometric data objects,
    handling both homogeneous and heterogeneous graphs.

    Parameters
    ----------
    edge_data : Data or HeteroData
        PyTorch Geometric data object containing edge information.
    edge_type : str, tuple, or None
        The type of edges to extract features for.
    is_hetero : bool
        Whether the graph is heterogeneous.
    metadata : GraphMetadata
        Graph metadata containing edge feature column information.

    Returns
    -------
    dict
        Dictionary mapping feature column names to numpy arrays containing
        the extracted edge features.

    See Also
    --------
    _create_edge_geometries : Create edge geometries from edge indices.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> data = Data(edge_attr=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    >>> features = _extract_edge_features(data, None, False, metadata)
    """
    edge_data_dict: dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32]]] = {}

    if not (hasattr(edge_data, "edge_attr") and edge_data.edge_attr is not None):
        return edge_data_dict

    feature_cols = metadata.edge_feature_cols

    # Determine column names based on graph type
    cols = None
    if is_hetero and isinstance(edge_type, tuple) and isinstance(feature_cols, dict):
        rel_type = edge_type[1]
        cols = feature_cols.get(rel_type)
    elif not is_hetero and isinstance(feature_cols, list):
        cols = feature_cols

    features_dict = _extract_tensor_data(edge_data.edge_attr, cols)
    edge_data_dict.update(features_dict)

    return edge_data_dict


def _create_edge_geometries(
    edge_data: Data,
    edge_type: str | tuple[str, str, str] | None,
    is_hetero: bool,
    data: Data | HeteroData,
) -> gpd.array.GeometryArray | None:
    """
    Create edge geometries from edge indices and node positions.

    This function creates LineString geometries for edges by connecting
    the positions of source and destination nodes.

    Parameters
    ----------
    edge_data : Data
        PyTorch Geometric data object containing edge information.
    edge_type : str, tuple, or None
        The type of edges to create geometries for.
    is_hetero : bool
        Whether the graph is heterogeneous.
    data : Data or HeteroData
        Complete PyTorch Geometric data object containing node positions.

    Returns
    -------
    gpd.array.GeometryArray or None
        Array of LineString geometries for the edges, or None if
        node positions are not available.

    See Also
    --------
    _extract_edge_features : Extract edge features from data.

    Examples
    --------
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> data = Data(edge_index=torch.tensor([[0, 1], [1, 0]]),
    ...             pos=torch.tensor([[0.0, 0.0], [1.0, 1.0]]))
    >>> geom = _create_edge_geometries(data, None, False, data)
    """
    # Get edge index array
    edge_index_array = edge_data.edge_index.detach().cpu().numpy()

    # Set default positions as None
    src_pos_array: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None = None
    dst_pos_array: np.ndarray[tuple[int, ...], np.dtype[np.float64]] | None = None

    # If hetero and specific edge type, get source and destination positions
    if is_hetero and isinstance(edge_type, tuple) and len(edge_type) == 3:
        src_type, _, dst_type = edge_type
        if hasattr(data[src_type], "pos") and data[src_type].pos is not None:
            src_pos_array = data[src_type].pos.detach().cpu().numpy()
        if hasattr(data[dst_type], "pos") and data[dst_type].pos is not None:
            dst_pos_array = data[dst_type].pos.detach().cpu().numpy()

    # If not hetero or no specific edge type, use default positions
    elif hasattr(data, "pos") and data.pos is not None:
        pos_array: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = (
            data.pos.detach().cpu().numpy()
        )
        src_pos_array = pos_array
        dst_pos_array = pos_array

    if src_pos_array is None or dst_pos_array is None:
        return None

    geometries = _create_linestring_geometries(edge_index_array, src_pos_array, dst_pos_array)
    return gpd.array.from_shapely(geometries)


def _set_edge_index_names(
    gdf: gpd.GeoDataFrame,
    edge_type: str | tuple[str, str, str] | None,
    is_hetero: bool,
    metadata: GraphMetadata,
) -> None:
    """
    Set index names on edge GeoDataFrame.

    This function configures the index names for an edge GeoDataFrame
    based on metadata information.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        The edge GeoDataFrame to configure.
    edge_type : str, tuple, or None
        The type of edges in the GeoDataFrame.
    is_hetero : bool
        Whether the graph is heterogeneous.
    metadata : GraphMetadata
        Graph metadata containing edge index name information.

    See Also
    --------
    _reconstruct_edge_gdf : Reconstruct edge GeoDataFrame from PyTorch data.

    Examples
    --------
    >>> import geopandas as gpd
    >>> gdf = gpd.GeoDataFrame({'col1': [1, 2]})
    >>> _set_edge_index_names(gdf, 'road', False, metadata)
    """
    index_names: list[str] | None = None
    if is_hetero and edge_type and isinstance(metadata.edge_index_names, dict):
        if isinstance(edge_type, tuple):
            index_names = metadata.edge_index_names.get(edge_type)
    elif not is_hetero and isinstance(metadata.edge_index_names, list):
        index_names = metadata.edge_index_names

    if (
        hasattr(gdf.index, "names")
        and isinstance(index_names, list)
        and len(index_names) > 1
        and isinstance(gdf.index, pd.MultiIndex)
    ):
        gdf.index.names = index_names


def _reconstruct_edge_gdf(
    data: Data | HeteroData,
    metadata: GraphMetadata,
    edge_type: str | tuple[str, str, str] | None = None,
) -> gpd.GeoDataFrame:
    """
    Reconstruct edge GeoDataFrame from PyTorch Geometric data.

    This function reconstructs a GeoDataFrame containing edge information
    from PyTorch Geometric data objects and metadata.

    Parameters
    ----------
    data : Data or HeteroData
        PyTorch Geometric data object containing edge information.
    metadata : GraphMetadata
        Graph metadata with mapping and feature information.
    edge_type : str, tuple, or None, optional
        The type of edges to reconstruct. Required for heterogeneous graphs.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing reconstructed edge data with geometry,
        features, and proper indexing.

    See Also
    --------
    _extract_edge_features : Extract edge features from data.
    _create_edge_geometries : Create edge geometries from edge indices.

    Examples
    --------
    >>> from torch_geometric.data import Data
    >>> import torch
    >>> data = Data(edge_index=torch.tensor([[0, 1], [1, 0]]))
    >>> gdf = _reconstruct_edge_gdf(data, metadata)
    """
    is_hetero = metadata.is_hetero

    edge_data = data[edge_type] if is_hetero and edge_type else data

    # Extract edge features
    edge_data_dict = _extract_edge_features(edge_data, edge_type, is_hetero, metadata)

    # Create geometries from edge indices and node positions
    geometry = _create_edge_geometries(edge_data, edge_type, is_hetero, data)

    # Reconstruct index from stored values
    edge_data_dict = _extract_edge_features(edge_data, edge_type, is_hetero, metadata)

    # Create geometries from edge indices and node positions
    geometry = _create_edge_geometries(edge_data, edge_type, is_hetero, data)

    # Reconstruct index from stored values
    index_values = _reconstruct_edge_index(edge_type, is_hetero, edge_data_dict, metadata)

    # Create GeoDataFrame with geometry
    if geometry is not None:
        gdf = gpd.GeoDataFrame(edge_data_dict, geometry=geometry, index=index_values)
    else:
        # If no geometry, create an empty GeoSeries for the geometry column
        # and explicitly set its CRS if metadata.crs is available.
        empty_geom = gpd.GeoSeries([], crs=metadata.crs if metadata.crs else None)
        gdf = gpd.GeoDataFrame(edge_data_dict, geometry=empty_geom, index=index_values)

    # Set index names if available
    _set_edge_index_names(gdf, edge_type, is_hetero, metadata)

    # Set CRS
    if metadata.crs:
        # Check if the geometry column is truly empty or all null
        if gdf.empty or (gdf.geometry is not None and gdf.geometry.isna().all()):
            gdf.crs = metadata.crs
        else:
            # Use set_crs for non-empty geometries
            gdf.set_crs(metadata.crs, allow_override=True, inplace=True)

    return gdf


# ============================================================================
# NETWORKX CONVERSION HELPERS
# ============================================================================


def _add_homo_nodes_to_graph(graph: nx.Graph, data: Data) -> None:
    """
    Add homogeneous nodes to NetworkX graph.

    This function adds nodes from homogeneous PyTorch Geometric data
    to a NetworkX graph with their attributes.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph to add nodes to.
    data : Data
        PyTorch Geometric data object containing node information.

    See Also
    --------
    _add_homo_edges_to_graph : Add homogeneous edges to NetworkX graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> graph = nx.Graph()
    >>> data = Data(x=torch.tensor([[1.0, 2.0]]))
    >>> _add_homo_nodes_to_graph(graph, data)
    """
    metadata = data.graph_metadata
    node_mapping_info = metadata.node_mappings.get("default", {})
    original_ids = node_mapping_info.get("original_ids", [])
    num_nodes = data.x.size(0)

    # Prepare base attributes
    attrs_df = pd.DataFrame(
        {
            "_original_index": [
                original_ids[i] if i < len(original_ids) else i for i in range(num_nodes)
            ],
        },
    )

    # Add positions using vectorized operations
    if hasattr(data, "pos") and data.pos is not None:
        pos_np: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = data.pos.detach().cpu().numpy()
        attrs_df["pos"] = [tuple(pos_np[i]) for i in range(min(num_nodes, len(pos_np)))]

    # Add features using vectorized operations
    if hasattr(data, "x") and data.x is not None:
        x_np: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = data.x.detach().cpu().numpy()
        feature_cols = metadata.node_feature_cols or [f"feat_{j}" for j in range(x_np.shape[1])]
        for j, col_name in enumerate(feature_cols[: x_np.shape[1]]):
            attrs_df[col_name] = x_np[:, j]

    # Add labels using vectorized operations
    if hasattr(data, "y") and data.y is not None:
        y_np: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = data.y.detach().cpu().numpy()
        label_cols = metadata.node_label_cols or [f"label_{j}" for j in range(y_np.shape[1])]
        for j, col_name in enumerate(label_cols[: y_np.shape[1]]):
            attrs_df[col_name] = y_np[:, j]

    # Add nodes in bulk
    graph.add_nodes_from([(i, attrs_df.iloc[i].to_dict()) for i in range(num_nodes)])


def _add_homo_edges_to_graph(graph: nx.Graph, data: Data) -> None:
    """
    Add homogeneous edges to NetworkX graph.

    This function adds edges from homogeneous PyTorch Geometric data
    to a NetworkX graph with their attributes.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph to add edges to.
    data : Data
        PyTorch Geometric data object containing edge information.

    See Also
    --------
    _add_homo_nodes_to_graph : Add homogeneous nodes to NetworkX graph.

    Examples
    --------
    >>> import networkx as nx
    >>> import torch
    >>> from torch_geometric.data import Data
    >>> graph = nx.Graph()
    >>> data = Data(edge_index=torch.tensor([[0, 1], [1, 0]]))
    >>> _add_homo_edges_to_graph(graph, data)
    """
    metadata = data.graph_metadata
    edge_feature_cols = metadata.edge_feature_cols
    original_edge_indices = metadata.edge_index_values

    edge_index = data.edge_index.detach().cpu().numpy()
    num_edges = edge_index.shape[1]

    # Initialize attributes DataFrame
    attrs_df = pd.DataFrame(index=range(num_edges))

    # Add edge attributes if available
    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        edge_attrs_np = data.edge_attr.detach().cpu().numpy()
        columns = edge_feature_cols or [f"edge_feat_{j}" for j in range(edge_attrs_np.shape[1])]
        edge_attrs_df = pd.DataFrame(edge_attrs_np, columns=columns)
        attrs_df = pd.concat([attrs_df, edge_attrs_df], axis=1)

    # Add original edge indices if available
    if original_edge_indices:
        attrs_df["_original_edge_index"] = list(zip(*original_edge_indices, strict=True))

    # Convert to list of dictionaries and add edges in bulk
    attrs_list = attrs_df.to_dict("records")
    src_nodes = edge_index[0]
    dst_nodes = edge_index[1]

    graph.add_edges_from(zip(src_nodes, dst_nodes, attrs_list, strict=True))


def _add_hetero_nodes_to_graph(graph: nx.Graph, data: HeteroData) -> dict[str, int]:
    """
    Add heterogeneous nodes to NetworkX graph and return node offsets.

    This function adds nodes from heterogeneous PyTorch Geometric data
    to a NetworkX graph and tracks node type offsets.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph to add nodes to.
    data : HeteroData
        PyTorch Geometric heterogeneous data object containing node information.

    Returns
    -------
    dict[str, int]
        Dictionary mapping node types to their starting offsets in the graph.

    See Also
    --------
    _add_hetero_edges_to_graph : Add heterogeneous edges to NetworkX graph.

    Examples
    --------
    >>> import networkx as nx
    >>> from torch_geometric.data import HeteroData
    >>> graph = nx.Graph()
    >>> data = HeteroData()
    >>> offsets = _add_hetero_nodes_to_graph(graph, data)
    """
    node_offset = {}
    current_offset = 0
    metadata = data.graph_metadata

    for node_type in metadata.node_types:
        node_offset[node_type] = current_offset
        node_data = data[node_type]
        num_nodes = node_data.num_nodes

        # Get original node IDs and prepare base attributes
        node_mapping_info = metadata.node_mappings.get(node_type, {})
        original_ids = node_mapping_info.get("original_ids", list(range(num_nodes)))
        attrs_df = pd.DataFrame(
            {
                "node_type": node_type,
                "_original_index": [
                    original_ids[i] if i < len(original_ids) else i for i in range(num_nodes)
                ],
            },
        )

        # Add positions using vectorized operations
        if hasattr(node_data, "pos") and node_data.pos is not None:
            pos_np = node_data.pos.detach().cpu().numpy()
            attrs_df["pos"] = [tuple(pos_np[i]) for i in range(min(num_nodes, len(pos_np)))]

        # Add features using vectorized operations
        if hasattr(node_data, "x") and node_data.x is not None:
            x_np = node_data.x.detach().cpu().numpy()
            # Handle the type union for node_feature_cols
            feature_cols = metadata.node_feature_cols.get(node_type) or [
                f"feat_{j}" for j in range(x_np.shape[1])
            ]
            for j, col_name in enumerate(feature_cols[: x_np.shape[1]]):
                attrs_df[col_name] = x_np[:, j]

        # Add labels using vectorized operations
        if hasattr(node_data, "y") and node_data.y is not None:
            y_np = node_data.y.detach().cpu().numpy()
            # Handle the type union for node_label_cols
            label_cols = metadata.node_label_cols.get(node_type) or [
                f"label_{j}" for j in range(y_np.shape[1])
            ]
            for j, col_name in enumerate(label_cols[: y_np.shape[1]]):
                attrs_df[col_name] = y_np[:, j]

        # Add nodes in bulk
        graph.add_nodes_from(
            [(current_offset + i, attrs_df.iloc[i].to_dict()) for i in range(num_nodes)],
        )
        current_offset += num_nodes

    return node_offset


def _add_hetero_edges_to_graph(
    graph: nx.Graph,
    data: HeteroData,
    node_offset: dict[str, int],
) -> None:
    """
    Add heterogeneous edges to NetworkX graph.

    This function adds edges from heterogeneous PyTorch Geometric data
    to a NetworkX graph using node offsets for proper indexing.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX graph to add edges to.
    data : HeteroData
        PyTorch Geometric heterogeneous data object containing edge information.
    node_offset : dict[str, int]
        Dictionary mapping node types to their starting offsets in the graph.

    See Also
    --------
    _add_hetero_nodes_to_graph : Add heterogeneous nodes to NetworkX graph.

    Examples
    --------
    >>> import networkx as nx
    >>> from torch_geometric.data import HeteroData
    >>> graph = nx.Graph()
    >>> offsets = {'building': 0, 'road': 100}
    >>> _add_hetero_edges_to_graph(graph, data, offsets)
    """
    metadata = data.graph_metadata

    for edge_type in metadata.edge_types:
        src_type, rel_type, dst_type = edge_type
        edge_store = data[edge_type]

        edge_index = edge_store.edge_index.detach().cpu().numpy()
        num_edges = edge_index.shape[1]

        # Create attributes DataFrame using helper functions
        attrs_df = _create_edge_attrs_dataframe(
            edge_store,
            metadata,
            rel_type,
            edge_type,
            num_edges,
        )

        # Add relation type and convert to records
        attrs_df["edge_type"] = rel_type
        attrs_list = attrs_df.to_dict("records")

        # Add edges with offset adjustments
        src_nodes = edge_index[0] + node_offset[src_type]
        dst_nodes = edge_index[1] + node_offset[dst_type]

        graph.add_edges_from(zip(src_nodes, dst_nodes, attrs_list, strict=True))


def _create_edge_attrs_dataframe(
    edge_store: Data,
    metadata: GraphMetadata,
    rel_type: str,
    edge_type: tuple[str, str, str],
    num_edges: int,
) -> pd.DataFrame:
    """
    Create edge attributes DataFrame with features and original indices.

    This function extracts edge attributes from a PyTorch Geometric edge store
    and creates a pandas DataFrame with feature columns and original edge indices.

    Parameters
    ----------
    edge_store : Data
        PyTorch Geometric Data object containing edge information.
    metadata : GraphMetadata
        Metadata object containing graph structure information.
    rel_type : str
        Relationship type identifier for the edges.
    edge_type : tuple[str, str, str]
        Tuple specifying the edge type (source_type, relation, target_type).
    num_edges : int
        Number of edges in the edge store.

    Returns
    -------
    pd.DataFrame
        DataFrame containing edge attributes and original indices.

    See Also
    --------
    _get_edge_attrs_array : Extract edge attributes array from edge store.
    _get_edge_feature_columns : Get feature column names.

    Examples
    --------
    >>> edge_store = Data(edge_attr=torch.randn(100, 5))
    >>> metadata = GraphMetadata(...)
    >>> df = _create_edge_attrs_dataframe(edge_store, metadata, "connects",
    ...                                   ("node", "connects", "node"), 100)
    """
    # Start with base DataFrame
    attrs_df = pd.DataFrame(index=range(num_edges))

    # Add edge features if available
    edge_attrs_array = _get_edge_attrs_array(edge_store)
    if edge_attrs_array is not None:
        feature_columns = _get_edge_feature_columns(metadata, rel_type, edge_attrs_array.shape[1])
        feature_df = pd.DataFrame(edge_attrs_array, columns=feature_columns)
        attrs_df = pd.concat([attrs_df, feature_df], axis=1)

    # Add original edge indices if available
    original_indices = None
    if isinstance(metadata.edge_index_values, dict):
        original_indices = metadata.edge_index_values.get(edge_type)
    if original_indices:
        attrs_df["_original_edge_index"] = list(zip(*original_indices, strict=True))

    return attrs_df


def _get_edge_attrs_array(
    edge_store: Data,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float32]] | None:
    """
    Extract edge attributes array from edge store, or None if not available.

    This function safely extracts the edge attribute tensor from a PyTorch Geometric
    Data object and converts it to a NumPy array. Returns None if no edge attributes
    are present.

    Parameters
    ----------
    edge_store : Data
        PyTorch Geometric Data object that may contain edge attributes.

    Returns
    -------
    np.ndarray or None
        Edge attributes as a NumPy array of shape (num_edges, num_features),
        or None if no edge attributes are available.

    See Also
    --------
    _create_edge_attrs_dataframe : Create edge attributes DataFrame.

    Examples
    --------
    >>> edge_store = Data(edge_attr=torch.randn(100, 5))
    >>> attrs = _get_edge_attrs_array(edge_store)
    >>> attrs.shape
    (100, 5)
    """
    return (
        edge_store.edge_attr.detach().cpu().numpy()
        if hasattr(edge_store, "edge_attr") and edge_store.edge_attr is not None
        else None
    )


def _get_edge_feature_columns(
    metadata: GraphMetadata,
    rel_type: str,
    num_features: int,
) -> list[str]:
    """
    Get feature column names, using metadata or generating defaults.

    This function retrieves edge feature column names from metadata if available,
    or generates default column names based on the number of features.

    Parameters
    ----------
    metadata : GraphMetadata
        Metadata object containing graph structure information.
    rel_type : str
        Relationship type identifier for the edges.
    num_features : int
        Number of edge features.

    Returns
    -------
    list[str]
        List of column names for edge features.

    See Also
    --------
    _create_edge_attrs_dataframe : Create edge attributes DataFrame.

    Examples
    --------
    >>> metadata = GraphMetadata(...)
    >>> cols = _get_edge_feature_columns(metadata, "connects", 5)
    >>> cols
    ['edge_feat_0', 'edge_feat_1', 'edge_feat_2', 'edge_feat_3', 'edge_feat_4']
    """
    feature_cols = None
    if isinstance(metadata.edge_feature_cols, dict):
        feature_cols = metadata.edge_feature_cols.get(rel_type)
    # For heterogeneous graphs, edge_feature_cols should be dict or None, not list
    # If it's a list, we ignore it as it indicates homogeneous usage
    return feature_cols or [f"edge_feat_{j}" for j in range(num_features)]


def _convert_homo_pyg_to_nx(data: Data, metadata: GraphMetadata) -> nx.Graph:
    """
    Convert homogeneous PyG data to NetworkX graph.

    This function converts a homogeneous PyTorch Geometric Data object to a
    NetworkX Graph, preserving node and edge attributes along with metadata.

    Parameters
    ----------
    data : Data
        Homogeneous PyTorch Geometric Data object to convert.
    metadata : GraphMetadata
        Metadata object containing graph structure information.

    Returns
    -------
    nx.Graph
        NetworkX graph with nodes, edges, and attributes from the PyG data.

    See Also
    --------
    _convert_hetero_pyg_to_nx : Convert heterogeneous PyG data to NetworkX.
    _add_homo_nodes_to_graph : Add homogeneous nodes to graph.
    _add_homo_edges_to_graph : Add homogeneous edges to graph.

    Examples
    --------
    >>> data = Data(x=torch.randn(100, 10), edge_index=torch.randint(0, 100, (2, 200)))
    >>> metadata = GraphMetadata(...)
    >>> graph = _convert_homo_pyg_to_nx(data, metadata)
    >>> len(graph.nodes)
    100
    """
    graph = nx.Graph()

    # Add metadata
    graph.graph["crs"] = metadata.crs
    graph.graph["is_hetero"] = False

    # Add nodes and edges
    _add_homo_nodes_to_graph(graph, data)
    _add_homo_edges_to_graph(graph, data)

    # Store index information for reconstruction
    graph.graph["node_index_names"] = metadata.node_index_names
    graph.graph["edge_index_names"] = metadata.edge_index_names

    return graph


def _convert_hetero_pyg_to_nx(data: HeteroData, metadata: GraphMetadata) -> nx.Graph:
    """
    Convert heterogeneous PyG data to NetworkX graph.

    This function converts a heterogeneous PyTorch Geometric HeteroData object to a
    NetworkX Graph, flattening the heterogeneous structure while preserving node
    and edge attributes along with type information.

    Parameters
    ----------
    data : HeteroData
        Heterogeneous PyTorch Geometric HeteroData object to convert.
    metadata : GraphMetadata
        Metadata object containing graph structure information.

    Returns
    -------
    nx.Graph
        NetworkX graph with nodes, edges, and attributes from the hetero PyG data.

    See Also
    --------
    _convert_homo_pyg_to_nx : Convert homogeneous PyG data to NetworkX.
    _add_hetero_nodes_to_graph : Add heterogeneous nodes to graph.
    _add_hetero_edges_to_graph : Add heterogeneous edges to graph.

    Examples
    --------
    >>> data = HeteroData()
    >>> data['node'].x = torch.randn(100, 10)
    >>> data['edge'].x = torch.randn(50, 5)
    >>> metadata = GraphMetadata(...)
    >>> graph = _convert_hetero_pyg_to_nx(data, metadata)
    >>> graph.graph['is_hetero']
    True
    """
    graph = nx.Graph()

    # Add metadata
    graph.graph["crs"] = metadata.crs
    graph.graph["is_hetero"] = True
    graph.graph["node_types"] = metadata.node_types
    graph.graph["edge_types"] = metadata.edge_types

    # Store metadata for reconstruction
    graph.graph["metadata"] = metadata

    # Add nodes and edges
    node_offset = _add_hetero_nodes_to_graph(graph, data)
    _add_hetero_edges_to_graph(graph, data, node_offset)
    graph.graph["node_offset"] = node_offset

    return graph
