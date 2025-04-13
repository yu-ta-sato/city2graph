"""
Module for creating heterogeneous graph representations of urban environments.
Converts geodataframes containing spatial data into PyTorch Geometric HeteroData objects.
"""

try:
    import torch
    import torch.nn as nn
    from torch_geometric.data import HeteroData, Data
    from torch_geometric.utils import to_networkx as pyg_to_networkx
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create placeholder classes to prevent import errors
    class HeteroData:
        pass
    
    class Data:
        pass

import numpy as np
import geopandas as gpd
import warnings
from typing import Optional, Union, List, Tuple, Dict
import networkx as nx

# Define the public API for this module
__all__ = [
    "homogeneous_graph",
    "heterogeneous_graph",
    "from_morphological_network",
    "to_networkx",
    "is_torch_available",
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


def _get_device(device: Optional[Union[str, 'torch.device']] = None) -> Union['torch.device', str]:
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
        raise ImportError(
            "PyTorch and PyTorch Geometric are required for this function. "
            "Please install them using: poetry install --with torch or "
            "pip install city2graph[torch]"
        )
    
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, torch.device):
        return device
    elif device in ["cuda", "cpu"]:
        return torch.device(device)
    else:
        raise ValueError("Device must be 'cuda', 'cpu', a torch.device object, or None")


def _detect_edge_columns(
    edge_gdf: gpd.GeoDataFrame,
    id_col: Optional[str] = None,
    source_hint: Optional[List[str]] = None,
    target_hint: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str]]:
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
    elif len(edge_gdf.columns) >= 3 and "geometry" in edge_gdf.columns[0]:
        return edge_gdf.columns[1], edge_gdf.columns[2]

    return None, None


def _get_edge_columns(
    edge_gdf: gpd.GeoDataFrame,
    source_col: Optional[str],
    target_col: Optional[str],
    source_mapping: Dict[str, int],
    target_mapping: Dict[str, int],
    id_col: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Consolidate logic for detecting or confirming source/target columns.
    """
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


def _extract_node_id_mapping(
    node_gdf: gpd.GeoDataFrame, id_col: Optional[str] = None
) -> Tuple[Dict[str, int], str]:
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
        raise ValueError(
            f"Provided id_col '{id_col}' not found in node GeoDataFrame columns."
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
def _create_node_features(
    node_gdf: gpd.GeoDataFrame,
    feature_cols: Optional[List[str]] = None,
    device: Optional[Union[str, 'torch.device']] = None,
) -> 'torch.Tensor':
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
        # Return empty feature tensor if no valid columns
        return torch.zeros((len(node_gdf), 0), dtype=torch.float, device=device)

    else:
        # Check which columns actually exist
        valid_cols = [col for col in feature_cols if col in node_gdf.columns]

        if valid_cols:
            # Convert to tensor and move to appropriate device
            return torch.tensor(
                node_gdf[valid_cols].values, dtype=torch.float, device=device
            )


def _create_edge_features(
    edge_gdf: gpd.GeoDataFrame,
    feature_cols: Optional[List[str]] = None,
    device: Optional[Union[str, 'torch.device']] = None,
) -> 'torch.Tensor':
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
    valid_cols = [col for col in feature_cols if col in edge_gdf.columns]
    if not valid_cols:
        return torch.empty((edge_gdf.shape[0], 0), dtype=torch.float, device=device)
    return torch.tensor(edge_gdf[valid_cols].values, dtype=torch.float, device=device)


def _map_edge_strings(
    edge_gdf: gpd.GeoDataFrame, source_col: str, target_col: str
) -> gpd.GeoDataFrame:
    """
    Convert source/target columns to string once for vectorized lookups.
    """
    edge_gdf[f"__{source_col}_str"] = edge_gdf[source_col].astype(str)
    edge_gdf[f"__{target_col}_str"] = edge_gdf[target_col].astype(str)
    return edge_gdf


def _create_edge_idx_pairs(
    edge_gdf: gpd.GeoDataFrame,
    source_mapping: Dict[str, int],
    target_mapping: Optional[Dict[str, int]] = None,
    source_col: Optional[str] = None,
    target_col: Optional[str] = None,
) -> List[List[int]]:
    """
    Process edges to create edge indices.

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
        edge_gdf, source_col, target_col, source_mapping, target_mapping
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
    edge_gdf = _map_edge_strings(edge_gdf, source_col, target_col)

    # Vectorized approach for valid edges
    valid_src_mask = edge_gdf[f"__{source_col}_str"].isin(source_mapping)
    valid_dst_mask = edge_gdf[f"__{target_col}_str"].isin(target_mapping)
    valid_edges_mask = valid_src_mask & valid_dst_mask

    # Count missing IDs
    missing_src_count = (~valid_src_mask).sum()
    missing_dst_count = (~valid_dst_mask).sum()

    # Process valid edges
    valid_edges = edge_gdf[valid_edges_mask]
    edge_count = len(valid_edges)

    if edge_count == 0 and (missing_src_count > 0 or missing_dst_count > 0):
        warnings.warn(
            f"No valid edges were found with columns {source_col}, {target_col}",
            category=UserWarning,
            stacklevel=2,
        )

    if not valid_edges.empty:
        # Map IDs to indices
        from_indices = valid_edges[f"__{source_col}_str"].map(source_mapping).values
        to_indices = valid_edges[f"__{target_col}_str"].map(target_mapping).values

        # Create edge list
        edges = np.column_stack([from_indices, to_indices]).tolist()
        return edges

    return []


# New helper to check if an edge GeoDataFrame is valid
def _is_valid_edge_df(edge_gdf: Optional[gpd.GeoDataFrame]) -> bool:
    return edge_gdf is not None and not edge_gdf.empty


# Remove the is_hetero parameter and always expect dictionaries for nodes and edges.
# Updated _build_graph_data to accept node_y_attribute_cols parameter
# Modified _build_graph_data: renamed parameters and references
def _build_graph_data(
    nodes: Dict[str, gpd.GeoDataFrame],
    edges: Dict[Tuple[str, str, str], gpd.GeoDataFrame],
    node_id_cols: Dict[str, str],
    node_feature_cols: Dict[str, List[str]],
    node_label_cols: Optional[Dict[str, List[str]]],
    edge_source_cols: Dict[Tuple[str, str, str], str],
    edge_target_cols: Dict[Tuple[str, str, str], str],
    edge_feature_cols: Optional[Dict[Tuple[str, str, str], List[str]]],
    device: Optional[Union[str, 'torch.device']],
) -> HeteroData:
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
        # ...existing node ID mapping and feature creation...
        id_col = node_id_cols.get(node_type)
        id_mapping, actual_id_col = _extract_node_id_mapping(node_gdf, id_col)
        node_id_mappings[node_type] = {"mapping": id_mapping, "id_col": actual_id_col}
        # Update: use feature columns instead of attribute_cols
        feature_cols = node_feature_cols.get(node_type)
        data[node_type].x = _create_node_features(node_gdf, feature_cols, device)

        # Add positional attributes if geometry is present
        if "geometry" in node_gdf.columns:
            pos = torch.tensor(
                np.array(
                    [
                        (
                            [geom.x, geom.y]
                            if hasattr(geom, "x")
                            else [geom.centroid.x, geom.centroid.y]
                        )
                        for geom in node_gdf.geometry
                    ]
                ),
                dtype=torch.float,
                device=device,
            )
            data[node_type].pos = pos

        # Update: use label columns instead of y_attribute_cols
        if node_label_cols and node_label_cols.get(node_type):
            data[node_type].y = _create_node_features(
                node_gdf, node_label_cols[node_type], device
            )
        elif "y" in node_gdf.columns:
            data[node_type].y = torch.tensor(
                node_gdf["y"].values, dtype=torch.float, device=device
            )

    # Process edges across types
    for edge_type, edge_gdf in edges.items():
        if not isinstance(edge_type, tuple) or len(edge_type) != 3:
            warnings.warn(
                f"Edge type key must be a tuple of (source_type, relation_type, target_type). Got {edge_type} instead. Skipping.",
                category=UserWarning,
                stacklevel=2,
            )
            continue

        src_type, rel_type, dst_type = edge_type

        if src_type not in node_id_mappings or dst_type not in node_id_mappings:
            warnings.warn(
                f"Edge type {edge_type} references node type(s) not present in nodes. Skipping.",
                category=UserWarning,
                stacklevel=2,
            )
            continue
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
                    np.array(pairs).T, dtype=torch.long, device=device
                )
            else:
                data[src_type, rel_type, dst_type].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long, device=device
                )
            feature_cols = (
                edge_feature_cols.get(edge_type) if edge_feature_cols else None
            )
            data[src_type, rel_type, dst_type].edge_attr = _create_edge_features(
                edge_gdf, feature_cols, device
            )
        else:
            data[src_type, rel_type, dst_type].edge_index = torch.zeros(
                (2, 0), dtype=torch.long, device=device
            )
            data[src_type, rel_type, dst_type].edge_attr = torch.empty(
                (0, 0), dtype=torch.float, device=device
            )

    # Set CRS metadata from node GeoDataFrames more efficiently
    crs_values = [gdf.crs for gdf in nodes.values() if hasattr(gdf, "crs") and gdf.crs]

    if not crs_values:
        data.crs = {}
    elif all(crs == crs_values[0] for crs in crs_values):
        data.crs = crs_values[0]
    else:
        raise ValueError("CRS mismatch among node GeoDataFrames.")

    return data


def homogeneous_graph(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: Optional[gpd.GeoDataFrame] = None,
    node_id_col: Optional[str] = None,
    node_feature_cols: Optional[List[str]] = None,
    node_label_cols: Optional[List[str]] = None,
    edge_source_col: Optional[str] = None,
    edge_target_col: Optional[str] = None,
    edge_feature_cols: Optional[List[str]] = None,
    device: Optional[Union[str, 'torch.device']] = None,
) -> Data:
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
        raise ImportError(
            "PyTorch and PyTorch Geometric are required for this function. "
            "Please install them using: poetry install --with torch or "
            "pip install city2graph[torch]"
        )
    
    # Preprocess homogeneous graph inputs into dictionaries.
    nodes_dict = {"node": nodes_gdf}
    edges_dict = {}
    if edges_gdf is not None and not edges_gdf.empty:
        edges_dict[("node", "edge", "node")] = edges_gdf
    node_id_cols = {"node": node_id_col} if node_id_col else {}
    node_feature_cols = {"node": node_feature_cols} if node_feature_cols else {}
    node_label_cols = {"node": node_label_cols} if node_label_cols else None
    edge_source_cols = (
        {("node", "edge", "node"): edge_source_col} if edge_source_col else {}
    )
    edge_target_cols = (
        {("node", "edge", "node"): edge_target_col} if edge_target_col else {}
    )

    hetero_data = _build_graph_data(
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

    data = Data(
        x=hetero_data["node"].x,
        edge_index=hetero_data[("node", "edge", "node")].edge_index,
        edge_attr=hetero_data[("node", "edge", "node")].edge_attr,
        pos=hetero_data["node"].get("pos", None),
    )
    # Assign "y" node attribute if exists
    data.y = hetero_data["node"].get("y", None)
    # Assign CRS metadata from hetero_data.
    data.crs = hetero_data.crs
    return data


def heterogeneous_graph(
    nodes_dict: Dict[str, gpd.GeoDataFrame],
    edges_dict: Dict[Tuple[str, str, str], gpd.GeoDataFrame],
    node_id_cols: Optional[Dict[str, str]] = None,
    node_feature_cols: Optional[Dict[str, List[str]]] = None,
    node_label_cols: Optional[Dict[str, List[str]]] = None,
    edge_source_cols: Optional[Dict[Tuple[str, str, str], str]] = None,
    edge_target_cols: Optional[Dict[Tuple[str, str, str], str]] = None,
    edge_feature_cols: Optional[Dict[Tuple[str, str, str], List[str]]] = None,
    device: Optional[Union[str, 'torch.device']] = None,
) -> HeteroData:
    """
    Create a heterogeneous graph HeteroData object from node and edge dictionaries.

    Parameters
    ----------
    nodes_dict : dict
        Dictionary of GeoDataFrames for each node type.
    edges_dict : dict
        Dictionary of GeoDataFrames for each edge type, with keys as (source_type, relation, target_type).
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
        raise ImportError(
            "PyTorch and PyTorch Geometric are required for this function. "
            "Please install them using: poetry install --with torch or "
            "pip install city2graph[torch]"
        )
    
    if node_id_cols is None:
        node_id_cols = {}
    if node_feature_cols is None:
        node_feature_cols = {}
    if edge_source_cols is None:
        edge_source_cols = {}
    if edge_target_cols is None:
        edge_target_cols = {}

    data = _build_graph_data(
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

    return data


def from_morphological_network(
    network_output: Dict,
    private_id_col: str = 'tess_id',
    public_id_col: str = 'id',
    private_node_feature_cols: Optional[List[str]] = None,
    public_node_feature_cols: Optional[List[str]] = None,
    device: Optional[Union[str, 'torch.device']] = None,
) -> Union[HeteroData, Data]:
    """
    Create a graph representation from the output of create_morphological_network.

    Parameters
    ----------
    network_output : dict
        Output dictionary from create_morphological_network containing:
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
        raise ImportError(
            "PyTorch and PyTorch Geometric are required for this function. "
            "Please install them using: poetry install --with torch or "
            "pip install city2graph[torch]"
        )
    
    # Validate device
    device = _get_device(device)
    
    # Extract data from network_output
    if not isinstance(network_output, dict):
        raise ValueError("network_output must be a dictionary returned from create_morphological_network")
    
    # Extract GeoDataFrames from the network output
    private_gdf = network_output.get('tessellations')
    public_gdf = network_output.get('segments')
    private_to_private_gdf = network_output.get('private_to_private')
    public_to_public_gdf = network_output.get('public_to_public')
    private_to_public_gdf = network_output.get('private_to_public')
    
    # Validate that required data exists
    has_private = private_gdf is not None and not private_gdf.empty
    has_public = public_gdf is not None and not public_gdf.empty
    
    # Case 1: We have both private and public nodes - create heterogeneous graph
    if has_private and has_public:
        # Create nodes dictionary
        nodes_dict = {
            "private": private_gdf,
            "public": public_gdf
        }

        # Create edges dictionary with edge type tuples
        edges_dict = {
            ("private", "connects_to", "private"): private_to_private_gdf,
            ("private", "adjacent_to", "public"): private_to_public_gdf,
            ("public", "connects_to", "public"): public_to_public_gdf,
        }

        # Create node ID columns dictionary
        node_id_cols = {
            "private": private_id_col,
            "public": public_id_col
        }

        # Create node feature columns dictionary
        node_feature_cols = {}
        if private_node_feature_cols is not None:
            node_feature_cols["private"] = private_node_feature_cols
        if public_node_feature_cols is not None:
            node_feature_cols["public"] = public_node_feature_cols

        # Prepare edge source/target column mappings based on morphological_network.py function output columns
        edge_source_cols = {
            ("private", "connects_to", "private"): "from_private_id",
            ("private", "adjacent_to", "public"): "private_id",
            ("public", "connects_to", "public"): "from_public_id",
        }

        edge_target_cols = {
            ("private", "connects_to", "private"): "to_private_id",
            ("private", "adjacent_to", "public"): "public_id",
            ("public", "connects_to", "public"): "to_public_id",
        }

        # Create the heterogeneous graph
        return heterogeneous_graph(
            nodes_dict=nodes_dict,
            edges_dict=edges_dict,
            node_id_cols=node_id_cols,
            node_feature_cols=node_feature_cols,
            node_label_cols=None,
            edge_source_cols=edge_source_cols,
            edge_target_cols=edge_target_cols,
            edge_feature_cols=None,
            device=device,
        )

    # Case 2: We only have private nodes - create homogeneous graph
    elif has_private:
        return homogeneous_graph(
            nodes_gdf=private_gdf,
            edges_gdf=private_to_private_gdf,
            node_id_col=private_id_col,
            node_feature_cols=private_node_feature_cols,
            node_label_cols=None,
            edge_source_col="from_private_id",
            edge_target_col="to_private_id",
            edge_feature_cols=None,
            device=device,
        )

    # Case 3: We only have public nodes - create homogeneous graph
    elif has_public:
        return homogeneous_graph(
            nodes_gdf=public_gdf,
            edges_gdf=public_to_public_gdf,
            node_id_col=public_id_col,
            node_feature_cols=public_node_feature_cols,
            node_label_cols=None,
            edge_source_col="from_public_id",
            edge_target_col="to_public_id",
            edge_feature_cols=None,
            device=device,
        )

    # Case 4: No valid nodes - raise an error to prevent unintended empty graphs.
    else:
        raise ValueError(
            "No valid node data provided. Cannot create a graph without nodes."
        )


def to_networkx(graph: Union[Data, HeteroData]) -> nx.Graph:
    """
    Convert PyTorch Geometric Data or HeteroData to a NetworkX graph.
    
    Parameters
    ----------
    graph : Union[Data, HeteroData]
        The PyTorch Geometric graph to convert
    
    Returns
    -------
    nx.Graph
        A NetworkX graph representation
    """
    # Custom behavior for HeteroData
    if hasattr(graph, "node_types"):
        nx_graph = nx.MultiDiGraph()
        # Add nodes from each node type
        for ntype in graph.node_types:
            num_nodes = graph[ntype].x.size(0) if "x" in graph[ntype] else 0
            for i in range(num_nodes):
                node_id = f"{ntype}_{i}"
                node_attr = {"node_type": ntype}
                if "x" in graph[ntype]:
                    node_attr["x"] = graph[ntype].x[i].tolist()
                if "pos" in graph[ntype]:
                    node_attr["pos"] = graph[ntype].pos[i].tolist()
                nx_graph.add_node(node_id, **node_attr)
        # Add edges for each edge type
        for edge_type in graph.edge_types:
            src_type, rel_type, dst_type = edge_type
            edge_data = graph[edge_type]
            edge_index = edge_data.edge_index
            num_edges = edge_index.size(1)
            for j in range(num_edges):
                src = f"{src_type}_{int(edge_index[0, j])}"
                dst = f"{dst_type}_{int(edge_index[1, j])}"
                attr = {"relation": rel_type}
                if "edge_attr" in edge_data:
                    attr["edge_attr"] = edge_data.edge_attr[j].tolist()
                nx_graph.add_edge(src, dst, **attr)
        # Preserve global attributes if present
        if "crs" in graph:
            nx_graph.graph["crs"] = graph["crs"]
        return nx_graph
    # Default behavior for Data
    else:
        nx_graph = pyg_to_networkx(
            graph,
            node_attrs=["x", "pos"],
            edge_attrs=["edge_attr"],
            graph_attrs=["crs"],
        )
        return nx_graph
