"""
Core Utilities Module.

This module provides essential utilities for graph conversion, data validation,
and spatial analysis operations. It serves as the foundation for the city2graph
package, offering robust data structures and conversion functions that enable
seamless integration between different graph representations and geospatial
data formats.

Key Features
------------
- Bidirectional conversion between GeoDataFrames and NetworkX graphs
- Comprehensive data validation and type checking
- Support for both homogeneous and heterogeneous graph structures
- Spatial analysis utilities (tessellation, isochrones, filtering)
- Robust metadata preservation across conversions
- Integration with multiple geospatial libraries

Main Functions
--------------
gdf_to_nx : Convert GeoDataFrames to NetworkX graphs
nx_to_gdf : Convert NetworkX graphs to GeoDataFrames
segments_to_graph : Convert line segments to graph representation
dual_graph : Create dual graph representations
filter_graph_by_distance : Spatial filtering based on network distance
create_isochrone : Generate accessibility isochrones
create_tessellation : Create spatial tessellations
validate_gdf : Validate GeoDataFrame inputs
validate_nx : Validate NetworkX graph inputs

Core Classes
------------
GraphMetadata : Centralized graph metadata management
GeoDataProcessor : Common GeoDataFrame processing operations
GraphConverter : Unified graph conversion engine
GraphAnalyzer : Graph analysis and filtering operations

See Also
--------
city2graph.graph : PyTorch Geometric integration utilities
city2graph.morphology : Urban morphology analysis functions
city2graph.proximity : Spatial proximity analysis functions
"""

# Standard library imports
import logging

# Third-party imports
import geopandas as gpd
import momepy
import networkx as nx
import pandas as pd
from shapely.geometry import LineString
from shapely.geometry import Point

# Public API definition
__all__ = [
    "create_isochrone",
    "create_tessellation",
    "dual_graph",
    "filter_graph_by_distance",
    "gdf_to_nx",
    "nx_to_gdf",
    "segments_to_graph",
    "validate_gdf",
    "validate_nx",
]

# Module logger configuration
logger = logging.getLogger(__name__)

# =============================================================================
# CORE DATA STRUCTURES AND VALIDATION
# =============================================================================

class GraphMetadata:
    """Centralized graph metadata management."""

    def __init__(self, crs: str | int | dict[str, object] | object | None = None, is_hetero: bool = False) -> None:
        # Core metadata
        self.crs = crs
        self.is_hetero = is_hetero

        # Graph structure metadata
        self.node_types: list[str] = []
        self.edge_types: list[tuple[str, str, str]] = []

        # Index management
        self.node_index_names: dict[str, list[str] | None] | list[str] | None = None
        self.edge_index_names: dict[tuple[str, str, str], list[str] | None] | list[str] | None = None

        # Geometry column tracking
        self.node_geom_cols: list[str] = []
        self.edge_geom_cols: list[str] = []

        # PyTorch Geometric specific metadata
        self.node_mappings: dict[str, dict[str, dict[str | int, int] | str | list[str | int]]] = {}
        self.node_feature_cols: dict[str, list[str]] | list[str] | None = None
        self.node_label_cols: dict[str, list[str]] | list[str] | None = None
        self.edge_feature_cols: dict[str, list[str]] | list[str] | None = None
        self.edge_index_values: dict[tuple[str, str, str], list[list[str | int]]] | list[list[str | int]] | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for NetworkX graph metadata."""
        return self.__dict__

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "GraphMetadata":
        """Create from dictionary."""
        crs = data.get("crs")
        is_hetero_obj = data.get("is_hetero", False)

        # Type check the parameters
        if crs is not None and not isinstance(crs, (str, int, dict)) and not hasattr(crs, "to_wkt"):
            msg = "CRS must be str, int, dict, a CRS-like object, or None"
            raise TypeError(msg)
        if not isinstance(is_hetero_obj, bool):
            msg = "is_hetero must be bool"
            raise TypeError(msg)
        is_hetero: bool = is_hetero_obj

        metadata = cls(crs, is_hetero)
        for key, value in data.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        return metadata

class GeoDataProcessor:
    """Common processor for GeoDataFrame operations."""

    @staticmethod
    def validate_gdf(
        gdf: gpd.GeoDataFrame | None,
        expected_geom_types: list[str] | None = None,
        allow_empty: bool = True,
    ) -> gpd.GeoDataFrame | None:
        """Unified GeoDataFrame validation."""
        if gdf is None:
            return None

        if not isinstance(gdf, gpd.GeoDataFrame):
            msg = "Input must be a GeoDataFrame"
            raise TypeError(msg)

        if gdf.empty and not allow_empty:
            msg = "GeoDataFrame cannot be empty"
            raise ValueError(msg)

        if gdf.empty:
            return gdf

        # Validate geometry types
        if expected_geom_types:
            valid_mask = gdf.geometry.geom_type.isin(expected_geom_types)
            if not valid_mask.all():
                invalid_count = (~valid_mask).sum()
                logger.warning("Removed %d geometries with invalid types", invalid_count)
                gdf = gdf[valid_mask]

        # Remove invalid geometries
        invalid_mask = gdf.geometry.isna() | ~gdf.geometry.is_valid | gdf.geometry.is_empty
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            logger.warning("Removed %d invalid geometries", invalid_count)
            gdf = gdf[~invalid_mask]

        if gdf.empty and not allow_empty:
            msg = "GeoDataFrame cannot be empty"
            raise ValueError(msg)

        return gdf

    @staticmethod
    def validate_nx(graph: nx.Graph | nx.MultiGraph) -> None:
        """Validate a NetworkX graph.

        Checks if the input is a NetworkX graph, ensures it is not empty,
        and verifies that it contains the necessary metadata for conversion
        back to GeoDataFrames or PyG objects.

        Parameters
        ----------
        graph : networkx.Graph or networkx.MultiGraph
            The NetworkX graph to validate.

        Raises
        ------
        TypeError
            If the input is not a NetworkX graph.
        ValueError
            If the graph has no nodes, no edges, or is missing essential metadata.
        """
        if graph.number_of_nodes() == 0:
            msg = "Graph has no nodes"
            raise ValueError(msg)
        if graph.number_of_edges() == 0:
            msg = "Graph has no edges"
            raise ValueError(msg)

        # Check for essential graph-level metadata
        if not hasattr(graph, "graph") or not isinstance(graph.graph, dict):
            msg = "Graph is missing 'graph' attribute dictionary for metadata."
            raise ValueError(msg)

        metadata_keys = ["is_hetero", "crs"]
        for key in metadata_keys:
            if key not in graph.graph:
                msg = f"Graph metadata is missing required key: '{key}'"
                raise ValueError(msg)

        # Check for node-level attributes in a single pass
        is_hetero = graph.graph.get("is_hetero", False)

        if is_hetero:
            if "node_types" not in graph.graph or not graph.graph["node_types"]:
                msg = "Heterogeneous graph metadata is missing 'node_types'."
                raise ValueError(msg)
            if "edge_types" not in graph.graph:
                msg = "Heterogeneous graph metadata is missing 'edge_types'."
                raise ValueError(msg)

        # Validate all node attributes in a single loop
        for _, node_data in graph.nodes(data=True):
            # Check for position/geometry attributes
            if "pos" not in node_data and "geometry" not in node_data:
                msg = "All nodes must have a 'pos' or 'geometry' attribute."
                raise ValueError(msg)

            # Check for node_type in heterogeneous graphs
            if is_hetero and "node_type" not in node_data:
                msg = "All nodes in a heterogeneous graph must have a 'node_type' attribute."
                raise ValueError(msg)

        # Validate edge attributes for heterogeneous graphs
        if is_hetero:
            for _, _, edge_data in graph.edges(data=True):
                if "edge_type" not in edge_data:
                    msg = "All edges in a heterogeneous graph must have an 'edge_type' attribute."
                    raise ValueError(msg)

    @staticmethod
    def ensure_crs_consistency(*gdfs: gpd.GeoDataFrame | None) -> tuple[gpd.GeoDataFrame | None, ...]:
        """Ensure all GeoDataFrames have consistent CRS."""
        non_empty_gdfs = [gdf for gdf in gdfs if gdf is not None and not gdf.empty]
        if not non_empty_gdfs:
            return gdfs

        reference_crs = non_empty_gdfs[0].crs
        for gdf in non_empty_gdfs[1:]:
            if gdf.crs != reference_crs:
                msg = "All GeoDataFrames must have the same CRS"
                raise ValueError(msg)

        return gdfs

    @staticmethod
    def extract_coordinates(gdf: gpd.GeoDataFrame, start: bool = True) -> pd.Series:
        """Extract start or end coordinates from LineString geometries."""
        if start:
            return gdf.geometry.apply(lambda g: g.coords[0] if g else None)
        return gdf.geometry.apply(lambda g: g.coords[-1] if g else None)

    @staticmethod
    def compute_centroids(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
        """Compute centroids efficiently."""
        return gdf.geometry.centroid

# =============================================================================
# GRAPH CONVERSION ENGINE
# =============================================================================

class GraphConverter:
    """Unified graph conversion engine for both homogeneous and heterogeneous graphs."""

    def __init__(self, keep_geom: bool = True, multigraph: bool = False, directed: bool = False) -> None:
        self.keep_geom = keep_geom
        self.multigraph = multigraph
        self.directed = directed
        self.processor = GeoDataProcessor()

    def gdf_to_nx(
        self,
        nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None = None,
        edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    ) -> nx.Graph | nx.MultiGraph:
        """Convert GeoDataFrames to NetworkX graph."""
        if nodes is None and edges is None:
            msg = "Either nodes or edges must be provided."
            raise ValueError(msg)

        is_nodes_dict = isinstance(nodes, dict)
        is_edges_dict = isinstance(edges, dict)

        # Determine graph type
        is_hetero = is_nodes_dict or is_edges_dict

        if is_hetero:
            return self._convert_heterogeneous(nodes, edges)
        return self._convert_homogeneous(nodes, edges)

    def nx_to_gdf(
        self,
        graph: nx.Graph | nx.MultiGraph,
        nodes: bool = True,
        edges: bool = True,
    ) -> (
        tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] |
        tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]
    ):
        """Convert NetworkX graph to GeoDataFrames."""
        metadata = GraphMetadata.from_dict(graph.graph)

        if metadata.is_hetero:
            return self._reconstruct_heterogeneous(graph, metadata, nodes, edges)
        return self._reconstruct_homogeneous(graph, metadata, nodes, edges)

    def _convert_homogeneous(
        self,
        nodes: gpd.GeoDataFrame | None,
        edges: gpd.GeoDataFrame | None,
    ) -> nx.Graph | nx.MultiGraph:
        """Convert homogeneous GeoDataFrames to NetworkX."""
        # Validate inputs
        nodes = self.processor.validate_gdf(nodes, allow_empty=True)
        edges = self.processor.validate_gdf(
            edges, ["LineString", "MultiLineString"], allow_empty=True,
        )
        # mypy: ensure edges is GeoDataFrame
        if edges is None:
            msg = "Edges GeoDataFrame cannot be None"
            raise ValueError(msg)

        self.processor.ensure_crs_consistency(nodes, edges)

        # Create graph and metadata
        if self.multigraph:
            graph = nx.MultiDiGraph() if self.directed else nx.MultiGraph()
        else:
            graph = nx.DiGraph() if self.directed else nx.Graph()
        metadata = GraphMetadata(crs=edges.crs, is_hetero=False)

        # Add nodes
        if nodes is not None:
            self._add_homogeneous_nodes(graph, nodes)
            metadata.node_geom_cols = list(nodes.select_dtypes(include=["geometry"]).columns)
            if isinstance(nodes.index, pd.MultiIndex):
                metadata.node_index_names = nodes.index.names
            else:
                metadata.node_index_names = [nodes.index.name]

        # Add edges
        self._add_homogeneous_edges(graph, edges, nodes)
        metadata.edge_geom_cols = list(edges.select_dtypes(include=["geometry"]).columns)
        metadata.edge_index_names = edges.index.names

        # Store metadata
        graph.graph.update(metadata.to_dict())
        return graph

    def _convert_heterogeneous(
        self,
        nodes_dict: dict[str, gpd.GeoDataFrame] | None,
        edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    ) -> nx.Graph | nx.MultiGraph:
        """Convert heterogeneous GeoDataFrames to NetworkX."""
        # Validate inputs
        if nodes_dict is not None:
            for node_type, node_gdf in nodes_dict.items():
                nodes_dict[node_type] = self.processor.validate_gdf(node_gdf,
                                                                    allow_empty=True)

        if edges_dict is not None:
            for edge_type, edge_gdf in edges_dict.items():
                edges_dict[edge_type] = self.processor.validate_gdf(
                    edge_gdf, ["LineString", "MultiLineString"],
                    allow_empty=True,
                )

        # Create graph and metadata
        if self.multigraph:
            graph = nx.MultiDiGraph() if self.directed else nx.MultiGraph()
        else:
            graph = nx.DiGraph() if self.directed else nx.Graph()
        metadata = GraphMetadata(is_hetero=True)

        if nodes_dict is not None and nodes_dict:
            metadata.crs = next(iter(nodes_dict.values())).crs
            metadata.node_types = list(nodes_dict.keys())

        if edges_dict is not None and edges_dict:
            if not metadata.crs:
                metadata.crs = next(iter(edges_dict.values())).crs
            metadata.edge_types = list(edges_dict.keys())

        # Add nodes and edges
        if nodes_dict:
            self._add_heterogeneous_nodes(graph, nodes_dict, metadata)
        if edges_dict:
            self._add_heterogeneous_edges(graph, edges_dict, metadata)

        # Store metadata
        graph.graph.update(metadata.to_dict())
        return graph

    def _add_homogeneous_nodes(self, graph: nx.Graph | nx.MultiGraph, nodes_gdf: gpd.GeoDataFrame) -> None:
        """Add homogeneous nodes to graph."""
        centroids = self.processor.compute_centroids(nodes_gdf)
        node_data = nodes_gdf if self.keep_geom else nodes_gdf.drop(columns="geometry")

        # Convert to list of dictionaries for attributes
        node_attrs_list = node_data.to_dict("records")

        # Create nodes with attributes
        nodes_to_add = [
            (idx, {
                **attrs,
                "_original_index": orig_idx,
                "pos": (centroid.x, centroid.y),
            })
            for idx, (orig_idx, attrs, centroid) in enumerate(
                zip(nodes_gdf.index, node_attrs_list, centroids, strict=False),
            )
        ]

        graph.add_nodes_from(nodes_to_add)

    def _add_homogeneous_edges(
        self,
        graph: nx.Graph | nx.MultiGraph,
        edges_gdf: gpd.GeoDataFrame,
        nodes_gdf: gpd.GeoDataFrame,
    ) -> None:
        """Add homogeneous edges to graph."""
        if edges_gdf.empty:
            return

        if nodes_gdf is not None and not nodes_gdf.empty:
            # Use node mapping
            coord_to_node = {
                node_data["pos"]: node_id
                for node_id, node_data in graph.nodes(data=True)
            }

            start_coords = self.processor.extract_coordinates(edges_gdf, start=True)
            end_coords = self.processor.extract_coordinates(edges_gdf, start=False)

            u_nodes = start_coords.map(coord_to_node)
            v_nodes = end_coords.map(coord_to_node)

            # Filter valid edges
            valid_mask = u_nodes.notna() & v_nodes.notna()
            valid_edges = edges_gdf[valid_mask]
            valid_u = u_nodes[valid_mask]
            valid_v = v_nodes[valid_mask]

            self._create_edge_list(graph, valid_u, valid_v, valid_edges, self.keep_geom)
        else:
            # Use coordinate tuples as node IDs
            start_coords = self.processor.extract_coordinates(edges_gdf, start=True)
            end_coords = self.processor.extract_coordinates(edges_gdf, start=False)

            # Add unique nodes
            all_coords = pd.concat([start_coords, end_coords]).unique()
            nodes_to_add = [(coord, {"pos": coord}) for coord in all_coords]
            graph.add_nodes_from(nodes_to_add)

            self._create_edge_list(graph, start_coords, end_coords, edges_gdf, self.keep_geom)

    def _add_heterogeneous_nodes(
        self,
        graph: nx.Graph | nx.MultiGraph,
        nodes_dict: dict[str, gpd.GeoDataFrame],
        metadata: "GraphMetadata",
    ) -> dict[str, int]:
        """Add heterogeneous nodes to graph."""
        if metadata.node_index_names is None or not isinstance(metadata.node_index_names, dict):
            metadata.node_index_names = {}

        node_offset = {}
        current_offset = 0

        for node_type, node_gdf in nodes_dict.items():
            node_offset[node_type] = current_offset
            metadata.node_index_names[node_type] = node_gdf.index.name

            centroids = self.processor.compute_centroids(node_gdf)
            node_data = node_gdf if self.keep_geom else node_gdf.drop(columns="geometry")

            nodes_to_add = [
                (current_offset + idx, {
                    **attrs,
                    "node_type": node_type,
                    "_original_index": orig_idx,
                    "pos": (centroid.x, centroid.y),
                })
                for idx, (orig_idx, attrs, centroid) in enumerate(
                    zip(node_gdf.index, node_data.to_dict("records"), centroids, strict=False),
                )
            ]

            graph.add_nodes_from(nodes_to_add)
            current_offset += len(node_gdf)

        return node_offset

    def _add_heterogeneous_edges(
        self,
        graph: nx.Graph | nx.MultiGraph,
        edges_dict: dict[tuple[str, str, str], gpd.GeoDataFrame],
        metadata: "GraphMetadata",
    ) -> None:
        """Add heterogeneous edges to graph."""
        if metadata.edge_index_names is None:
            metadata.edge_index_names = {}

        # Ensure edge_index_names is a dict for type safety
        if not isinstance(metadata.edge_index_names, dict):
            metadata.edge_index_names = {}

        for edge_type, edge_gdf in edges_dict.items():
            # Get edge type components
            src_type, rel_type, dst_type = edge_type
            metadata.edge_index_names[edge_type] = edge_gdf.index.names

            # Create node lookup
            node_lookup = self._create_node_lookup(graph, [src_type, dst_type])

            # Map edge indices to node IDs
            src_indices = edge_gdf.index.get_level_values(0)
            dst_indices = edge_gdf.index.get_level_values(1)

            u_nodes = pd.Series(src_indices.values, index=edge_gdf.index).map(node_lookup.get(src_type, {}))
            v_nodes = pd.Series(dst_indices.values, index=edge_gdf.index).map(node_lookup.get(dst_type, {}))

            valid_mask = u_nodes.notna() & v_nodes.notna()
            if not valid_mask.all():
                logger.warning(
                    "Could not find nodes for %d edges of type %s",
                    (~valid_mask).sum(), edge_type,
                )

            valid_edges = edge_gdf[valid_mask]
            valid_u = u_nodes[valid_mask]
            valid_v = v_nodes[valid_mask]

            self._create_edge_list(graph, valid_u, valid_v, valid_edges, self.keep_geom, edge_type)

    def _create_edge_list(
        self,
        graph: nx.Graph | nx.MultiGraph,
        u_nodes: pd.Series,
        v_nodes: pd.Series,
        edges_gdf: gpd.GeoDataFrame,
        keep_geom: bool,
        edge_type: str | tuple[str, str, str] | None = None,
    ) -> None:
        """Create edge list for adding to graph."""
        attrs_df = edges_gdf if keep_geom else edges_gdf.drop(columns="geometry")
        edge_attrs = attrs_df.to_dict("records")

        if (
            isinstance(graph, nx.MultiGraph) and
            isinstance(edges_gdf.index, pd.MultiIndex) and
            edges_gdf.index.nlevels >= 2  # Check for at least u, v
        ):
            keys = (
                edges_gdf.index.get_level_values(2)
                if edges_gdf.index.nlevels == 3
                else range(len(edges_gdf))
            )
            edges_to_add_multi = [
                (u, v, k, {
                    **attrs,
                    "_original_edge_index": orig_idx,
                    **({"edge_type": edge_type} if edge_type else {}),
                })
                for u, v, k, orig_idx, attrs in zip(
                    u_nodes, v_nodes, keys, edges_gdf.index, edge_attrs, strict=True,
                )
            ]
            graph.add_edges_from(edges_to_add_multi)
        else:
            edges_to_add_simple = [
                (u, v, {
                    **attrs,
                    "_original_edge_index": orig_idx,
                    **({"edge_type": edge_type} if edge_type else {}),
                })
                for u, v, orig_idx, attrs in zip(
                    u_nodes, v_nodes, edges_gdf.index, edge_attrs, strict=True,
                )
            ]
            graph.add_edges_from(edges_to_add_simple)

    def _create_node_lookup(
        self, graph: nx.Graph | nx.MultiGraph, node_types: list[str],
    ) -> dict[str, dict[str, int]]:
        """Create lookup dictionary for heterogeneous nodes."""
        node_lookup: dict[str, dict[str, int]] = {}
        for node_id, node_data in graph.nodes(data=True):
            node_type = node_data.get("node_type")
            orig_idx = node_data.get("_original_index")

            if node_type in node_types and orig_idx is not None:
                if node_type not in node_lookup:
                    node_lookup[node_type] = {}
                node_lookup[node_type][orig_idx] = node_id

        return node_lookup

    def _reconstruct_homogeneous(
        self,
        graph: nx.Graph | nx.MultiGraph,
        metadata: "GraphMetadata",
        nodes: bool = True,
        edges: bool = True,
    ) -> tuple[gpd.GeoDataFrame | None, gpd.GeoDataFrame | None] | gpd.GeoDataFrame:
        """Reconstruct homogeneous GeoDataFrames from NetworkX graph."""
        result: list[gpd.GeoDataFrame] = []

        if nodes:
            nodes_gdf = self._create_homogeneous_nodes_gdf(graph, metadata)
            result.append(nodes_gdf)

        if edges:
            edges_gdf = self._create_homogeneous_edges_gdf(graph, metadata)
            result.append(edges_gdf)

        if len(result) == 1:
            return result[0]
        return (result[0], result[1])

    def _reconstruct_heterogeneous(
        self,
        graph: nx.Graph | nx.MultiGraph,
        metadata: "GraphMetadata",
        nodes: bool = True,
        edges: bool = True,
    ) -> tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]:
        """Reconstruct heterogeneous GeoDataFrames from NetworkX graph."""
        nodes_dict = {}
        edges_dict = {}

        if metadata.node_index_names is None:
            metadata.node_index_names = {}
        if metadata.edge_index_names is None:
            metadata.edge_index_names = {}

        if nodes:
            nodes_dict = self._create_heterogeneous_nodes_dict(graph, metadata)

        if edges:
            edges_dict = self._create_heterogeneous_edges_dict(graph, metadata)

        return nodes_dict, edges_dict

    def _create_homogeneous_nodes_gdf(
        self,
        graph: nx.Graph | nx.MultiGraph,
        metadata: "GraphMetadata",
    ) -> gpd.GeoDataFrame:
        """Create homogeneous nodes GeoDataFrame."""
        node_data = dict(graph.nodes(data=True))

        # Extract original indices and create records
        original_indices = [attrs.get("_original_index", nid) for nid, attrs in node_data.items()]

        # Use list comprehension for records, prioritize geometry over pos
        records = [
            {
                **{k: v for k, v in attrs.items() if k not in ["pos", "_original_index"]},
                "geometry": attrs["geometry"] if "geometry" in attrs and attrs["geometry"] is not None
                            else (Point(attrs["pos"]) if "pos" in attrs else None),
            }
            for nid, attrs in node_data.items()
        ]

        index_names = metadata.node_index_names

        # Handle different types of index_names
        if isinstance(index_names, list):
            names = index_names if len(index_names) > 1 else (index_names[0] if index_names else None)
            index = (pd.MultiIndex.from_tuples(original_indices, names=names)
                if len(index_names) > 1
                else pd.Index(original_indices, name=names))
        else:
            # Handle str, None, or other types
            index = pd.Index(original_indices, name=index_names if isinstance(index_names, str) else None)

        gdf = gpd.GeoDataFrame(records, index=index, crs=metadata.crs)

        # Convert geometry columns
        for col in metadata.node_geom_cols:
            if col in gdf.columns:
                gdf[col] = gpd.GeoSeries(gdf[col], crs=metadata.crs)

        return gdf

    def _create_heterogeneous_nodes_dict(
        self,
        graph: nx.Graph | nx.MultiGraph,
        metadata: "GraphMetadata",
    ) -> dict[str, gpd.GeoDataFrame]:
        """Create heterogeneous nodes dictionary."""
        nodes_dict = {}

        for node_type in metadata.node_types:
            type_nodes = [
                (n, d) for n, d in graph.nodes(data=True)
                if d.get("node_type") == node_type
            ]

            node_ids, attrs_list = zip(*type_nodes, strict=False)
            indices = [attrs.get("_original_index") for attrs in attrs_list]

            # Use list comprehension for records, prioritize geometry over pos
            records = [
                {
                    **{k: v for k, v in attrs.items() if k not in ["pos", "node_type", "_original_index"]},
                    "geometry": attrs["geometry"] if "geometry" in attrs and attrs["geometry"] is not None
                                else (Point(attrs["pos"]) if "pos" in attrs else None),
                }
                for attrs in attrs_list
            ]

            # Handle index names safely
            index_names = metadata.node_index_names.get(node_type) if isinstance(metadata.node_index_names, dict) else None

            index = pd.Index(indices, name=index_names) if isinstance(index_names, str) else pd.Index(indices, name=None)
            gdf = gpd.GeoDataFrame(records, geometry="geometry", index=index, crs=metadata.crs)

            nodes_dict[node_type] = gdf

        return nodes_dict

    def _create_homogeneous_edges_gdf(
        self,
        graph: nx.Graph | nx.MultiGraph,
        metadata: "GraphMetadata",
    ) -> gpd.GeoDataFrame:
        """Create homogeneous edges GeoDataFrame."""
        if graph.number_of_edges() == 0:
            # Create empty GeoDataFrame with expected columns
            return gpd.GeoDataFrame({"weight": [], "geometry": []}, crs=metadata.crs)

        is_multigraph = isinstance(graph, nx.MultiGraph)
        if is_multigraph:
            edge_data = list(graph.edges(data=True, keys=True))
            original_indices = [
                attrs.get("_original_edge_index", (u, v, k))
                for u, v, k, attrs in edge_data
            ]
        else:
            edge_data = list(graph.edges(data=True))
            original_indices = [
                attrs.get("_original_edge_index", (u, v))
                for u, v, attrs in edge_data
            ]

        records = []
        for edge in edge_data:
            if is_multigraph:
                # Multigraph edges have format (u, v, k, attrs)
                u, v, _, attrs = edge
            else:
                # Regular edges have format (u, v, attrs)
                u, v, attrs = edge

            geom = attrs.get("geometry")
            if geom is None and "pos" in graph.nodes[u] and "pos" in graph.nodes[v]:
                geom = LineString([graph.nodes[u]["pos"], graph.nodes[v]["pos"]])

            records.append(
                {
                    **{k: v for k, v in attrs.items() if k not in ["_original_edge_index", "weight"]},
                    "weight": attrs.get("weight"),
                    "geometry": geom,
                },
            )

        # Handle MultiIndex
        if original_indices and isinstance(original_indices[0], tuple):
            index = pd.MultiIndex.from_tuples(original_indices)
        else:
            index = pd.Index(original_indices)

        gdf = gpd.GeoDataFrame(records, index=index, crs=metadata.crs)

        # Restore index names
        index_names = metadata.edge_index_names

        if index_names and hasattr(gdf.index, "names"):
            gdf.index.names = index_names

        # Convert geometry columns
        for col in metadata.edge_geom_cols:
            if col in gdf.columns:
                gdf[col] = gpd.GeoSeries(gdf[col], crs=metadata.crs)

        return gdf

    def _create_heterogeneous_edges_dict(
        self,
        graph: nx.Graph | nx.MultiGraph,
        metadata: "GraphMetadata",
    ) -> dict[tuple[str, str, str], gpd.GeoDataFrame]:
        """Create heterogeneous edges dictionary."""
        edges_dict = {}
        is_multigraph = isinstance(graph, nx.MultiGraph)

        for edge_type in metadata.edge_types:
            src_type, rel_type, dst_type = edge_type

            if is_multigraph:
                multigraph_edges: list[tuple[object, object, object, dict[str, object]]] = [
                    (u, v, k, d) for u, v, k, d in graph.edges(data=True, keys=True)
                    if d.get("edge_type") == edge_type
                ]
                # Convert to unified format for processing
                type_edges: list[tuple[object, object, object, dict[str, object]]] = multigraph_edges
            else:
                regular_edges: list[tuple[object, object, dict[str, object]]] = [
                    (u, v, d) for u, v, d in graph.edges(data=True)
                    if d.get("edge_type") == edge_type
                ]
                # Convert to unified format for processing (adding None for key)
                type_edges = [(u, v, None, d) for u, v, d in regular_edges]

            if not type_edges:
                edges_dict[edge_type] = gpd.GeoDataFrame(geometry=[], crs=metadata.crs)
                continue

            original_indices = [
                edge[-1].get("_original_edge_index") for edge in type_edges
            ]
            records = []
            for edge in type_edges:
                # Unified format: (u, v, k_or_None, attrs)
                u, v, k, attrs = edge

                geom = attrs.get("geometry")
                if geom is None and "pos" in graph.nodes[u] and "pos" in graph.nodes[v]:
                    geom = LineString([graph.nodes[u]["pos"], graph.nodes[v]["pos"]])

                records.append(
                    {
                        **{
                            k: v
                            for k, v in attrs.items()
                            if k not in ["full_edge_type", "_original_edge_index"]
                        },
                        "geometry": geom,
                    },
                )

            # Handle MultiIndex
            index = pd.MultiIndex.from_tuples(original_indices)
            gdf = gpd.GeoDataFrame(records, geometry="geometry", index=index, crs=metadata.crs)

            # Restore index names safely
            if isinstance(metadata.edge_index_names, dict):
                index_names = metadata.edge_index_names.get(edge_type)
                if isinstance(index_names, list) and hasattr(gdf.index, "names"):
                    gdf.index.names = index_names

            edges_dict[edge_type] = gdf

        return edges_dict

class GraphAnalyzer:
    """Unified graph analysis operations."""

    def __init__(self) -> None:
        self.processor = GeoDataProcessor()
        self.converter = GraphConverter()

    def filter_graph_by_distance(
        self,
        graph: gpd.GeoDataFrame | nx.Graph | nx.MultiGraph,
        center_point: Point | gpd.GeoSeries,
        distance: float,
        edge_attr: str = "length",
        node_id_col: str | None = None,
    ) -> gpd.GeoDataFrame | nx.Graph | nx.MultiGraph:
        """Extract a filtered graph containing only elements within a given shortest-path distance."""
        is_graph_input = isinstance(graph, (nx.Graph, nx.MultiGraph))

        # Convert to NetworkX if needed
        if is_graph_input:
            nx_graph = graph
            original_crs = nx_graph.graph.get("crs")
        else:
            nx_graph = self.converter.gdf_to_nx(edges=graph)
            original_crs = graph.crs if hasattr(graph, "crs") else None

        # Extract node positions
        pos_dict = self._extract_node_positions(nx_graph)
        if not pos_dict:
            graph_type = type(graph) if is_graph_input else nx.Graph
            return self._create_empty_result(is_graph_input, original_crs, graph_type)

        # Create nodes GeoDataFrame for distance calculations
        node_id_name = node_id_col or "node_id"
        nodes_gdf = self._create_nodes_gdf(pos_dict, node_id_name, original_crs)

        # Normalize center points
        center_points = self._normalize_center_points(center_point)

        # Compute nodes within distance
        nodes_within_distance = self._compute_nodes_within_distance(
            nx_graph, center_points, nodes_gdf, distance, edge_attr, node_id_name,
        )

        # Create subgraph
        subgraph = nx_graph.subgraph(nodes_within_distance)

        if is_graph_input:
            return subgraph

        # Convert back to GeoDataFrame
        return self.converter.nx_to_gdf(subgraph, nodes=False, edges=True)

    def create_isochrone(
        self,
        graph: gpd.GeoDataFrame | nx.Graph | nx.MultiGraph,
        center_point: Point | gpd.GeoSeries | gpd.GeoDataFrame,
        distance: float,
        edge_attr: str = "length",
    ) -> gpd.GeoDataFrame:
        """Generate isochrone polygon(s) as convex hull of reachable areas within distance."""
        reachable = self.filter_graph_by_distance(graph, center_point, distance, edge_attr)

        # Convert to GeoDataFrame if NetworkX
        if isinstance(reachable, (nx.Graph, nx.MultiGraph)):
            reachable = self.converter.nx_to_gdf(reachable, nodes=False, edges=True)

        if reachable.empty:
            return gpd.GeoDataFrame(geometry=[], crs=getattr(reachable, "crs", None))

        # Create convex hull
        union_geom = reachable.union_all()
        hull = union_geom.convex_hull
        return gpd.GeoDataFrame(geometry=[hull], crs=reachable.crs)

    def _extract_node_positions(self, graph: nx.Graph | nx.MultiGraph) -> dict[object, object] | None:
        """Extract node positions from a NetworkX graph."""
        pos_dict: dict[object, object] = nx.get_node_attributes(graph, "pos")

        if pos_dict:
            return pos_dict

        return None

    def _create_nodes_gdf(
        self,
        pos_dict: dict[object, object],
        node_id_col: str,
        crs: str | int | None,
    ) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame from node positions."""
        node_ids, coordinates = zip(*pos_dict.items(), strict=False)
        geometries = [Point(coord) for coord in coordinates]

        return gpd.GeoDataFrame(
            {node_id_col: node_ids, "geometry": geometries}, crs=crs,
        )

    def _normalize_center_points(
        self, center_point: Point | gpd.GeoSeries,
    ) -> list[Point] | gpd.GeoSeries:
        """Normalize center point input to a consistent format."""
        if isinstance(center_point, gpd.GeoSeries):
            return center_point
        return [center_point]

    def _compute_nodes_within_distance(
        self,
        graph: nx.Graph | nx.MultiGraph,
        center_points: list[Point] | gpd.GeoSeries,
        nodes_gdf: gpd.GeoDataFrame,
        distance: float,
        edge_attr: str,
        node_id_name: str,
    ) -> set[object]:
        """Compute all nodes within distance from any center point."""
        center_points_list = (
            center_points.tolist() if hasattr(center_points, "tolist") else list(center_points)
        )

        # Get nearest nodes for all center points
        source_nodes = []
        for point in center_points_list:
            nearest_node = self._get_nearest_node(point, nodes_gdf, node_id_name)
            source_nodes.append(nearest_node)

        # Compute single-source shortest paths from all sources
        all_reachable = set()
        for source in source_nodes:
            lengths = nx.single_source_dijkstra_path_length(
                graph, source, cutoff=distance, weight=edge_attr,
            )
            all_reachable.update(lengths.keys())
        return all_reachable

    def _get_nearest_node(
        self, point: Point | gpd.GeoSeries, nodes_gdf: gpd.GeoDataFrame, node_id: str,
    ) -> object:
        """Find the nearest node in a GeoDataFrame."""
        nearest_idx = nodes_gdf.distance(point).idxmin()
        return nodes_gdf.loc[nearest_idx, node_id]

    def _create_empty_result(
        self,
        is_graph_input: bool,
        original_crs: str | int | None,
        graph_type: type = nx.Graph,
    ) -> gpd.GeoDataFrame | nx.Graph | nx.MultiGraph:
        """Create an empty result in the appropriate format."""
        return gpd.GeoDataFrame(geometry=[], crs=original_crs) if not is_graph_input else graph_type()


# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def dual_graph(
    graph: tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.MultiGraph,
    edge_id_col: str | None,
    keep_original_geom: bool = False,
    as_nx: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] | nx.Graph | nx.MultiGraph:
    """Convert a primal graph represented by nodes and edges GeoDataFrames to its dual graph.

    In the dual graph, original edges become nodes and original nodes become edges connecting
    adjacent original edges.

    Parameters
    ----------
    graph : tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame] or networkx.Graph or networkx.MultiGraph
        A graph containing nodes and edges GeoDataFrames or a NetworkX graph of the primal graph.
    edge_id_col : str, optional
        The name of the column in the edges GeoDataFrame to be used as unique identifiers
        for dual graph nodes. If None, the index of the edges GeoDataFrame is used.
        Default is None.
    keep_original_geom : bool, default False
        If True, preserve the original geometry of the edges in a new column named
        'original_geometry' in the dual nodes GeoDataFrame.
    as_nx : bool, default False
        If True, return the dual graph as a NetworkX graph instead of GeoDataFrames.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        A tuple containing the nodes and edges of the dual graph as GeoDataFrames.
        - Dual nodes GeoDataFrame: Nodes represent original edges. The geometry is the
          centroid of the original edge's geometry. The index is derived from `edge_id_col`
          or the original edge index.
        - Dual edges GeoDataFrame: Edges represent adjacency between original edges (i.e.,
          they shared a node in the primal graph). The geometry is a LineString connecting
          the centroids of the two dual nodes. The index is a MultiIndex of the connected
          dual node IDs.

    Examples
    --------
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> from shapely.geometry import Point, LineString
    >>> # Primal graph nodes
    >>> nodes = gpd.GeoDataFrame(
    ...     {"node_id": [0, 1, 2]},
    ...     geometry=[Point(0, 0), Point(1, 1), Point(1, 0)],
    ...     crs="EPSG:32633"
    ... ).set_index("node_id")
    >>> # Primal graph edges
    >>> edges = gpd.GeoDataFrame(
    ...     {"edge_id": ["a", "b"]},
    ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (1, 0)])],
    ...     crs="EPSG:32633"
    ... ).set_index(pd.MultiIndex.from_tuples([(0, 1), (1, 2)]))
    >>> # Convert to dual graph
    >>> dual_nodes, dual_edges = dual_graph(
    ...     graph=(nodes, edges), edge_id_col="edge_id", keep_original_geom=True
    ... )
    >>> print(dual_nodes)
    >>> print(dual_edges)
    >>>                     geometry      original_geometry    mm_len
    ... edge_id
    ... a        LINESTRING (0 0, 1 1)  LINESTRING (0 0, 1 1)  1.414214
    ... b        LINESTRING (1 1, 1 0)  LINESTRING (1 1, 1 0)  1.000000
    ...                          angle  geometry
    ... from_edge_id to_edge_id
    ... a            b           135.0  LINESTRING (0.5 0.5, 1 0.5)
    """
    processor = GeoDataProcessor()

    # Validate input type
    if not (
        isinstance(graph, (nx.Graph, nx.MultiGraph))
        or (isinstance(graph, tuple) and len(graph) == 2)
    ):
        msg = "Input `graph` must be a tuple of (nodes_gdf, edges_gdf) or a NetworkX graph."
        raise TypeError(msg)

    if isinstance(graph, (nx.Graph, nx.MultiGraph)):
        # If input is a NetworkX graph, convert it to GeoDataFrames
        converter = GraphConverter()
        nodes_gdf, edges_gdf = converter.nx_to_gdf(graph, nodes=True, edges=True)
    else:
        # Input is guaranteed to be tuple[GeoDataFrame, GeoDataFrame] by type annotation
        nodes_gdf, edges_gdf = graph

    processor.ensure_crs_consistency(nodes_gdf, edges_gdf)

    # Validate edges_gdf is a GeoDataFrame and clean it.
    # This will raise TypeError for non-GDF input, fixing one test failure.
    edges_clean = processor.validate_gdf(
        edges_gdf, ["LineString", "MultiLineString"], allow_empty=True,
    )

    # Handle empty or cleaned-to-empty edges GeoDataFrame.
    # This will fix the StopIteration test failure.
    if edges_clean is None or edges_clean.empty:
        crs = getattr(edges_gdf, "crs", None)
        dual_nodes = gpd.GeoDataFrame(geometry=[], crs=crs)
        dual_edges = gpd.GeoDataFrame(geometry=[], crs=crs)
        return dual_nodes, dual_edges

    # edges_clean is guaranteed to be non-None and non-empty here
    assert edges_clean is not None
    assert not edges_clean.empty

    if keep_original_geom:
        edges_clean["original_geometry"] = gpd.GeoSeries(
            edges_clean.geometry.copy(), crs=edges_clean.crs,
        )

    # If no edge_id_col, we'll use the index. Let's add it as a column
    # so it's carried over as a node attribute in the dual graph.
    preserve_index = edge_id_col is None
    # momepy uses the index of the input GDF as node IDs in the dual graph
    graph_nx = momepy.gdf_to_nx(edges_clean, approach="dual", multigraph=False, preserve_index=preserve_index)

    # Ensure all edges from the primal graph are present as nodes in the dual graph, with their attributes.
    if preserve_index:
        if edges_clean is not None:
            node_attrs = edges_clean.to_dict("index")
            for node_id, attrs in node_attrs.items():
                if node_id not in graph_nx.nodes:
                    graph_nx.add_node(node_id, **attrs)
    elif edges_clean is not None:
        records = edges_clean.to_dict("records")
        nodes_to_add = [(i, attrs) for i, attrs in enumerate(records) if i not in graph_nx.nodes]
        graph_nx.add_nodes_from(nodes_to_add)

    # Add edge attributes of geometry as "geometry" of linestrings between centroids of nodes
    for u, v in graph_nx.edges():
        u_geom = graph_nx.nodes[u]["geometry"]
        v_geom = graph_nx.nodes[v]["geometry"]
        line = LineString([u_geom.centroid, v_geom.centroid])
        graph_nx.edges[u, v]["geometry"] = line

    # Convert the NetworkX graph to GeoDataFrames
    dual_nodes, dual_edges = nx_to_gdf(graph_nx, nodes=True, edges=True)

    # Ensure dual_nodes is a GeoDataFrame for type checking
    assert isinstance(dual_nodes, gpd.GeoDataFrame)
    assert isinstance(dual_edges, gpd.GeoDataFrame)

    new_index_name = None
    if edge_id_col:
        # Create a mapping from the old index (used by momepy) to the new index values
        id_map = dual_nodes[edge_id_col]

        # Set the new index for the dual nodes
        dual_nodes = dual_nodes.set_index(edge_id_col)
        new_index_name = edge_id_col

        # Remap the dual edges' MultiIndex to use the new node IDs
        if isinstance(dual_edges, gpd.GeoDataFrame) and not dual_edges.empty:
            level_0 = dual_edges.index.get_level_values(0).map(id_map)
            level_1 = dual_edges.index.get_level_values(1).map(id_map)
            dual_edges.index = pd.MultiIndex.from_arrays([level_0, level_1])

    # Align edge index names with the new node index name
    if new_index_name and isinstance(dual_edges, gpd.GeoDataFrame) and not dual_edges.empty:
        dual_edges.index.names = [f"from_{new_index_name}", f"to_{new_index_name}"]

    return dual_nodes, dual_edges if not as_nx else gdf_to_nx(dual_nodes, dual_edges)


def segments_to_graph(
    segments_gdf: gpd.GeoDataFrame,
    multigraph: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    r"""Convert a GeoDataFrame of LineString segments into a graph structure.

    This function takes a GeoDataFrame of LineStrings and processes it into a
    topologically explicit graph representation, consisting of a GeoDataFrame of
    unique nodes (the endpoints of the lines) and a GeoDataFrame of edges.

    The resulting nodes GeoDataFrame contains unique points representing the start
    and end points of the input line segments. The edges GeoDataFrame is a copy
    of the input, but with a new MultiIndex (`from_node_id`, `to_node_id`) that
    references the IDs in the new nodes GeoDataFrame. If `multigraph` is True
    and there are multiple edges between the same pair of nodes, an additional
    index level (`edge_key`) is added to distinguish them.

    Parameters
    ----------
    segments_gdf : geopandas.GeoDataFrame
        A GeoDataFrame where each row represents a line segment, and the
        'geometry' column contains LineString objects.
    multigraph : bool, default False
        If True, supports multiple edges between the same pair of nodes by
        adding an `edge_key` level to the MultiIndex. This is useful when
        the input contains duplicate node-to-node connections that should
        be preserved as separate edges.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        A tuple containing two GeoDataFrames:
        - nodes_gdf: A GeoDataFrame of unique nodes (Points), indexed by `node_id`.
        - edges_gdf: A GeoDataFrame of edges (LineStrings), with a MultiIndex
          mapping to the `node_id` in `nodes_gdf`. If `multigraph` is True,
          the index includes a third level (`edge_key`) for duplicate connections.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import LineString
    >>> # Create a GeoDataFrame of line segments
    >>> segments = gpd.GeoDataFrame(
    ...     {"road_name": ["A", "B"]},
    ...     geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (1, 0)])],
    ...     crs="EPSG:32633"
    ... )
    >>> # Convert to graph representation
    >>> nodes_gdf, edges_gdf = segments_to_graph(segments)
    >>> print(nodes_gdf)
    >>> print(edges_gdf)
    node_id  geometry
    0        POINT (0 0)
    1        POINT (1 1)
    2        POINT (1 0)
                                    road_name   geometry
    from_node_id to_node_id
    0            1                  A           LINESTRING (0 0, 1 1)
    1            2                  B           LINESTRING (1 1, 1 0)

    >>> # Example with duplicate connections (multigraph)
    >>> segments_with_duplicates = gpd.GeoDataFrame(
    ...     {"road_name": ["A", "B", "C"]},
    ...     geometry=[LineString([(0, 0), (1, 1)]),
    ...               LineString([(0, 0), (1, 1)]),
    ...               LineString([(1, 1), (1, 0)])],
    ...     crs="EPSG:32633"
    ... )
    >>> nodes_gdf, edges_gdf = segments_to_graph(segments_with_duplicates, multigraph=True)
    >>> print(edges_gdf.index.names)
    ['from_node_id', 'to_node_id', 'edge_key']
    """
    processor = GeoDataProcessor()

    # Validate input
    segments_clean = processor.validate_gdf(segments_gdf, ["LineString"])

    if segments_clean is None or segments_clean.empty:
        empty_nodes = gpd.GeoDataFrame(columns=["geometry"], crs=segments_gdf.crs)
        empty_edges = gpd.GeoDataFrame(columns=["geometry"], crs=segments_gdf.crs)
        return empty_nodes, empty_edges

    # Extract coordinates
    start_coords = processor.extract_coordinates(segments_clean, start=True)
    end_coords = processor.extract_coordinates(segments_clean, start=False)

    # Create unique nodes
    all_coords = pd.concat([start_coords, end_coords]).drop_duplicates()
    coord_to_id = {coord: i for i, coord in enumerate(all_coords)}

    # Create nodes GeoDataFrame efficiently using gpd.points_from_xy
    coords_array = all_coords.to_numpy()
    x_coords = [coord[0] for coord in coords_array]
    y_coords = [coord[1] for coord in coords_array]

    # Create nodes GeoDataFrame with unique node IDs
    nodes_gdf = gpd.GeoDataFrame(
        {
            "node_id": range(len(all_coords)),
            "geometry": gpd.points_from_xy(x_coords, y_coords),
        },
        crs=segments_clean.crs,
    ).set_index("node_id", drop=True)

    # Create edges with MultiIndex
    from_ids = start_coords.map(coord_to_id)
    to_ids = end_coords.map(coord_to_id)

    edges_gdf = segments_clean.copy()

    if multigraph:
        # For multigraph, handle potential duplicate node pairs by adding edge keys
        edge_pairs_df = pd.DataFrame({"from_id": from_ids, "to_id": to_ids})
        edge_keys = edge_pairs_df.groupby(["from_id", "to_id"]).cumcount()

        edges_gdf.index = pd.MultiIndex.from_arrays(
            [from_ids, to_ids, edge_keys],
            names=["from_node_id", "to_node_id", "edge_key"],
        )
    else:
        edges_gdf.index = pd.MultiIndex.from_arrays(
            [from_ids, to_ids], names=["from_node_id", "to_node_id"],
        )

    return nodes_gdf, edges_gdf

def gdf_to_nx(
    nodes: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None = None,
    edges: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    keep_geom: bool = True,
    multigraph: bool = False,
    directed: bool = False,
) -> nx.Graph | nx.MultiGraph | nx.DiGraph | nx.MultiDiGraph:
    """Convert GeoDataFrames of nodes and edges to a NetworkX graph.

    This function provides a high-level interface to convert geospatial data,
    represented as GeoDataFrames, into a NetworkX graph. It supports both
    homogeneous and heterogeneous graphs.

    For homogeneous graphs, provide a single GeoDataFrame for nodes and edges.
    For heterogeneous graphs, provide dictionaries mapping type names to
    GeoDataFrames.

    Parameters
    ----------
    nodes : geopandas.GeoDataFrame or dict[str, geopandas.GeoDataFrame], optional
        Node data. For homogeneous graphs, a single GeoDataFrame. For
        heterogeneous graphs, a dictionary mapping node type names to
        GeoDataFrames. Node IDs are taken from the GeoDataFrame index.
    edges : geopandas.GeoDataFrame or dict, optional
        Edge data. For homogeneous graphs, a single GeoDataFrame. For
        heterogeneous graphs, a dictionary mapping edge type tuples
        (source_type, relation_type, target_type) to GeoDataFrames.
        Edge relationships are defined by a MultiIndex on the edge
        GeoDataFrame (source ID, target ID). For MultiGraphs, a third level
        in the index can be used for edge keys.
    keep_geom : bool, default True
        If True, the geometry of the nodes and edges GeoDataFrames will be
        preserved as attributes in the NetworkX graph.
    multigraph : bool, default False
        If True, a `networkx.MultiGraph` is created, which can store multiple
        edges between the same two nodes.
    directed : bool, default False
        If True, a directed graph (`networkx.DiGraph` or `networkx.MultiDiGraph`)
        is created. Otherwise, an undirected graph is created.

    Returns
    -------
    networkx.Graph or networkx.MultiGraph or networkx.DiGraph or networkx.MultiDiGraph
        A NetworkX graph object representing the spatial network. Graph-level
        metadata, such as CRS and heterogeneity information, is stored in
        `graph.graph`.

    Examples
    --------
    >>> # Homogeneous graph
    >>> import geopandas as gpd
    >>> import pandas as pd
    >>> from shapely.geometry import Point, LineString
    >>> nodes_gdf = gpd.GeoDataFrame(
    ...     geometry=[Point(0, 0), Point(1, 1)],
    ...     index=pd.Index([10, 20], name="node_id")
    ... )
    >>> edges_gdf = gpd.GeoDataFrame(
    ...     {"length": [1.414]},
    ...     geometry=[LineString([(0, 0), (1, 1)])],
    ...     index=pd.MultiIndex.from_tuples([(10, 20)], names=["u", "v"])
    ... )
    >>> G = gdf_to_nx(nodes=nodes_gdf, edges=edges_gdf)
    >>> print(G.nodes(data=True))
    >>> [(0, {'geometry': <POINT (0 0)>,
    ...       '_original_index': 10,
    ...       'pos': (0.0, 0.0)}),
    ...  (1, {'geometry': <POINT (1 1)>,
    ...       '_original_index': 20,
    ...       'pos': (1.0, 1.0)})]
    >>> print(G.edges(data=True))
    >>> [(0, 1, {'length': 1.414,
    ...          'geometry': <LINESTRING (0 0, 1 1)>,
    ...          '_original_edge_index': (10, 20)})]

    >>> # Heterogeneous graph
    >>> buildings_gdf = gpd.GeoDataFrame(geometry=[Point(0, 0)], index=pd.Index(['b1'], name="b_id"))
    >>> streets_gdf = gpd.GeoDataFrame(geometry=[Point(1, 1)], index=pd.Index(['s1'], name="s_id"))
    >>> connections_gdf = gpd.GeoDataFrame(
    ...     geometry=[LineString([(0,0), (1,1)])],
    ...     index=pd.MultiIndex.from_tuples([('b1', 's1')])
    ... )
    >>> nodes_dict = {"building": buildings_gdf, "street": streets_gdf}
    >>> edges_dict = {("building", "connects_to", "street"): connections_gdf}
    >>> H = gdf_to_nx(nodes=nodes_dict, edges=edges_dict)
    >>> print(H.nodes(data=True))
    >>> [(0, {'geometry': <POINT (0 0)>,
    ...       'node_type': 'building',
    ...       '_original_index': 'b1',
    ...       'pos': (0.0, 0.0)}),
    ...  (1, {'geometry': <POINT (1 1)>,
    ...       'node_type': 'street',
    ...       '_original_index': 's1',
    ...       'pos': (1.0, 1.0)})]
    >>> print(H.edges(data=True))
    >>> [(0, 1, {'geometry': <LINESTRING (0 0, 1 1)>,
    ...          'full_edge_type': ('building', 'connects_to', 'street'),
    ...          '_original_edge_index': ('b1', 's1')})]
    """
    # Validate inputs using enhanced validation with type detection
    validated_nodes, validated_edges, _ = validate_gdf(
        nodes_gdf=nodes, edges_gdf=edges, allow_empty=True,
    )

    converter = GraphConverter(keep_geom=keep_geom, multigraph=multigraph, directed=directed)
    return converter.gdf_to_nx(validated_nodes, validated_edges)

def nx_to_gdf(
    G: nx.Graph | nx.MultiGraph, nodes: bool = True, edges: bool = True,
) -> (gpd.GeoDataFrame | tuple[gpd.GeoDataFrame, gpd.GeoDataFrame] |
      tuple[dict[str, gpd.GeoDataFrame], dict[tuple[str, str, str], gpd.GeoDataFrame]]):
    r"""Convert a NetworkX graph to GeoDataFrames for nodes and/or edges.

    This function reconstructs GeoDataFrames from a NetworkX graph that was
    created by `gdf_to_nx` or follows a similar structure. It can handle both
    homogeneous and heterogeneous graphs, extracting node and edge attributes
    and reconstructing geometries from position data.

    Parameters
    ----------
    G : networkx.Graph or networkx.MultiGraph
        The NetworkX graph to convert. It is expected to have metadata stored
        in `G.graph` to guide the conversion, including CRS and heterogeneity
        information. Node positions are expected in a 'pos' attribute.
    nodes : bool, default True
        If True, a GeoDataFrame for nodes will be created and returned.
    edges : bool, default True
        If True, a GeoDataFrame for edges will be created and returned.

    Returns
    -------
    geopandas.GeoDataFrame or tuple
        The returned type depends on the graph type and input parameters:
        - Homogeneous graph:
            - `(nodes_gdf, edges_gdf)` if `nodes` and `edges` are True.
            - `nodes_gdf` if only `nodes` is True.
            - `edges_gdf` if only `edges` is True.
        - Heterogeneous graph:
            - `(nodes_dict, edges_dict)` where dicts map types to GeoDataFrames.

    Raises
    ------
    ValueError
        If both `nodes` and `edges` are False.

    Examples
    --------
    >>> import networkx as nx
    >>> # Create a simple graph with spatial attributes
    >>> G = nx.Graph(is_hetero=False, crs="EPSG:4326")
    >>> G.add_node(0, pos=(0, 0), population=100, geometry=Point(0,0))
    >>> G.add_node(1, pos=(1, 1), population=200, geometry=Point(1,1))
    >>> G.add_edge(0, 1, weight=1.5, geometry=LineString([(0, 0), (1, 1)]))
    >>> # Convert back to GeoDataFrames
    >>> nodes_gdf, edges_gdf = nx_to_gdf(G)
    >>> print(nodes_gdf)
    >>> print(edges_gdf)
    >>>           population     geometry
    ... 0         100           POINT (0 0)
    ... 1         200           POINT (1 1)
    ...           weight        geometry
    ... 0 1       1.5           LINESTRING (0 0, 1 1)
    """
    if not (nodes or edges):
        msg = "Must request at least one of nodes or edges"
        raise ValueError(msg)
    converter = GraphConverter()
    return converter.nx_to_gdf(G, nodes, edges)

def filter_graph_by_distance(
    graph: gpd.GeoDataFrame | nx.Graph | nx.MultiGraph,
    center_point: Point | gpd.GeoSeries,
    distance: float,
    edge_attr: str = "length",
    node_id_col: str | None = None,
) -> gpd.GeoDataFrame | nx.Graph | nx.MultiGraph:
    """Filter a graph to include only elements within a specified distance from a center point.

    This function calculates the shortest path from a center point to all nodes
    in the graph and returns a subgraph containing only the nodes (and their
    induced edges) that are within the given distance. The input can be a
    NetworkX graph or an edges GeoDataFrame.

    Parameters
    ----------
    graph : geopandas.GeoDataFrame or networkx.Graph or networkx.MultiGraph
        The graph to filter. If a GeoDataFrame, it represents the edges of the
        graph and will be converted to a NetworkX graph internally.
    center_point : Point or geopandas.GeoSeries
        The origin point(s) for the distance calculation. If multiple points
        are provided, the filter will include nodes reachable from any of them.
    distance : float
        The maximum shortest-path distance for a node to be included in the
        filtered graph.
    edge_attr : str, default "length"
        The name of the edge attribute to use as weight for shortest path
        calculations (e.g., 'length', 'travel_time').
    node_id_col : str, optional
        The name of the node identifier column if the input graph is a
        GeoDataFrame. Defaults to the index.

    Returns
    -------
    geopandas.GeoDataFrame or networkx.Graph or networkx.MultiGraph
        The filtered subgraph. The return type matches the input `graph` type.
        If the input was a GeoDataFrame, the output is a GeoDataFrame of the
        filtered edges.

    Examples
    --------
    >>> import networkx as nx
    >>> from shapely.geometry import Point
    >>> # Create a graph
    >>> G = nx.Graph()
    >>> G.add_node(0, pos=(0, 0))
    >>> G.add_node(1, pos=(10, 0))
    >>> G.add_node(2, pos=(20, 0))
    >>> G.add_edge(0, 1, length=10)
    >>> G.add_edge(1, 2, length=10)
    >>> # Filter the graph
    >>> center = Point(1, 0)
    >>> filtered_graph = filter_graph_by_distance(G, center, distance=12)
    >>> print(list(filtered_graph.nodes))
    >>> [0, 1]
    """
    analyzer = GraphAnalyzer()
    return analyzer.filter_graph_by_distance(graph, center_point, distance, edge_attr, node_id_col)

def create_isochrone(
    graph: gpd.GeoDataFrame | nx.Graph | nx.MultiGraph,
    center_point: Point | gpd.GeoSeries | gpd.GeoDataFrame,
    distance: float,
    edge_attr: str = "length",
) -> gpd.GeoDataFrame:
    """Generate an isochrone polygon from a graph.

    An isochrone represents the area reachable from a center point within a
    given travel distance or time. This function computes the set of reachable
    edges and nodes in a network and generates a polygon (the convex hull)
    that encloses this reachable area.

    Parameters
    ----------
    graph : geopandas.GeoDataFrame or networkx.Graph or networkx.MultiGraph
        The network graph. If a GeoDataFrame, it represents the edges of the
        graph.
    center_point : Point or geopandas.GeoSeries or geopandas.GeoDataFrame
        The origin point(s) for the isochrone calculation.
    distance : float
        The maximum travel distance (or time) that defines the boundary of the
        isochrone.
    edge_attr : str, default "length"
        The edge attribute to use as the cost of travel (e.g., 'length',
        'travel_time').

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing a single Polygon geometry that represents the
        isochrone.

    Examples
    --------
    >>> import networkx as nx
    >>> from shapely.geometry import Point
    >>> # Create a graph
    >>> G = nx.Graph(crs="EPSG:32633")
    >>> G.add_node(0, pos=(0, 0))
    >>> G.add_node(1, pos=(10, 0))
    >>> G.add_node(2, pos=(0, 10))
    >>> G.add_edge(0, 1, length=10)
    >>> G.add_edge(0, 2, length=10)
    >>> # Create an isochrone
    >>> center = Point(0, 0)
    >>> isochrone = create_isochrone(G, center, distance=12)
    >>> print(isochrone.geometry.iloc[0].wkt)
    POLYGON ((0 0, 10 0, 0 10, 0 0))
    """
    analyzer = GraphAnalyzer()
    return analyzer.create_isochrone(graph, center_point, distance, edge_attr)

def create_tessellation(
    geometry: gpd.GeoDataFrame | gpd.GeoSeries,
    primary_barriers: gpd.GeoDataFrame | gpd.GeoSeries | None = None,
    shrink: float = 0.4,
    segment: float = 0.5,
    threshold: float = 0.05,
    n_jobs: int = -1,
    **kwargs: object,
) -> gpd.GeoDataFrame:
    """Create tessellations from given geometries, with optional barriers.

    This function generates either morphological or enclosed tessellations based on
    the input geometries. If `primary_barriers` are provided, it creates an
    enclosed tessellation; otherwise, it generates a morphological tessellation.

    Parameters
    ----------
    geometry : geopandas.GeoDataFrame or geopandas.GeoSeries
        The geometries (typically building footprints) to tessellate around.
    primary_barriers : geopandas.GeoDataFrame or geopandas.GeoSeries, optional
        Geometries (typically road network) to use as barriers for enclosed
        tessellation. If provided, `momepy.enclosed_tessellation` is used.
        Default is None.
    shrink : float, default 0.4
        The distance to shrink the geometry for the skeleton endpoint generation.
        Passed to `momepy.morphological_tessellation` or `momepy.enclosed_tessellation`.
    segment : float, default 0.5
        The segment length for discretizing the geometry. Passed to
        `momepy.morphological_tessellation` or `momepy.enclosed_tessellation`.
    threshold : float, default 0.05
        The threshold for snapping skeleton endpoints to the boundary. Only used
        for enclosed tessellation.
    n_jobs : int, default -1
        The number of jobs to use for parallel processing. -1 means using all
        available processors. Only used for enclosed tessellation.
    **kwargs : object, optional
        Additional keyword arguments passed to the underlying `momepy`
        tessellation function.

    Returns
    -------
    geopandas.GeoDataFrame
        A GeoDataFrame containing the tessellation cells as polygons. Each cell
        has a unique `tess_id`.

    Raises
    ------
    ValueError
        If `primary_barriers` are not provided and the geometry is in a
        geographic CRS (e.g., EPSG:4326), as morphological tessellation
        requires a projected CRS.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Polygon
    >>> # Create some building footprints
    >>> buildings = gpd.GeoDataFrame(
    ...     geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    ...               Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])],
    ...     crs="EPSG:32633"
    ... )
    >>> # Generate morphological tessellation
    >>> tessellation = create_tessellation(buildings)
    >>> print(tessellation.head())

    >>> # Generate enclosed tessellation with roads as barriers
    >>> from shapely.geometry import LineString
    >>> roads = gpd.GeoDataFrame(
    ...     geometry=[LineString([(0, -1), (3, -1)]), LineString([(1.5, -1), (1.5, 4)])],
    ...     crs="EPSG:32633"
    ... )
    >>> enclosed_tess = create_tessellation(buildings, primary_barriers=roads)
    >>> print(enclosed_tess.head())
    """
    if geometry.empty:
        if primary_barriers is not None:
            # Enclosed tessellation needs 'enclosure_index' column
            return gpd.GeoDataFrame(
                columns=["geometry", "enclosure_index"],
                geometry="geometry",
                crs=geometry.crs,
            )
        return gpd.GeoDataFrame(geometry=[], crs=geometry.crs)

    if primary_barriers is not None:
        # Enclosed tessellation
        enclosures = momepy.enclosures(
            primary_barriers=primary_barriers,
            limit=None,
            additional_barriers=None,
            enclosure_id="eID",
            clip=False,
        )

        if not enclosures.empty:
            try:
                tessellation = momepy.enclosed_tessellation(
                    geometry=geometry,
                    enclosures=enclosures,
                    shrink=shrink,
                    segment=segment,
                    threshold=threshold,
                    n_jobs=n_jobs,
                    **kwargs,
                )
            except ValueError as e:
                if "No objects to concatenate" in str(e):
                    logger.warning("Momepy could not generate tessellation, returning empty GeoDataFrame.")
                    return gpd.GeoDataFrame(
                        columns=["geometry", "enclosure_index", "tess_id"],
                        geometry="geometry",
                        crs=geometry.crs,
                    )
        else:
            tessellation = gpd.GeoDataFrame(
                columns=["geometry", "enclosure_index"],
                geometry="geometry",
                crs=geometry.crs,
            )

        if tessellation.empty:
            return gpd.GeoDataFrame(
                columns=["geometry", "enclosure_index"],
                geometry="geometry",
                crs=geometry.crs,
            )

        tessellation["tess_id"] = [
            f"{i}_{j}"
            for i, j in zip(tessellation["enclosure_index"], tessellation.index, strict=False)
        ]
        tessellation = tessellation.reset_index(drop=True)
    else:
        # Morphological tessellation
        tessellation = momepy.morphological_tessellation(
            geometry=geometry, clip="bounding_box", shrink=shrink, segment=segment,
        )

        tessellation["tess_id"] = tessellation.index

    return tessellation

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_gdf(
    nodes_gdf: gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None = None,
    edges_gdf: gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None = None,
    allow_empty: bool = True,
) -> tuple[
    gpd.GeoDataFrame | dict[str, gpd.GeoDataFrame] | None,
    gpd.GeoDataFrame | dict[tuple[str, str, str], gpd.GeoDataFrame] | None,
    bool,
]:
    """Validate node and edge GeoDataFrames with type detection.

    This function validates both homogeneous and heterogeneous GeoDataFrame inputs,
    performs type checking, and determines whether the input represents a
    heterogeneous graph structure.

    Parameters
    ----------
    nodes_gdf : geopandas.GeoDataFrame or dict[str, geopandas.GeoDataFrame], optional
        The GeoDataFrame containing node data to validate, or a dictionary mapping
        node type names to GeoDataFrames for heterogeneous graphs.
    edges_gdf : geopandas.GeoDataFrame or dict[tuple[str, str, str], geopandas.GeoDataFrame], optional
        The GeoDataFrame containing edge data to validate, or a dictionary mapping
        edge type tuples to GeoDataFrames for heterogeneous graphs.
    allow_empty : bool, default True
        If True, allows the GeoDataFrames to be empty. If False, raises an error.

    Returns
    -------
    tuple[geopandas.GeoDataFrame | dict[str, geopandas.GeoDataFrame] | None,
          geopandas.GeoDataFrame | dict[tuple[str, str, str], geopandas.GeoDataFrame] | None,
          bool]
        A tuple containing:
        - validated nodes_gdf (same type as input)
        - validated edges_gdf (same type as input)
        - is_hetero: boolean indicating if this is a heterogeneous graph

    Raises
    ------
    TypeError
        If an input is not a GeoDataFrame or appropriate dictionary type.
    ValueError
        If the input types are inconsistent or invalid.

    Examples
    --------
    >>> import geopandas as gpd
    >>> from shapely.geometry import Point, LineString
    >>> nodes = gpd.GeoDataFrame(geometry=[Point(0, 0)])
    >>> edges = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1)])])
    >>> try:
    ...     validated_nodes, validated_edges, is_hetero = validate_gdf(nodes, edges)
    ...     print(f"Validation successful. Heterogeneous: {is_hetero}")
    ... except (TypeError, ValueError) as e:
    ...     print(f"Validation failed: {e}")
    Validation successful. Heterogeneous: False
    """
    processor = GeoDataProcessor()

    # Type detection and validation
    is_nodes_dict = isinstance(nodes_gdf, dict)
    is_edges_dict = isinstance(edges_gdf, dict)

    # Check for type consistency
    if is_nodes_dict and edges_gdf is not None and not is_edges_dict:
        msg = "If nodes is a dict, edges must also be a dict or None."
        raise TypeError(msg)
    if not is_nodes_dict and is_edges_dict and nodes_gdf is not None:
        msg = "If edges is a dict, nodes must also be a dict or None."
        raise TypeError(msg)

    is_hetero = is_nodes_dict or is_edges_dict

    validated_nodes: dict[str, gpd.GeoDataFrame] | gpd.GeoDataFrame | None = None
    validated_edges: dict[tuple[str, str, str], gpd.GeoDataFrame] | gpd.GeoDataFrame | None = None

    if is_hetero:
        # Validate heterogeneous inputs
        if nodes_gdf is not None:
            validated_nodes = {}
            for node_type, node_gdf in nodes_gdf.items():
                if not isinstance(node_type, str):
                    msg = "Node type keys must be strings"
                    raise TypeError(msg)
                validated_nodes[node_type] = processor.validate_gdf(node_gdf, allow_empty=True)

        if edges_gdf is not None:
            validated_edges = {}
            for edge_type, edge_gdf in edges_gdf.items():
                if not isinstance(edge_type, tuple) or len(edge_type) != 3:
                    msg = "Edge type keys must be tuples of (source_type, relation_type, target_type)"
                    raise TypeError(msg)
                if not all(isinstance(t, str) for t in edge_type):
                    msg = "All elements in edge type tuples must be strings"
                    raise TypeError(msg)
                validated_edges[edge_type] = processor.validate_gdf(
                    edge_gdf, ["LineString", "MultiLineString"], allow_empty=allow_empty,
                )
    else:
        # Validate homogeneous inputs
        if nodes_gdf is not None:
            validated_nodes = processor.validate_gdf(
                nodes_gdf, allow_empty=allow_empty,
            )

        if edges_gdf is not None:
            validated_edges = processor.validate_gdf(
                edges_gdf,
                ["LineString", "MultiLineString"],
                allow_empty=allow_empty,
            )

    # Ensure CRS consistency
    all_gdfs_to_check: list[gpd.GeoDataFrame] = []
    if validated_nodes is not None:
        all_gdfs_to_check.extend(
            validated_nodes.values() if is_hetero else [validated_nodes],
        )
    if validated_edges is not None:
        all_gdfs_to_check.extend(
            validated_edges.values() if is_hetero else [validated_edges],
        )

    processor.ensure_crs_consistency(*all_gdfs_to_check)

    return validated_nodes, validated_edges, is_hetero


def validate_nx(graph: nx.Graph | nx.MultiGraph) -> None:
    """Validate a NetworkX graph with comprehensive type checking.

    Checks if the input is a NetworkX graph, ensures it is not empty
    (i.e., it has both nodes and edges), and verifies that it contains the
    necessary metadata for conversion back to GeoDataFrames or PyG objects.

    Parameters
    ----------
    graph : networkx.Graph or networkx.MultiGraph
        The NetworkX graph to validate.

    Raises
    ------
    TypeError
        If the input is not a NetworkX graph.
    ValueError
        If the graph has no nodes, no edges, or is missing essential metadata.

    Examples
    --------
    >>> import networkx as nx
    >>> from shapely.geometry import Point
    >>> G = nx.Graph(is_hetero=False, crs="EPSG:4326")
    >>> G.add_node(0, pos=(0, 0))
    >>> G.add_node(1, pos=(1, 1))
    >>> G.add_edge(0, 1)
    >>> try:
    ...     validate_nx(G)
    ...     print("Validation successful.")
    ... except (TypeError, ValueError) as e:
    ...     print(f"Validation failed: {e}")
    Validation successful.
    """
    # Type validation
    if not isinstance(graph, (nx.Graph, nx.MultiGraph)):
        msg = "Input must be a NetworkX Graph or MultiGraph"
        raise TypeError(msg)

    processor = GeoDataProcessor()
    processor.validate_nx(graph)
