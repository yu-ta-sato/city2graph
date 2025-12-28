# Utils Module

The utils module provides core utility functions for graph conversion, validation, and visualization.

| Function | Description |
| :--- | :--- |
| `gdf_to_nx` | Convert GeoDataFrames of nodes and edges to a NetworkX graph. |
| `nx_to_gdf` | Convert a NetworkX graph to GeoDataFrames for nodes and/or edges. |
| `nx_to_rx` | Convert a NetworkX graph to a rustworkx graph. |
| `rx_to_nx` | Convert a rustworkx graph to a NetworkX graph. |
| `validate_gdf` | Validate node and edge GeoDataFrames with type detection. |
| `validate_nx` | Validate a NetworkX graph with comprehensive type checking. |
| `dual_graph` | Convert a primal graph represented by nodes and edges GeoDataFrames to its dual graph. |
| `filter_graph_by_distance` | Filter a graph to include only elements within a specified threshold from a center point. |
| `create_tessellation` | Create tessellations from given geometries, with optional barriers. |
| `create_isochrone` | Generate an isochrone polygon from a graph. |
| `plot_graph` | Plot a graph with a unified interface. |

## Conversion Functions

::: city2graph.utils
    options:
      show_root_heading: false
      members:
        - gdf_to_nx
        - nx_to_gdf
        - nx_to_rx
        - rx_to_nx

## Validation Functions

::: city2graph.utils
    options:
      show_root_heading: false
      members:
        - validate_gdf
        - validate_nx

## Graph Operations

::: city2graph.utils
    options:
      show_root_heading: false
      members:
        - dual_graph
        - filter_graph_by_distance

## Spatial Operations

::: city2graph.utils
    options:
      show_root_heading: false
      members:
        - create_tessellation
        - create_isochrone

## Visualization

::: city2graph.utils
    options:
      show_root_heading: false
      members:
        - plot_graph
