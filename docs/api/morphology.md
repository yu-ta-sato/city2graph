# Morphology Module

The morphology module provides functions for creating morphological graphs from urban form data, capturing spatial relationships between buildings, streets, and public spaces.

| Function | Description |
| :--- | :--- |
| `morphological_graph` | Create a morphological graph from buildings and street segments. |
| `private_to_private_graph` | Create edges between contiguous private polygons based on spatial adjacency. |
| `private_to_public_graph` | Create edges between private polygons and nearby public geometries. |
| `public_to_public_graph` | Create edges between connected public segments based on topological connectivity. |
| `segments_to_graph` | Convert a GeoDataFrame of LineString segments into a graph structure. |

## Composite Graphs

::: city2graph.morphology
    options:
      show_root_heading: false
      members:
        - morphological_graph

## Component Graphs

::: city2graph.morphology
    options:
      show_root_heading: false
      members:
        - private_to_private_graph
        - private_to_public_graph
        - public_to_public_graph
        - segments_to_graph
