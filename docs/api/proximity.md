# Proximity Module

The proximity module provides functions for generating proximity-based graph networks from spatial data. These algorithms create edges between geometries based on various spatial relationships.

| Function | Description |
| :--- | :--- |
| `knn_graph` | Generate a k-nearest neighbour graph from a GeoDataFrame of points. |
| `delaunay_graph` | Generate a Delaunay triangulation graph from a GeoDataFrame of points. |
| `gabriel_graph` | Generate a Gabriel graph from a GeoDataFrame of points. |
| `relative_neighborhood_graph` | Generate a Relative-Neighbourhood Graph (RNG) from a GeoDataFrame. |
| `euclidean_minimum_spanning_tree` | Generate a (generalised) Euclidean Minimum Spanning Tree from a GeoDataFrame of points. |
| `fixed_radius_graph` | Generate a fixed-radius graph from a GeoDataFrame of points. |
| `waxman_graph` | Generate a probabilistic Waxman graph from a GeoDataFrame of points. |
| `bridge_nodes` | Build directed proximity edges between every ordered pair of node layers. |
| `group_nodes` | Create a heterogeneous graph linking polygon zones to contained points. |
| `contiguity_graph` | Generate a contiguity-based spatial graph from polygon geometries. |

## Geometric Graph Algorithms

::: city2graph.proximity
    options:
      show_root_heading: false
      members:
        - knn_graph
        - delaunay_graph
        - gabriel_graph
        - relative_neighborhood_graph
        - euclidean_minimum_spanning_tree

## Distance-Based Graphs

::: city2graph.proximity
    options:
      show_root_heading: false
      members:
        - fixed_radius_graph
        - waxman_graph

## Node Manipulation

::: city2graph.proximity
    options:
      show_root_heading: false
      members:
        - bridge_nodes
        - group_nodes

## Contiguity Graphs

::: city2graph.proximity
    options:
      show_root_heading: false
      members:
        - contiguity_graph
