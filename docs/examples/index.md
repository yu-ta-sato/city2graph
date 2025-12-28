# Examples

This section provides examples of how to use City2Graph in various urban analysis scenarios.

| Example | Description |
| :--- | :--- |
| [Constructing Metapaths](add_metapaths.ipynb) | Demonstrates the workflow of metapath construction from a heterogeneous graph. The example creates a dual graph of streets, connect amenities, and materialize metapaths between amenities via streets. |
| [Generating Graphs by Proximity](generating_graphs_by_proximity.ipynb) | Illustrates how to generate and visualize different spatial graph types based on proximity metrics (KNN, Delaunay, Gilbert, Waxman), and demonstrates how to bridge and group nodes for constructing heterogeneous graphs. |
| [Generating Graphs from OD Matrix](generating_graphs_from_od_matrix.ipynb) | Demonstrates the conversion of Origin Destination (OD) Matrix into graph, supporting both edgelist and adjacency formats. |
| [Transportation Graphs from GTFS](gtfs.ipynb) | Transforms complex transit schedules (GTFS) into intuitive graph representations for urban accessibility analysis and network visualization. |
| [Morphological Graph from OvertureMaps and OpenStreetMap](morphological_graph_from_overturemaps.ipynb) | Create morphological graphs from Overture Maps and OpenStreetMap, capturing relationships between public (streets) and private (tessellations) spaces. |

## External Links

| Title | Author | Language | Type | Release |
| :--- | :--- | :--- | :--- | :--- |
| [City2Graph: Python package for spatial network analysis and GeoAI with GNNs](https://medium.com/@yuta.sato.now/city2graph-a-python-package-for-spatial-network-analysis-and-graph-neural-networks-gnns-bc943dd6d85e) | Yuta Sato | EN | Tutorial | Sep 23, 2025 |
| [I created a Python library that converts geospatial data into graph representations for heterogeneous GNNs](https://zenn.dev/yutasato/articles/9d7994dc53d378) | Yuta Sato | JA | Tutorial | Oct 14, 2025 |

*Note*: We welcome external examples! Please submit [a pull request](https://github.com/c2g-dev/city2graph/pulls) if you have an example to share.
