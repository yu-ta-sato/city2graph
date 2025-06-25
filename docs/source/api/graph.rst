=====
Graph
=====

The graph module provides comprehensive functionality for converting spatial data (GeoDataFrames)
into PyTorch Geometric graph objects, supporting both homogeneous and heterogeneous graphs.
It handles bidirectional conversion between GeoDataFrames, NetworkX graphs, and PyTorch Geometric objects.

.. currentmodule:: city2graph.graph

.. autofunction:: gdf_to_pyg

.. autofunction:: nx_to_pyg

.. autofunction:: pyg_to_gdf

.. autofunction:: pyg_to_nx

.. autofunction:: is_torch_available