# Graph Module

The graph module provides functions for converting between different graph representations, including GeoDataFrames, NetworkX graphs, and PyTorch Geometric data objects.

| Function | Description |
| :--- | :--- |
| `gdf_to_pyg` | Convert GeoDataFrames (nodes/edges) to a PyTorch Geometric object. |
| `nx_to_pyg` | Convert NetworkX graph to PyTorch Geometric object. |
| `pyg_to_gdf` | Convert PyTorch Geometric data to GeoDataFrames. |
| `pyg_to_nx` | Convert a PyTorch Geometric object to a NetworkX graph. |
| `is_torch_available` | Check if PyTorch Geometric is available. |
| `validate_pyg` | Validate PyTorch Geometric Data or HeteroData objects and return metadata. |

## Conversion Functions

::: city2graph.graph
    options:
      show_root_heading: false
      members:
        - gdf_to_pyg
        - nx_to_pyg
        - pyg_to_gdf
        - pyg_to_nx

## Validation Functions

::: city2graph.graph
    options:
      show_root_heading: false
      members:
        - is_torch_available
        - validate_pyg
