# Data Module

The data module provides functions for loading and processing geospatial data from various sources, with a focus on [Overture Maps](https://overturemaps.org/) data.

| Function | Description |
| :--- | :--- |
| `load_overture_data` | Load data from Overture Maps using the CLI tool and optionally save to GeoJSON files. |
| `process_overture_segments` | Process segments from Overture Maps to be split by connectors and extract barriers. |

## Functions

::: city2graph.data
    options:
      show_root_heading: false
      members:
        - load_overture_data
        - process_overture_segments
