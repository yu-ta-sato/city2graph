# Transportation Module

The transportation module provides functions for processing GTFS (General Transit Feed Specification) data and creating transportation network graphs.

| Function | Description |
| :--- | :--- |
| `load_gtfs` | Parse a GTFS zip file and enrich stops/shapes with geometry. |
| `get_od_pairs` | Materialise origin-destination pairs for every trip and service day. |
| `travel_summary_graph` | Aggregate stop-to-stop travel time & frequency into an edge list. |

## GTFS Processing

::: city2graph.transportation
    options:
      show_root_heading: false
      members:
        - load_gtfs
        - get_od_pairs

## Graph Construction

::: city2graph.transportation
    options:
      show_root_heading: false
      members:
        - travel_summary_graph
