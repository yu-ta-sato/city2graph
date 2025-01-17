# city2graph
=======
This repository is for constructing graphs from geospatial dataset of urban forms and functions.
=======
## Directory Structure
```
.
├── Dockerfile
├── README.md
├── city2graph
│   ├── __init__.py
│   └── tests
│       └── test_sample.py
├── docker-compose.yml
├── environment.yml
├── notebooks
│   ├── demo.ipynb
│   ├── liverpool_address.geojson
│   ├── liverpool_bathymetry.geojson
│   ├── liverpool_building.geojson
│   ├── liverpool_building_part.geojson
│   ├── liverpool_connector.geojson
│   ├── liverpool_division.geojson
│   ├── liverpool_division_area.geojson
│   ├── liverpool_division_boundary.geojson
│   ├── liverpool_infrastructure.geojson
│   ├── liverpool_land.geojson
│   ├── liverpool_land_cover.geojson
│   ├── liverpool_land_use.geojson
│   ├── liverpool_place.geojson
│   ├── liverpool_segment.geojson
│   └── liverpool_water.geojson
└── pyproject.toml
```

## Usage
1. Build the Docker image: (not required if `conda` is installed)
    ```sh
    docker build -t city2graph .
    ```

2. Run the Docker container: (not required if `conda` is installed)
    ```sh
    docker run -it --rm city2graph
    ```

3. Install dependencies using `environment.yml`:
    ```sh
    conda env create -f environment.yml
    conda activate city2graph
    ```

4. Run the application: (under dev)
    ```sh
    python -m city2graph
    ```

# Notes
## Data summary (Overture Maps)

```
types = ["address", "bathymetry", "building", "building_part", "division", "division_area", 
"division_boundary", "place", "segment", "connector", "infrastructure", "land", "land_cover", "land_use", "water"]
```

`address` : unknown (empty in the case of Liverpool)

`bathymetry` : depth of water from ETOPO GLOBathy

`building` : building footprints from OSM and Microsoft ML Buildings

`building_part` : comprementary part of building footprints? seemingly from OSM

`division` : division of jurisdiction from OSM (with tags of Wikidata)

`division_area` : area of the division as polygons from OSM and geoBoundaries

`division_boundary` : unknown (empty in the case of Liverpool)

`place` : POIs mainly from Meta
`segment` : streets from OSM

`connector` : intersections from OSM
`infrastructure` : barriers from OSM

`land` : ?

`land_cover` : ?

`land_use` : ?

`water` : ?

## Functions needed

- a function that cleans up `building_part` and merge them into `building`
- a function that partitions the `building` into plot systems (using `connector` (nodes) and `segment` (edges))
- a function that identifies the adjacency of plot systems to `segment`
- a function that constructs a networkx graph from `connector` (nodes) and `segment` (edges)
- a function that flips the networkx graph to the dual graph (`connector` (edges) and `segment` (nodes))
- a function that maps `place` as POIs onto the plot systems
- a function that maps `land_use` onto the plot systems
- a function that converts plot systems and into networkx graph

## Notes

Since the `connector` and `segment` are from OSM, osmnx can be used to create & manipulate the networkx graph
The merit of using Overture Maps is the availability of `place` (POIs) mainly from Meta
The merit of using Overture Maps is the availability of `building` both from OSM and Microsoft ML Buildings
Some of the `place` (POIs) are in the area of `segment`
`place` (POIs) are not well covered compared to OSM sometimes
JSON needs to be cleaned up by renaming attributes (in particular, it should be globally standarised for the paper 2)