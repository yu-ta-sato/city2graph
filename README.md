# city2graph
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
