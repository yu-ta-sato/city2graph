# city2graph

<p align="center">
  <img src="docs/source/_static/city2graph_logo_main.png" width="400" alt="city2graph logo">
</p>

**city2graph** is a Python library for converting urban geometry into graph representations, enabling advanced analysis of urban environments. For more information, please reach out to the document (https://ysato.blog/city2graph).

## Features

- Construct graphs from morphological datasets (e.g. buildings, streets, and land use)
- Construct graphs from transportation datasets (e.g. public transport of buses, trams, and trains)
- Construct graphs from contiguity datasets (e.g. land use, land cover, and administrative boundaries)
- Construct graphs from mobility datasets (e.g. bike-sharing, migration, and pedestrian flows)
- Convert geospatial data into pytorch tensors for graph representation learning, such as Graph Neural Networks (GNNs)


## Installation

### From PyPI (To be enabled)

```bash
pip install city2graph
```

**Important:** PyTorch and PyTorch Geometric do not support `conda` installations anymore. For the most reliable setup, we recommend using `pip` or `poetry` installation methods described below.

If you encounter issues with `conda`, consider creating a `conda` environment for Python only and then installing the remaining dependencies with `pip`:


### Using Poetry

```bash
# Install Poetry if you haven't already done it
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/yourusername/city2graph.git
cd city2graph

# Install dependencies using Poetry
poetry install

# Activate the virtual environment (for Poetry 2.0.0+)
poetry env activate
```

### Using Docker Compose

```bash
# Build and run in detached mode
docker-compose up -d

# Access Jupyter notebook at http://localhost:8888

# Stop containers when done
docker-compose down
```

You can customize the services in the `docker-compose.yml` file according to your needs.