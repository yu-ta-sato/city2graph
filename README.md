# city2graph

  <img src="docs/source/_static/social_preview.png" width="400" alt="city2graph logo">

**city2graph** is a Python library for converting urban geometry into graph representations, enabling advanced analysis of urban environments. For more information, please reach out to the document (https://city2graph.net).

## Features

- Construct graphs from morphological datasets (e.g. buildings, streets, and land use)
- Construct graphs from transportation datasets (e.g. public transport of buses, trams, and trains)
- Construct graphs from contiguity datasets (e.g. land use, land cover, and administrative boundaries)
- Construct graphs from mobility datasets (e.g. bike-sharing, migration, and pedestrian flows)
- Convert geospatial data into pytorch tensors for graph representation learning, such as Graph Neural Networks (GNNs)


## Installation

### Without PyTorch

The simplest way to install city2graph is via pip:

```bash
# Basic installation (without PyTorch)
pip install city2graph
```

This installs the core functionality without PyTorch and PyTorch Geometric.

### With PyTorch (CPU)

If you need the Graph Neural Networks functionality, install with the torch option:

```bash
# Install with PyTorch and PyTorch Geometric (CPU version)
pip install "city2graph[torch]"
```

This will install PyTorch and PyTorch Geometric with CPU support, suitable for development and small-scale processing.

### With PyTorch + CUDA (GPU)

For GPU acceleration with a specific CUDA version, we recommend installing PyTorch and PyTorch Geometric separately before installing city2graph:

```bash
# Step 1: Install PyTorch with your desired CUDA version (e.g. PyTorch 2.4.0 + CUDA 12.1)
pip install torch==2.4.0  --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install PyTorch Geometric and its CUDA dependencies
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Step 3: Now install city2graph (it will detect the pre-installed PyTorch)
pip install city2graph
```

**Important:** Due to the low demand, `conda` distributions are deprecated for PyTorch and PyTorch Geometric. For the most reliable setup, we recommend using pip or uv as described above.

### Using uv

#### Basic Installation (without PyTorch)

```bash
# Install uv if you haven't already done it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yu-ta-sato/city2graph.git
cd city2graph

# Install core dependencies (without PyTorch)
uv sync
```

This installs the core functionality without PyTorch and PyTorch Geometric, suitable for basic graph operations with networkx, shapely, geopandas, etc.

#### With PyTorch

```bash
uv sync --extra torch
```

This will install PyTorch and PyTorch Geometric with CPU support, suitable for development and small-scale processing.

#### With Specific CUDA Version (for GPU acceleration)

```bash
# Step 1: Install the base dependencies with torch extra
uv sync --extra torch

# Step 2: Install PyG extensions with CUDA support via pip
uv run pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

**Note:** Due to uv's current limitations with custom index URLs for PyTorch packages, PyG extensions must be installed via pip within the uv environment.

#### With Development or Documentation Dependencies

```bash
# Install with development dependencies (includes ipython, jupyter, notebook, and code formatting tools)
uv sync --group dev

# Install Jupyter kernel for the project (recommended for notebook development)
uv run ipython kernel install --user --env VIRTUAL_ENV $(pwd)/.venv --name=city2gragh-uv

# Start IPython for interactive development
uv run ipython kernel install --name "your-env-name" --user

# Start Jupyter Notebook (the city2graph kernel will be available)
uv run jupyter notebook
```

#### Development Environment

The development dependencies include:
- `ipython`: Enhanced interactive Python shell with Jupyter kernel support
- `jupyter` and `notebook`: For running Jupyter notebooks with project-specific kernel
- `isort`: Code formatting tools
- `pytest` and `pytest-cov`: Testing tools

The Jupyter kernel installation ensures that when you start Jupyter notebooks, you can select the "city2graph" kernel which has access to all your project dependencies in the correct virtual environment.

## Using Docker Compose

Before using Docker Compose, ensure you have Docker and Docker Compose installed on your system:

```bash
# Check Docker installation
docker --version

# Check Docker Compose installation
docker compose version
```

If these commands don't work, you need to install Docker first:
- For macOS: Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- For Linux: Follow the [installation instructions](https://docs.docker.com/engine/install/) for your specific distribution
- For Windows: Install [Docker Desktop](https://www.docker.com/products/docker-desktop)

Once Docker is installed, clone the repository and start the containers:

```bash
# Clone the repository
git clone https://github.com/yu-ta-sato/city2graph.git
cd city2graph

# Build and run in detached mode
docker compose up -d

# Access Jupyter notebook at http://localhost:8888

# Stop containers when done
docker compose down
```

You can customize the services in the `docker-compose.yml` file according to your needs.

#### Docker Compose Configuration

The project includes a `docker-compose.yml` file that sets up:
- A Jupyter notebook server with all dependencies pre-installed
- GPU support if available on your system
- Mounted volumes for your data and notebooks