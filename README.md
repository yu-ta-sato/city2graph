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

### From PyPI

The simplest way to install city2graph is via pip:

```bash
# Basic installation (without PyTorch)
pip install city2graph
```

This installs the core functionality without PyTorch and PyTorch Geometric. It's suitable for basic graph operations with networkx, shapely, geopandas, etc.

### With PyTorch (Optional)

If you need the graph neural network functionality, install with the torch option:

```bash
# Install with PyTorch and PyTorch Geometric (CPU version)
pip install "city2graph[torch]"
```

This will install PyTorch and PyTorch Geometric with CPU support, suitable for development and small-scale processing.

### With Specific CUDA Version (for GPU acceleration)

For GPU acceleration with a specific CUDA version, we recommend installing PyTorch and PyTorch Geometric separately before installing city2graph:

```bash
# Step 1: Install PyTorch with your desired CUDA version
# Visit https://pytorch.org/get-started/locally/ and select your preferences
# Example for PyTorch 2.4.0 with CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Step 2: Install PyTorch Geometric and its CUDA dependencies
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Step 3: Now install city2graph (it will detect the pre-installed PyTorch)
pip install city2graph
```

For macOS users with Apple Silicon:

```bash
# PyTorch for macOS uses MPS (Metal Performance Shaders) instead of CUDA
pip install torch torchvision torchaudio
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install city2graph
```

**Important:** Due to the low demand, `conda` distributions are deprecated for PyTorch and PyTorch Geometric. For the most reliable setup, we recommend using pip or Poetry as described above.

### Using Poetry

```bash
# Install Poetry if you haven't already done it
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/yu-ta-sato/city2graph.git
cd city2graph

# Install core dependencies (without PyTorch)
poetry install --without torch torch-cuda

# For PyTorch with CUDA support:
# 1. First install PyTorch with your specific CUDA version
# (outside of Poetry, in your system Python)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Then install PyTorch Geometric with matching CUDA dependencies
pip install torch-geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# 3. Now install the remaining dependencies with Poetry
poetry install

# Alternatively, for CPU-only PyTorch:
poetry install --with torch

# Install with development dependencies (includes ipython, jupyter, notebook, and code formatting tools)
poetry install --with dev

# Install with documentation dependencies
poetry install --with docs

# Install with both development and documentation dependencies
poetry install --with dev,docs

# Activate the virtual environment
poetry shell

# Start IPython for interactive development
poetry run ipython

# Start Jupyter Notebook
poetry run jupyter notebook
```

#### Development Environment

The development dependencies include:
- `ipython`: Enhanced interactive Python shell
- `jupyter` and `notebook`: For running Jupyter notebooks
- `black` and `isort`: Code formatting tools
- `pytest` and `pytest-cov`: Testing tools

These tools help streamline development and maintain code quality.

### Using Docker Compose

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