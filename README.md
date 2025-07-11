# city2graph

[![city2graph](http://city2graph.net/_static/social_preview.png)](http://city2graph.net/_static/social_preview.png)

**city2graph** is a Python library for converting urban geometry into graph representations, enabling advanced analysis of urban environments. For more information, please reach out to the document (https://city2graph.net).

[![PyPI Version](https://badge.fury.io/py/city2graph.svg)](https://pypi.org/project/city2graph/)
[![codecov](https://codecov.io/gh/c2g-dev/city2graph/graph/badge.svg?token=2R449G75Z0)](https://codecov.io/gh/c2g-dev/city2graph)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/c2g-dev/city2graph/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Features

[![scope](http://city2graph.net/_static/scope.png)](http://city2graph.net/_static/scope.png)


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

If you need the Graph Neural Networks functionality, install with the `cpu` option:

```bash
# Install with PyTorch and PyTorch Geometric (CPU version)
pip install "city2graph[cpu]"
```

This will install PyTorch and PyTorch Geometric with CPU support, suitable for development and small-scale processing.

### With PyTorch + CUDA (GPU)

For GPU acceleration, you can install city2graph with a specific CUDA version extra. For example, for CUDA 12.8:

```bash
# e.g., for CUDA 12.8
pip install "city2graph[cu128]"
```

Supported CUDA versions are `cu118`, `cu124`, `cu126`, and `cu128`.

**Important:** The PyTorch Geometric extensions (`pyg_lib`, `torch_scatter`, etc.) are not included and must be installed separately. Please refer to the [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for instructions. Due to the low demand, `conda` distributions are deprecated for PyTorch and PyTorch Geometric. For the most reliable setup, we recommend using pip or uv as described above.

#### For Development

If you want to contribute to city2graph, you can set up a development environment using `uv`.

```bash
# Install uv if you haven't already done it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/c2g-dev/city2graph.git
cd city2graph

# Install development dependencies with a PyTorch variant (e.g., cpu or cu128)
uv sync --extra cpu --group dev
```

You can then run commands within the managed environment:

```bash
# Add IPython kernel for interactive development
uv run ipython kernel install --name "your-env-name" --user

# Or start Jupyter Notebook
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

## Contributing

We welcome contributions to the city2graph project! To contribute:

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/<your-name>/city2graph.git
   cd city2graph
   git remote add upstream https://github.com/c2g-dev/city2graph.git
   ```

2. **Set up the development environment:**
   ```bash
   uv sync --group dev --extra cpu
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make your changes and test:**
   ```bash
   # Run pre-commit checks
   uv run pre-commit run --all-files

   # Run tests
   uv run pytest --cov=city2graph --cov-report=html --cov-report=term
   ```

5. **Submit a pull request** with a clear description of your changes.

For detailed contributing guidelines, code style requirements, and documentation standards, please see our [Contributing Guide](docs/source/contributing.rst).

## Code Quality

We maintain strict code quality standards using:
- **Ruff**: For linting and formatting
- **mypy**: For static type checking
- **numpydoc**: For docstring style validation

All contributions must pass pre-commit checks before being merged.
