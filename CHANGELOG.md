# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 (2025-07-10)

### Changes

#### Core Features
- **Data Loading Module (`city2graph.data`)**: Comprehensive functionality for loading and processing geospatial data from various sources
  - Support for Overture Maps data integration
  - Data validation and coordinate reference system management
  - Geometric processing operations for urban network analysis
  - `load_overture_data()` and `process_overture_segments()` functions

- **Graph Conversion Module (`city2graph.graph`)**: Convert between GeoDataFrames and PyTorch Geometric objects
  - Seamless integration with Graph Neural Networks (GNNs)
  - Support for heterogeneous graph structures
  - PyTorch tensor conversion for machine learning workflows

- **Morphological Analysis Module (`city2graph.morphology`)**: Create morphological graphs from urban data
  - Private-to-private adjacency relationships between building tessellations
  - Public-to-public topological connectivity between street segments
  - Private-to-public interface relationships between private and public spaces
  - `morphological_graph()`, `private_to_private_graph()`, `private_to_public_graph()`, and `public_to_public_graph()` functions

- **Proximity Networks Module (`city2graph.proximity`)**: Generate graph networks based on spatial proximity relationships
  - Multiple proximity models (Euclidean, Manhattan, network-based distances)
  - Support for Delaunay triangulation, k-nearest neighbors, and radius-based networks
  - `bridge_nodes()` and other proximity-based graph generation functions

- **Transportation Networks Module (`city2graph.transportation`)**: Process GTFS data and create transportation networks
  - General Transit Feed Specification (GTFS) data processing
  - Public transit network representations
  - Origin-destination pair analysis
  - `get_od_pairs()`, `load_gtfs()`, and `travel_summary_graph()` functions

- **Utility Functions Module (`city2graph.utils`)**: Core utilities for graph conversion and validation
  - Graph conversion between different formats (NetworkX, GeoDataFrames, PyTorch Geometric)
  - Tessellation creation and dual graph operations
  - Distance filtering and validation utilities

#### Installation Options
- **Multiple PyTorch Installation Variants**: Support for different hardware configurations
  - Basic installation without PyTorch: `pip install city2graph`
  - CPU version: `pip install "city2graph[cpu]"`
  - CUDA support: `pip install "city2graph[cu118]"`, `pip install "city2graph[cu124]"`, `pip install "city2graph[cu126]"`, `pip install "city2graph[cu128]"`

#### Development Environment
- **Development Setup**: Comprehensive development environment using `uv`
  - Development dependencies including IPython, Jupyter, pytest, and testing tools
  - Jupyter kernel integration for interactive development
  - Pre-commit hooks and code formatting tools (isort, ruff)

- **Docker Support**: Complete Docker Compose setup
  - Jupyter notebook server with all dependencies pre-installed
  - GPU support when available
  - Mounted volumes for data and notebooks

#### Documentation and Examples
- **Comprehensive Documentation**: Detailed documentation available at https://city2graph.net
- **Example Notebooks**: Development notebook (`dev/dev.ipynb`) for testing and examples
- **API Documentation**: Complete docstring coverage for all public functions

#### Testing and Quality Assurance
- **Test Suite**: Comprehensive test coverage with pytest
  - Unit tests for all modules: `test_data.py`, `test_graph.py`, `test_morphology.py`, `test_proximity.py`, `test_transportation.py`, `test_utils.py`
  - Test data and utilities in `tests/data/` and `tests/utils/`
  - Code coverage reporting with codecov integration

- **Code Quality**:
  - Ruff linting and formatting
  - Type hints and static analysis
  - BSD-3-Clause license compliance

#### Dependencies
- **Core Dependencies**:
  - NetworkX ≥2.8 (graph operations)
  - OSMnx ≥2.0.3 (OpenStreetMap integration)
  - Shapely ≥2.1.0 (geometric operations)
  - GeoPandas >0.12.0 (geospatial data handling)
  - libpysal ≥4.12.1 (spatial analysis)
  - momepy (morphological analysis)
  - overturemaps (Overture Maps data)

- **Optional Dependencies**:
  - PyTorch ≥2.6.0 (machine learning backend)
  - PyTorch Geometric ≥2.6.1 (graph neural networks)
  - TorchVision ≥0.21.0 (computer vision utilities)

#### Platform Support
- **Python Version**: Requires Python ≥3.11, <4.0
- **Operating Systems**: macOS, Linux, Windows
- **Architecture**: CPU and GPU (CUDA) support

### Technical Details

#### Graph Types Supported
- **Morphological Graphs**: Buildings, streets, and land use relationships
- **Transportation Graphs**: Public transport networks (buses, trams, trains)
- **Proximity Graphs**: Spatial contiguity and distance-based relationships
- **Mobility Graphs**: Bike-sharing, migration, and pedestrian flow networks

#### Data Sources Integration
- **Overture Maps**: Direct integration with Overture Maps data
- **GTFS**: General Transit Feed Specification for public transport
- **OpenStreetMap**: Via OSMnx integration
- **Custom Geospatial Data**: Support for any GeoDataFrame input

#### Machine Learning Integration
- **PyTorch Geometric**: Native support for graph neural networks
- **Tensor Conversion**: Automatic conversion of geospatial data to PyTorch tensors
- **Heterogeneous Graphs**: Support for multi-type node and edge graphs

### Repository Structure
- **Main Package**: `city2graph/` - Core library modules
- **Tests**: `tests/` - Comprehensive test suite
- **Documentation**: `docs/` - Sphinx documentation source
- **Examples**: `dev/` - Development notebooks and examples
- **Docker**: `Dockerfile` and `docker-compose.yml` for containerized development

### Links
- **Documentation**: https://city2graph.net
- **PyPI Package**: https://pypi.org/project/city2graph/
- **GitHub Repository**: https://github.com/c2g-dev/city2graph
- **License**: BSD-3-Clause

---
