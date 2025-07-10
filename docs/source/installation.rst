============
Installation
============

Standard Installation
-------------------

The simplest way to install city2graph is via pip:

.. code-block:: bash

    # Basic installation (without PyTorch)
    pip install city2graph

This installs the core functionality without PyTorch and PyTorch Geometric.

.. warning::
    Conda distributions are deprecated for PyTorch and PyTorch Geometric due to limited demand and compatibility issues. We recommend using pip or uv for the most reliable installation experience.

With PyTorch (CPU)
----------------------

If you need the graph neural network functionality, install with the `cpu` option:

.. code-block:: bash

    # Install with PyTorch and PyTorch Geometric (CPU version)
    pip install "city2graph[cpu]"

This will install PyTorch and PyTorch Geometric with CPU support.

With PyTorch + CUDA (GPU)
-----------------------------------------------

For GPU acceleration, you can install city2graph with a specific CUDA version extra. For example, for CUDA 12.8:

.. code-block:: bash

    pip install "city2graph[cu128]"

Supported CUDA versions are `cu118`, `cu124`, `cu126`, and `cu128`.

.. note::
   The core package of PyTorch Geometric (`torch_geometric`) is independent from CUDA or CPU. However, the extensions (`pyg_lib`, `torch_scatter`, etc.) are CUDA-specific and must be installed separately.

Requirements
-----------

city2graph requires the following packages:

* networkx
* shapely
* geopandas
* libpysal
* momepy
* overturemaps

For graph neural network functionality, you'll also need:

* torch
* torch_geometric
