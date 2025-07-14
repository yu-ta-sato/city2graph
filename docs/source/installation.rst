============
Installation
============

Using pip
----------

Standard Installation
~~~~~~~~~~~~~~~~~~~~~

The simplest way to install city2graph is via pip:

.. code-block:: bash

    pip install city2graph

This installs the core functionality without PyTorch and PyTorch Geometric.

With PyTorch (CPU)
~~~~~~~~~~~~~~~~~~

If you need the graph neural network functionality, install with the `cpu` option:

.. code-block:: bash

    pip install "city2graph[cpu]"

This will install PyTorch and PyTorch Geometric with CPU support.

With PyTorch + CUDA (GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~

For GPU acceleration, you can install city2graph with a specific CUDA version extra. For example, for CUDA 12.8:

.. code-block:: bash

    pip install "city2graph[cu128]"

Supported CUDA versions are `cu118`, `cu124`, `cu126`, and `cu128`.

Using conda-forge
------------------

Basic Installation
~~~~~~~~~~~~~~~~~~

You can also install city2graph using conda from conda-forge:

.. code-block:: bash

    conda install -c conda-forge city2graph

This installs the core functionality without PyTorch and PyTorch Geometric.

With PyTorch (CPU)
~~~~~~~~~~~~~~~~~~

To use PyTorch and PyTorch Geometric with city2graph installed from conda-forge, you need to manually add these libraries to your environment:

.. code-block:: bash

    # Install city2graph
    conda install -c conda-forge city2graph

    # Then install PyTorch and PyTorch Geometric
    conda install -c conda-forge pytorch pytorch_geometric

With PyTorch + CUDA (GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~

For GPU support, you should select the appropriate PyTorch variant by specifying the version and CUDA build string. For example, to install PyTorch 2.7.1 with CUDA 12.8 support:

.. code-block:: bash

    # Install city2graph
    conda install -c conda-forge city2graph

    # Then install PyTorch with CUDA support
    conda install -c conda-forge pytorch=2.7.1=*cuda128*
    conda install -c conda-forge pytorch_geometric

You can browse available CUDA-enabled builds on the `conda-forge PyTorch files page <https://anaconda.org/conda-forge/pytorch/files>`_ and substitute the desired version and CUDA variant in your install command. Make sure that the versions of PyTorch and PyTorch Geometric you install are compatible with each other and with your system.

.. warning::
    conda is not officially supported by PyTorch and PyTorch Geometric anymore, and only conda-forge distributions are available for them. We recommend using pip or uv for the most streamlined installation experience if you need PyTorch functionality.

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
