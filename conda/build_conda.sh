#!/bin/bash
set -e

# Default values if not set as environment variables
export PYTHON_VERSION=$1  # "3.10", "3.11", "3.12"
export PYG_VERSION=$2     # "2.5",  "2.6"
export CUDA_VERSION=$3    # "cpu",  "cu118", "cu121", "cu124"

# Export these as environment variables for meta.yaml
# CUDA will be supported in the future
export CONDA_TORCH_CONSTRAINT="torch>=2.4.0"
export CONDA_PYG_CONSTRAINT="pyg=*=*cpu*"

# Add output directory to config
CONDA_BLD_PATH=${CONDA_BLD_PATH:-"$(pwd)/conda-bld"}
mkdir -p $CONDA_BLD_PATH

# Run conda build with the specified configuration
conda build . \
  -c pyg \
  -c conda-forge \
  --output-folder $CONDA_BLD_PATH

echo "Build completed. Packages are in $CONDA_BLD_PATH"

# Clean up AFTER the build is done
rm -rf $BUILD_DIR
