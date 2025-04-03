#!/bin/bash
set -e

# Default values if not set as environment variables
export PYTHON_VERSION=$1  # "3.10", "3.11", "3.12"
export PYG_VERSION=$2     # "2.5",  "2.6"
export CUDA_VERSION=$3    # "cpu",  "cu118", "cu121", "cu124"

# Export these as environment variables for meta.yaml
export CONDA_PYG_CONSTRAINT="pyg=${PYG_VERSION%.*}.*"

# Create temp directory for building
BUILD_DIR=$(mktemp -d)

# Copy recipe to build directory
cp -r $(dirname "$0")/* $BUILD_DIR/

# Add output directory to config
CONDA_BLD_PATH=${CONDA_BLD_PATH:-"$(pwd)/conda-bld"}
mkdir -p $CONDA_BLD_PATH

# Run conda build with the specified configuration
conda build $BUILD_DIR \
  --python=$PYTHON_VERSION \
  --output-folder $CONDA_BLD_PATH \
  --no-anaconda-upload \
  -c pyg \
  -c conda-forge

echo "Build completed. Packages are in $CONDA_BLD_PATH"

# Clean up AFTER the build is done
rm -rf $BUILD_DIR