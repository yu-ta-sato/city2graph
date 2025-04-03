# city2graph for conda environment

This directory contains the necessary files to build a conda environment. Actions will be made with the matrix of OS version, Python version and CUDA version or CPU. Those will be specified in `../.github/workflows/`.

## Usage

To build the conda environment, run the following script:

```bash
./build_conda.sh 3.12 cu124
```