#!/bin/bash

set -eo pipefail

GLT_ROOT_DIR=$(dirname $(dirname "$(realpath "$0")"))
CORES=$(cat < /proc/cpuinfo | grep -c "processor")

cd $GLT_ROOT_DIR

set -x

bash install_dependencies.sh
python3 -m pip install ninja parameterized

if [ -z "$WITH_CUDA" ]; then
  CUDA_OPTION=OFF
else
  CUDA_OPTION=$WITH_CUDA
fi

if [ -z "$WITH_VINEYARD" ]; then
  VINEYARD_OPTION=OFF
else
  VINEYARD_OPTION=$WITH_VINEYARD
fi

if [ "$VINEYARD_OPTION" = "ON" ]; then
  # in graphscope environment
  python3 -m pip install torch==1.13 --index-url https://download.pytorch.org/whl/cpu
  python3 -m pip install torch_geometric ogb 
  python3 -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
fi

# TODO(hongyi): build cpp with v6d
# cmake -DWITH_CUDA=$CUDA_OPTION -DWITH_VINEYARD=$VINEYARD_OPTION .
cmake -DWITH_CUDA=$CUDA_OPTION .
make -j$CORES

python3 setup.py bdist_wheel
python3 -m pip install install dist/* --force-reinstall