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

# TODO(hongyi): build cpp with v6d
# cmake -DWITH_CUDA=$CUDA_OPTION -DWITH_VINEYARD=$VINEYARD_OPTION .
cmake -DWITH_CUDA=$CUDA_OPTION .
make -j$CORES

python3 setup.py bdist_wheel
python3 -m pip install dist/* --force-reinstall