#!/bin/bash

set -e

GLT_ROOT_DIR=$(dirname $(dirname "$(realpath "$0")"))
CORES=$(cat < /proc/cpuinfo | grep -c "processor")

cd $GLT_ROOT_DIR

set -x

sh install_dependencies.sh
cmake .
make -j$CORES
python setup.py bdist_wheel
pip install dist/* --force-reinstall
