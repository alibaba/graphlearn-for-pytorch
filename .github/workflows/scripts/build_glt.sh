#!/bin/bash

set -eo pipefail

GLT_ROOT_DIR=$(dirname $(dirname $(dirname $(dirname "$(realpath "$0")"))))
PYBIN=/opt/python/${PYABI}/bin

cd $GLT_ROOT_DIR

set -x

${PYBIN}/pip install scipy
${PYBIN}/pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
${PYBIN}/pip install torch_geometric
${PYBIN}/pip install --no-index  torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
${PYBIN}/pip install auditwheel
RELEASE=TRUE WITH_CUDA=ON ${PYBIN}/python setup.py bdist_wheel

# Bundle external shared libraries into the wheels
for whl in dist/*-linux*.whl; do
  ${PYBIN}/auditwheel repair "$whl" -w dist/ --plat manylinux2014_x86_64 --exclude libtorch_cpu.so --exclude libc10.so --exclude libtorch_python.so --exclude libtorch.so
done

rm dist/*-linux*.whl
