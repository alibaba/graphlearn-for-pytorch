#!/bin/bash

set -eo pipefail

GLT_ROOT_DIR=$(dirname $(dirname $(dirname $(dirname "$(realpath "$0")"))))
PYBIN=/opt/python/${PYABI}/bin

cd $GLT_ROOT_DIR

set -x
RELEASE=TRUE WITH_CUDA=ON ${PYBIN}/python setup.py bdist_wheel

# Bundle external shared libraries into the wheels
for whl in dist/*-linux*.whl; do
  ${PYBIN}/auditwheel repair "$whl" -w dist/ --plat manylinux2014_x86_64 --exclude libtorch_cpu.so --exclude libc10.so --exclude libtorch_python.so --exclude libtorch.so
done

rm dist/*-linux*.whl
