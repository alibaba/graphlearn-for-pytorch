#!/bin/bash

set -eo pipefail

root_dir=$(pwd)
third_party_dir=${root_dir}/third_party
# googletest
echo "prepare googletest library ..."
if [ ! -d "${third_party_dir}/googletest/build" ]; then
  cd "${third_party_dir}/googletest"
  bash build.sh
fi
echo "googletest done."

# grpc
echo "prepare grpc library ..."
if [ ! -f "${third_party_dir}/grpc/build/include/grpc++/grpc++.h" ]; then
  pushd "${third_party_dir}/grpc"
  git submodule update --init grpc
  /bin/bash build.sh
  popd
fi
echo "grpc done."

# # pybind11
# echo "-- preparing pybind11 ..."
# pushd "${third_party_dir}/pybind11"
# git submodule update --init pybind11
# popd