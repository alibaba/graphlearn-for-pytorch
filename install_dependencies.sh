#!/bin/bash

set -eo pipefail

# libssl-dev
echo "prepare libssl-dev ..."
sudo apt-get update
sudo apt-get install -y libssl-dev
echo "libssl-dev done."

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