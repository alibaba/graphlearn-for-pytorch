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