#!/bin/bash

set -eo pipefail

GLT_ROOT_DIR=$(dirname $(dirname "$(realpath "$0")"))

# add environment variable
LD_LIBRARY_PATH=${GLT_ROOT_DIR}/built/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH

PYTHON=python

echo "Running python unit tests ..."

for file in $(ls -ld $(find $GLT_ROOT_DIR/test/python))
do
  if [[ $file == */test_*.py ]]
  then
    echo $file
    ${PYTHON} $file
    sleep 1s
  fi
done
