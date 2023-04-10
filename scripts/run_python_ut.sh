#!/bin/sh

set -eo pipefail

GLT_ROOT_DIR=$(dirname $(dirname "$(realpath "$0")"))

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
