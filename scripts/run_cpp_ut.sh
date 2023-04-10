#!/bin/bash

set -e

GLT_ROOT_DIR=$(dirname $(dirname "$(realpath "$0")"))

echo "Running cpp unit tests ..."

pushd $GLT_ROOT_DIR/built/bin/

for i in `ls test_*`
  do ./$i
done

popd
