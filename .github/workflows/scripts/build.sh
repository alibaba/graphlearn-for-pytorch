#!/bin/bash

set -eo pipefail

GLT_ROOT_DIR=$(dirname $(dirname $(dirname $(dirname "$(realpath "$0")"))))
GITHUB_REF=$1

rm -rf /usr/local/cuda
ln -s /usr/local/cuda-12.1 /usr/local/cuda

cd $GLT_ROOT_DIR

if [[ "$GITHUB_REF" =~ ^"refs/tags/" ]]; then
  export GITHUB_TAG_REF="$GITHUB_REF"
fi

if [ -z "$GITHUB_TAG_REF" ]; then
  echo "Not on a tag, won't deploy to pypi"
else
  PYABIS="cp38-cp38 cp39-cp39 cp310-cp310 cp311-cp311"
  for abi in $PYABIS; do
    PYABI=$abi bash .github/workflows/scripts/build_wheel.sh
  done
fi
