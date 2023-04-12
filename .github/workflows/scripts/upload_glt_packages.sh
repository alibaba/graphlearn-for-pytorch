#!/bin/bash

set -eo pipefail

GLT_ROOT_DIR=$(dirname $(dirname $(dirname $(dirname "$(realpath "$0")"))))
GITHUB_REF=$1
PYPI_PWD=$2

rm -rf /usr/local/cuda
ln -s /usr/local/cuda-11.7 /usr/local/cuda

cd $GLT_ROOT_DIR

if [[ "$GITHUB_REF" =~ ^"refs/tags/" ]]; then
  export GITHUB_TAG_REF="$GITHUB_REF"
fi

if [ -z "$GITHUB_TAG_REF" ]; then
  echo "Not on a tag, won't deploy to pypi"
else
  PYABIS="cp37-cp37m cp38-cp38 cp39-cp39 cp310-cp310"
  for abi in $PYABIS; do
    PYABI=$abi bash .github/workflows/scripts/build_glt.sh
  done

  echo "********"
  echo "Build packages:"
  ls graphlearn-for-pytorch/dist/
  echo "********"

  echo "[distutils]"                                 > ~/.pypirc
  echo "index-servers ="                             >> ~/.pypirc
  echo "    pypi"                                    >> ~/.pypirc
  echo "[pypi]"                                      >> ~/.pypirc
  echo "repository=https://upload.pypi.org/legacy/"  >> ~/.pypirc
  echo "username=__token__"                          >> ~/.pypirc
  echo "password=$PYPI_PWD"                          >> ~/.pypirc

  /opt/python/cp38-cp38/bin/pip install twine
  /opt/python/cp38-cp38/bin/python -m twine upload -r pypi --skip-existing graphlearn-for-pytorch/dist/*
fi
