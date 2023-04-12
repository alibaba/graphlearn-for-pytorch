#!/bin/bash

set -eo pipefail

GLT_ROOT_DIR=$(dirname $(dirname $(dirname $(dirname "$(realpath "$0")"))))
cd $GLT_ROOT_DIR

echo "********"
echo "Upload packages:"
ls ${GLT_ROOT_DIR}/dist/
echo "********"

echo "[distutils]"                                 > ~/.pypirc
echo "index-servers ="                             >> ~/.pypirc
echo "    pypi"                                    >> ~/.pypirc
echo "[pypi]"                                      >> ~/.pypirc
echo "repository=https://upload.pypi.org/legacy/"  >> ~/.pypirc
echo "username=__token__"                          >> ~/.pypirc
echo "password=$PYPI_PWD"                          >> ~/.pypirc

pip3 install twine
python3 -m twine upload -r pypi --skip-existing ${GLT_ROOT_DIR}/dist/*
