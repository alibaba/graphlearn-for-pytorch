#!/bin/bash

set -e

GLT_ROOT_DIR=$(dirname $(dirname "$(realpath "$0")"))
IMAGE_NAME=$1
JOB_NAME=$2
DESTDIR=$3

WITH_VINEYARD=${WITH_VINEYARD:-OFF}
WITH_CUDA=${WITH_CUDA:-ON}

ret=0
for i in $(seq 1 3); do
  [ $i -gt 1 ] && echo "WARNING: pull image failed, will retry in $((i-1)) times later" && sleep 10
  ret=0
  docker pull $IMAGE_NAME && break || ret=$?
done

if [ $ret -ne 0 ]
then
  echo "ERROR: Pull Image $IMAGE_NAME failed, exit."
  exit $ret
fi

docker_args="-itd --name $JOB_NAME --shm-size=1g -e WITH_VINEYARD=$WITH_VINEYARD -e WITH_CUDA=$WITH_CUDA -v $GLT_ROOT_DIR:$DESTDIR"

if [ "$WITH_CUDA" = "ON" ]
then
  docker_args+=" --gpus all"
fi

docker run $docker_args $IMAGE_NAME bash
