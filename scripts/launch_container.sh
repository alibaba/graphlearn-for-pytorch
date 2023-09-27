#!/bin/bash

set -e

GLT_ROOT_DIR=$(dirname $(dirname "$(realpath "$0")"))
IMAGE_NAME=$1
JOB_NAME=$2
DESTDIR=$3

WITH_VINEYARD=${WITH_VINEYARD:-OFF}
WITH_CUDA=${WITH_CUDA:-ON}
MOUNT_VOLUME=${MOUNT_VOLUME:-TRUE}

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

docker_args="-itd --name $JOB_NAME --shm-size=1g -e WITH_VINEYARD=$WITH_VINEYARD -e WITH_CUDA=$WITH_CUDA"

if [ "$WITH_CUDA" = "ON" ]
then
  docker_args+=" --gpus all"
fi

if [ "$MOUNT_VOLUME" = "TRUE" ]
then
  # Option 1: Using -v option
  docker_args+=" -v $GLT_ROOT_DIR:$DESTDIR"
  docker run $docker_args $IMAGE_NAME bash
else
  # Option 2: Using docker cp
  docker run $docker_args $IMAGE_NAME bash
  docker cp $GLT_ROOT_DIR $JOB_NAME:$DESTDIR
fi