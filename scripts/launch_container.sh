#!/bin/bash

set -e

GLT_ROOT_DIR=$(dirname $(dirname "$(realpath "$0")"))
IMAGE_NAME=$1
JOB_NAME=$2
DESTDIR=$3

WITH_VINEYARD=${WITH_VINEYARD:-OFF}
WITH_CUDA=${WITH_CUDA:-ON}
MOUNT_VOLUME=${MOUNT_VOLUME:-TRUE}

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