#!/usr/bin/env bash

DOCKER_IMAGE=$1

docker run --gpus all -d --privileged --hostname docker -v $HOME:$HOME -e HOME=$HOME -e LOGNAME=$LOGNAME -p 8888:8888 -it $DOCKER_IMAGE
