#!/usr/bin/env bash

KAGGLE_IMAGE=kaggle

docker run --gpus all -d --privileged --hostname $KAGGLE_IMAGE -v $HOME:$HOME -e HOME=$HOME -e LOGNAME=$LOGNAME -p 8888:8888 -it $KAGGLE_IMAGE
