#!/usr/bin/env bash

REPO_PATH=$(dirname "$PWD")

export PYTHONPATH=$PYTHONPATH:$REPO_PATH

pip install imageio-ffmpeg --no-cache-dir
