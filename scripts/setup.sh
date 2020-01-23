#!/usr/bin/env bash

REPO_PATH=$(dirname "$PWD")

export PYTHONPATH=$PYTHONPATH:$REPO_PATH

pip install imageio-ffmpeg --no-cache-dir

# Extra Requirements for Kaggle cloud kernel
pip install ../libraries/facenet_pytorch-2.0.1-py3-none-any.whl
