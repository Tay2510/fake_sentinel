#!/usr/bin/env bash

nohup jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser --notebook-dir=$HOME --NotebookApp.token='' &
