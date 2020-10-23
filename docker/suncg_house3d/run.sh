#!/bin/bash
#Usage: [ENV_OPTS] ./run_local [CMD] [ARGS]
USE_NVIDIA=1 IMAGE=${IMAGE-suncg_house3d} ./../../libs/dockers/common/run.sh "$@"
