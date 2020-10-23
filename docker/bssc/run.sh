#!/bin/bash
#Usage: [ENV_OPTS] ./run_local [CMD] [ARGS]
USE_NVIDIA=1 IMAGE=${IMAGE-bssc_net} ./../../libs/dockers/common/run.sh "$@"
