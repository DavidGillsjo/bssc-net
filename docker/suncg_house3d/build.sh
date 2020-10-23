#!/bin/bash
USE_NVIDIA=1 IMAGE=${IMAGE-suncg_house3d} ./../../libs/dockers/common/build.sh "$@"
