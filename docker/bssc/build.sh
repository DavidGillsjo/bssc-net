#!/bin/bash
USE_NVIDIA=1 IMAGE=${IMAGE-bssc_net} ./../../libs/dockers/common/build.sh "$@"
