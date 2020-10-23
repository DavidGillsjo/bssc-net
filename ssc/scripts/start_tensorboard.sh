#!/usr/bin/env bash
docker run -d -v $(pwd)/../../data/runs:/tblog -p 6006:6006 tensorflow/tensorflow:nightly tensorboard --logdir /tblog --samples_per_plugin images=20 --bind_all
