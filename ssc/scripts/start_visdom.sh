#!/usr/bin/env bash
sudo docker run -d -v $(pwd)/../data/visdom:/visdomdata -p 8097:8097 -e ENV_PATH=/visdomdata -e READONLY=False hypnosapos/visdom:latest
