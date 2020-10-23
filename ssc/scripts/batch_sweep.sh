#!/usr/bin/env bash
python3 sweep_params_mnist.py --base-cfg ../cfg/train_mnist_bayesian.yaml \
                              --sweep-cfg ../cfg/sweep.yaml

python3 sweep_params_mnist.py --base-cfg ../cfg/train_mnist_bayesian.yaml \
                              --sweep-cfg ../cfg/sweep_activation.yaml

python3 sweep_params_mnist.py --base-cfg ../cfg/train_mnist_bayesian.yaml \
                              --sweep-cfg ../cfg/sweep_beta.yaml

python3 sweep_params_mnist.py --base-cfg ../cfg/train_mnist.yaml \
                              --sweep-cfg ../cfg/sweep_wd.yaml
