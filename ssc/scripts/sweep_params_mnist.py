import itertools
from train_mnist import main
from sweep_params import get_cfg
import argparse
import os.path as osp
import yaml
from datetime import datetime

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sweep training parameters on MNIST')
    parser.add_argument('--base-cfg', type=str, help='Path to config file. Default: %(default)s', default=osp.join('..','cfg','train_mnist.yaml'))
    parser.add_argument('--sweep-cfg', type=str, help='Path to config file with parameters to sweep. Default: %(default)s', default=osp.join('..','cfg','sweep_mnist.yaml'))
    parser.add_argument('--result-dir', type=str, help='Path to result. Default: %(default)s', default = osp.join('..','..','data','runs','MNIST'))

    args = parser.parse_args()

    #Create log dir
    current_time = datetime.now().strftime('%b%d_%H-%M-%S_sweep')
    result_dir = osp.join(args.result_dir, current_time)

    with open(args.base_cfg) as f:
        base_cfg = yaml.safe_load(f)

    with open(args.sweep_cfg) as f:
        sweep_cfg = yaml.safe_load(f)

    for my_cfg, desc in get_cfg(sweep_cfg, base_cfg):
        main(my_cfg, osp.join(result_dir, desc))
