import itertools
from train import main
import argparse
import os.path as osp
import yaml
from datetime import datetime

#Assume keys in sweep_cfg are parameter names and the values are lists of options.
def get_cfg(sweep_cfg, cfg):
    params = sorted(sweep_cfg.keys())
    sweep_values = [sweep_cfg[p] for p in params]
    for vals in itertools.product(*sweep_values):
        desc = ''
        for i, name in enumerate(params):
            cfg[name] = vals[i]
            desc += '{}:{}|'.format(name, vals[i])
        yield cfg, desc.replace("'","").replace(" ","")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Sweep training parameters on SUNCG')
    parser.add_argument('root_dir', type=str, help='Path to suncg data folder')
    parser.add_argument('json_file', type=str, help='Path to json file describing the dataset')
    parser.add_argument('--val', type=str, help='Path to validation set', default=None)
    parser.add_argument('--cache-dir', type=str, help='Path to cache. Default: %(default)s', default = osp.join('..','..','data','_cache_'))
    parser.add_argument('--base-cfg', type=str, help='Path to config file. Default: %(default)s', default=osp.join('..','cfg','train_mnist.yaml'))
    parser.add_argument('--sweep-cfg', type=str, help='Path to config file with parameters to sweep. Default: %(default)s', default=osp.join('..','cfg','sweep_mnist.yaml'))
    parser.add_argument('--result-dir', type=str, help='Path to result. Default: %(default)s', default = osp.join('..','..','data','runs'))

    args = parser.parse_args()

    #Create log dir
    current_time = datetime.now().strftime('%b%d_%H-%M-%S_sweep')
    result_dir = osp.join(args.result_dir, current_time)

    with open(args.base_cfg) as f:
        base_cfg = yaml.safe_load(f)

    with open(args.sweep_cfg) as f:
        sweep_cfg = yaml.safe_load(f)

    for my_cfg, desc in get_cfg(sweep_cfg, base_cfg):
        main(args.root_dir, args.json_file, my_cfg, osp.join(result_dir, desc), args.cache_dir, val=args.val)
