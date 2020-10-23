import torch
import argparse
import os.path as osp
from datetime import datetime
import time
import yaml
import os
import git

from ssc.data.loader import SUNCGDataset
from ssc.scripts.train import TrainNet, seed, dict2md
import re

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaulate net on SUNCG data')
    parser.add_argument('root_dir', type=str, help='Path to suncg data folder')
    parser.add_argument('json_file', type=str, help='Path to json file describing the dataset')
    parser.add_argument('weights', type=str, help='Path to model weights to use, can be a folder to use several weights')
    parser.add_argument('--cfg', type=str, help='Path to config file. Default: %(default)s', default=osp.join('..','cfg','eval.yaml'))
    parser.add_argument('--result-dir', type=str, help='Path to result. Default: %(default)s', default = osp.join('..','..','data','runs'))
    parser.add_argument('--cache-dir', type=str, help='Path to cache. Default: %(default)s', default = osp.join('..','..','data','_cache_'))


    args = parser.parse_args()

    #Create log dir
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    result_dir = osp.join(args.result_dir, current_time)
    os.makedirs(result_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Get config from weights
    w_cfg_folder = args.weights if osp.isdir(args.weights) else osp.dirname(args.weights)
    try:
        with open(osp.join(w_cfg_folder, 'config.yaml')) as f:
            cfg = yaml.safe_load(f)['config']
    except FileNotFoundError:
        print('Could not find weight config, running in test mode')
        cfg = {}

    # Get evaluation config
    with open(args.cfg) as f:
        eval_cfg = yaml.safe_load(f)

    #Overwrite keys present in eval config
    cfg.update(eval_cfg)

    #Set var init if not defined, does not matter during eval
    cfg.setdefault('var_init', 0.1)

    # Get git repo version
    meta_info = vars(args)
    repo = git.Repo(search_parent_directories=True)
    meta_info['version'] = repo.head.object.hexsha
    meta_info['git_diff'] = '<pre><code>{}</code></pre>'.format(repo.git.diff('--ignore-submodules'))
    del repo

    # Make sure results are reproducible
    seed()
    dset_kwargs = {k:cfg.get(k, None) for k in ['mapping']}
    dset_kwargs['mp_loader'] = cfg['num_workers'] > 0
    data = SUNCGDataset(args.json_file, args.root_dir, args.cache_dir, **dset_kwargs, val = True)
    #Use TrainNet since it already has the functionality we need.
    trainer = TrainNet(cfg, device, data.get_nbr_classes(), result_dir, val_dataset = data, meta=meta_info)

    if osp.isdir(args.weights):
        #Sort after epoch, assumes name convention '<any string>_<epoch>.tar'
        fnames = {}
        re_pattern = re.compile(r'.+_(\d+)\.tar')
        for fn in os.listdir(args.weights):
            m = re.search(re_pattern, fn)
            if m:
                fnames[int(m.group(1))] = osp.join(args.weights, fn)
        for fn_epoch in sorted(fnames):
            trainer.eval_weights(fnames[fn_epoch])
    else:
        trainer.eval_weights(args.weights)
