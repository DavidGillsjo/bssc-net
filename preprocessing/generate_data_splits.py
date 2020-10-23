#!/usr/bin/python3

import csv
import os.path as osp
import os
import json
import random
import argparse
import numpy as np
import multiprocessing as mp
from functools import partial
import time


def read_cameras(camera_folder):
    house_list = sorted(os.listdir(camera_folder))
    cameras_nested = mp.Pool(processes=10).map(partial(_read_house_folder, camera_folder), house_list)
    #unpack and concatenate
    cameras = [cam for house_cams in cameras_nested for cam in house_cams]
    return cameras

def _read_house_folder(camera_folder, house_id):
    with open(osp.join(camera_folder, house_id, 'room_camera.txt')) as f:
        csvreader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
        cameras = [
            {'idx':      idx,
             'house_id': house_id,
             'pos':      cam[:3],
             'front':    cam[4:7],
             'up':       cam[8:11]}
                   for idx, cam in enumerate(csvreader)]

    return cameras

def write_dataset(result_dir, name, dset, nbr_mini):
    with open(osp.join(result_dir, '{}.json'.format(name)), 'w') as f:
        json.dump(dset, f)
    dset_mini = random.sample(dset, nbr_mini)
    with open(osp.join(result_dir, '{}_mini.json'.format(name)), 'w') as f:
        json.dump(dset_mini, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Create data splits from PBRS camera list')
    parser.add_argument('camera_folder', type=str, help='Path to camera folder')
    parser.add_argument('result_dir', type=str, help='Where to put the dataset files')
    parser.add_argument('--mini-size', nargs = 3, type=int, default = [2000, 1000, 1000],
                        help='Number of samples in train/val/test mini versions, default: %(default)s')
    parser.add_argument('--split', nargs = 3, type=float, default = [.7, .2, .1],
                        help='Split for train/val/test in percent, default: %(default)s')

    args = parser.parse_args()

    assert np.abs(1.0 - np.sum(args.split)) < 1e-5

    os.makedirs(args.result_dir, exist_ok=True)

    print('Reading cameras...')
    t = time.time()
    cameras = read_cameras(args.camera_folder)
    print('Took', time.time()-t, 'seconds')

    train_idx = round(args.split[0]*len(cameras))
    val_idx = train_idx + round(args.split[1]*len(cameras))

    print('Shuffling...')
    t = time.time()
    random.seed(0)
    random.shuffle(cameras)
    print('Took', time.time()-t, 'seconds')

    cameras_split = [cameras[:train_idx],
                     cameras[train_idx:val_idx],
                     cameras[val_idx:]]

    print('Writing result...')
    t = time.time()
    with mp.Pool(processes = 3) as pool:
        pool_result = []
        for name, dset, nbr_mini in zip(['train', 'val', 'test'], cameras_split, args.mini_size):
            r = pool.apply_async(write_dataset, (args.result_dir, name, dset, nbr_mini))
            pool_result.append(r)
        for r in pool_result:
            r.wait()
    print('Took', time.time()-t, 'seconds')
