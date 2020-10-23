#!/usr/bin/python3
import argparse
import os
import os.path as osp
import csv
import random
import json
import shutil

ALL_MODES = {'rgb', 'semantic', 'instance', 'depth', 'invdepth'}

def copy_example(root_dir, result_dir, nbr_houses, dataset, img_modes):
    house2camera = {}
    for cam in dataset:
        hid = cam['house_id']
        try:
            house2camera[hid].append(cam)
        except KeyError:
            house2camera[hid] = [cam]

    vox_folder = osp.join(result_dir, 'vox')
    os.mkdir(vox_folder)

    for m in img_modes:
        mode_folder = osp.join(result_dir, m)
        os.mkdir(mode_folder)

    sorted_houses = sorted(house2camera.keys())
    idx = 0
    new_dataset = []
    camera_params = []
    for hid in sorted_houses[:nbr_houses]:
        with open(osp.join(root_dir, hid, 'camera_params.json')) as f:
            cparam = json.load(f)

        for cam in house2camera[hid]:
            #Save camera params
            camera_params.append(cparam[cam['idx']])

            #Copy GT voxels
            shutil.copy(
                osp.join(root_dir, cam['house_id'], 'vox', '{:04}.npz'.format(cam['idx'])),
                osp.join(vox_folder,'{:04}.npz'.format(idx))
                )
            #Copy image modes
            for m in img_modes:
                shutil.copy(
                    osp.join(root_dir, cam['house_id'], m,'{:04}.png'.format(cam['idx'])),
                    osp.join(result_dir, m, '{:04}.png'.format(idx)))

            cam['idx'] = idx
            cam['house_id'] = '.'
            new_dataset.append(cam)
            idx += 1

    with open(osp.join(result_dir,'dataset.json'), 'w') as f:
        json.dump(new_dataset, f)

    with open(osp.join(result_dir, 'camera_params.json'), 'w') as f:
        json.dump(camera_params, f)


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Copy some data from specific dataset to example folder')
    parser.add_argument('suncg_dir', type=str, help='Path to suncg dir')
    parser.add_argument('--result-dir', type=str, default=osp.join('..','example'))
    parser.add_argument('--nbr-houses', type=int, default=1, help='Nbr houses to sample from')
    parser.add_argument('--dataset', type=str, help='Specify dataset JSON file to generate from')
    parser.add_argument('--modes', type=str, default=['rgb', 'invdepth'], nargs='+',
                        help='List of image modes to copy, supported: [{}]'.format(', '.join(ALL_MODES)))
    args = parser.parse_args()


    with open(args.dataset, 'r') as f:
        dset = json.load(f)

    shutil.rmtree(args.result_dir, ignore_errors=True)
    os.makedirs(args.result_dir)

    copy_example(args.suncg_dir, args.result_dir, args.nbr_houses, dset, args.modes)
