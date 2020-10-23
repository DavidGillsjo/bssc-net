import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os.path as osp
from loader import SUNCGDataset
from torch.utils.data import DataLoader
from ssc.visualization.mayavi_voxel import plot_voxels
from suncg import SUNCGLabels
import os
import shutil
import multiprocessing as mp
import time
import gc

def plot_data(sample, suncg_labels, plot_dir):
    base_dir = osp.join(plot_dir, '{}_{}'.format(sample['cam_house_id'], sample['cam_idx']))
    os.makedirs(base_dir, exist_ok = True)
    plot_voxels(sample['gt'].squeeze().numpy(), sample['vox_min'], sample['vox_unit'], save_path = osp.join(base_dir,'gt'), suncg_labels = suncg_labels, camera_P = data['cam_P'])
    plot_voxels(sample['visible_free'].numpy(), sample['vox_min'], sample['vox_unit'], save_path = osp.join(base_dir,'visible_free'), suncg_labels = ['occluded', 'visible_free'], camera_P = data['cam_P'])
    plot_voxels(sample['loss_mask'].numpy(), sample['vox_min'], sample['vox_unit'], save_path = osp.join(base_dir,'loss_mask'), suncg_labels = ['NA', 'loss'], camera_P = data['cam_P'])
    frustum_np = sample['frustum_mask'].numpy()
    plot_voxels(frustum_np, sample['vox_min'], sample['vox_unit'], save_path = osp.join(base_dir,'view_frustum'), suncg_labels = ['outside', 'inside'], camera_P = data['cam_P'])
    plot_voxels(frustum_np & ~sample['visible_free'].numpy(), sample['vox_min'], sample['vox_unit'], save_path = osp.join(base_dir,'occluded_frustum'), suncg_labels = ['', 'inside occluded'], camera_P = data['cam_P'])
    tsdf_np = sample['tsdf'].squeeze().numpy()
    tsdf_mask = (tsdf_np < 1) & frustum_np
    plot_voxels(tsdf_np ,sample['vox_min'], sample['vox_unit'], mask = tsdf_mask, save_path = osp.join(base_dir,'tsdf'), scalar=True, alpha = 1.0, vmin= -1, vmax = 1)
    flipped_tsdf_np = sample['flipped_tsdf'].squeeze().numpy()
    flipped_mask = (np.abs(flipped_tsdf_np) > 0) & frustum_np
    plot_voxels(flipped_tsdf_np, sample['vox_min'], sample['vox_unit'], mask = flipped_mask, save_path = osp.join(base_dir,'flipped_tsdf'), scalar=True, alpha = 1.0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test data loader and visualize data')
    parser.add_argument('root_dir', type=str, help='Path to suncg data folder')
    parser.add_argument('json_file', type=str, help='Path to json file describing the dataset')
    parser.add_argument('--cache-dir', type=str, help='Path to cache. Default: %(default)s', default = osp.join('..','..','data','_cache_'))
    parser.add_argument('--plot-dir', type=str, help='Where to put plots', default = None)
    parser.add_argument('--nbr-samples', type=int, default=None, help='Number of samples to plot. Default: %(default)s')
    parser.add_argument('--nbr-workers', type=int, default=5, help='Number of workers. Default: %(default)s')

    args = parser.parse_args()

    if args.plot_dir:
        shutil.rmtree(args.plot_dir, ignore_errors=True)
        os.makedirs(args.plot_dir, exist_ok=True)

    suncg = SUNCGDataset(args.json_file, args.root_dir, args.cache_dir, nbr_samples = args.nbr_samples)


    invalid_queue = mp.Queue()
    def test_data(idx):
        sample = suncg[idx]
        if sample is None:
            invalid_queue.put('{}_{}_None'.format(sample['cam_house_id'], sample['cam_idx']))
            return

        assert sample['gt'].squeeze().shape == sample['tsdf'].squeeze().shape, '{}_{} shape not equal ({} vs {})'.format(sample['cam_house_id'], sample['cam_idx'], sample['gt'].shape, sample['tsdf'].shape)

        if args.plot_dir:
            plot_data(sample, suncg.get_class_labels(), args.plot_dir)

    nbr_samples = len(suncg)
    pool = mp.Pool(args.nbr_workers)

    invalid_list = []
    iter_count = 0
    t = time.time()
    for i in range(nbr_samples):
        test_data(i)
    res = pool.map_async(test_data, range(nbr_samples), chunksize = 200)
    res.wait()
    pool.close()
    pool.join()
    print('took', time.time()-t, 'seconds')
    invalid_list = []
    while not invalid_queue.empty():
        invalid_list.append(invalid_queue.get())
    nbr_invalid = len(invalid_list)
    print('{} ({:.2f}%) invalid samples of {}'.format(nbr_invalid,100.0*nbr_invalid/nbr_samples, nbr_samples))
    print(invalid_list)
