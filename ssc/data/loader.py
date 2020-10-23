
import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import json
import logging

import ssc.data._init_tsdf
from ssc.data.suncg_mapping import SUNCGMapping
import fusion
from zipfile import BadZipFile
import hashlib
from numpy.linalg import inv
import math
from preprocessing.utils import InvalidHouses
import multiprocessing as mp


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def rotate_around_vector(v, phi):
    W = np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ],
    ])
    R = np.eye(3) + np.sin(phi)*W + (2*np.sin(phi/2)**2)*W**2
    return R

class SUNCGDataset(Dataset):
    """
    SUNCG dataset with depthimages and voxel GT.
    Uses multiprocessing manager to make sure memory is shared for large objects.
    """

    def __init__(self, json_file, root_dir, cache_dir, mapping = None, val = False, nbr_samples = None, mp_loader = True):
        """
        Args:
            json_file (string): Path to the json file with cameras.
            root_dir (string): Directory with all the data, so root_dir/house_id/[voxels,depth,rgb].
            transform (callable, optional): Optional transform to be applied
                on a sample.
            mp_loader: If Loader will use multiprocessing.
        """
        invalid_houses = InvalidHouses(osp.join(root_dir, 'invalid.json'))
        with open(json_file) as f:
            cameras = json.load(f)
        cameras = np.array([c for c in cameras if invalid_houses.is_valid(c['house_id']) ])
        if nbr_samples and nbr_samples < len(cameras):
            cameras = cameras[:nbr_samples]

        if mp_loader:
            # Make cameras and mapping shared since they are large and accessed every iteration.
            self.mapping_manager, self.label_mapping = SUNCGMapping.create_proxy(mapping)
            self.camera_manager = mp.Manager()
            self.cameras = self.camera_manager.list(cameras)
        else:
            self.label_mapping = SUNCGMapping(mapping)
            self.cameras = cameras

        self.root_dir = root_dir
        #Setup cache dir
        abs_root = osp.abspath(root_dir)
        cache_name = osp.split(abs_root)[-1] + '_' + hashlib.sha1(abs_root.encode('utf-8')).hexdigest()
        self.cache_dir = osp.join(cache_dir, cache_name)
        self.val = val
        self.nbr_tsdf_hist_bins = 4

    def __len__(self):
        return len(self.cameras)

    def _generate_tsdf(self, camera, voxel_bounds, voxel_resolution):
        # Initialize voxel volume
        with HiddenPrints():
            tsdf_vol = fusion.TSDFVolume(voxel_bounds, voxel_size=voxel_resolution, use_gpu = False)

        # Read depth image
        depth_name = osp.join(self.root_dir, camera['house_id'], 'invdepth', '{:04}.png'.format(camera['idx']))
        invdepth_img = cv2.imread(depth_name, cv2.IMREAD_COLOR)
        invdepth_img = invdepth_img[:,:,::-1].astype(np.uint16)
        idepth = invdepth_img[:, :, 0] * 256 + invdepth_img[:, :, 1]
        PIXEL_MAX = np.iinfo(np.uint16).max
        minDepth = 0.3
        depth = minDepth * PIXEL_MAX / idepth.astype(np.float)
        # Dont trust the extreme values
        valid_depth = (idepth != PIXEL_MAX) & (idepth != 0)
        #Cap at 20 meters
        valid_depth &= depth < 20
        #Set invalid depth to 0
        depth[~valid_depth] = 0

        #Read RGB image
        rgb_name = osp.join(self.root_dir, camera['house_id'], 'rgb', '{:04}.png'.format(camera['idx']))
        bgr_img = cv2.imread(rgb_name, cv2.IMREAD_COLOR)
        rgb_img = bgr_img[:,:,::-1]

        # Integrate observation into voxel volume (assume color aligned with depth)
        P_4x4 = np.vstack([camera['P'], np.array([0,0,0,1])])
        tsdf_vol.integrate(rgb_img, depth, camera['K'], np.linalg.inv(P_4x4))
        result = {}
        result['tsdf'], _ = tsdf_vol.get_volume()
        result['flipped_tsdf'] = np.sign(result['tsdf']) - result['tsdf'] #dmax = 1
        result['frustum_mask'] = tsdf_vol.get_frustum_mask()
        result['visible_free'] = result['frustum_mask'] & (result['tsdf'] > voxel_resolution/tsdf_vol._trunc_margin)
        result['occluded_mask'] = result['frustum_mask'] & (result['tsdf'] <= voxel_resolution/tsdf_vol._trunc_margin)
        result['tsdf_trunc_margin'] = tsdf_vol._trunc_margin

        return result


    def _get_voxel_params(self, camera, voxel_npz = None):
        if not voxel_npz:
            raise NotImplementedError('Relies on gt voxel size')

        vox_dim = voxel_npz['voxels'].shape
        vox_resolution = voxel_npz['vox_unit']
        vox_world_min = voxel_npz['vox_min']
        vox_world_max = voxel_npz['vox_max'] - vox_resolution/2.0 #For ceil in tsdf

        vox_bounds = np.vstack([vox_world_min, vox_world_max]).T

        return vox_bounds,vox_resolution


    def __getitem__(self, idx):
        camera = self.cameras[idx]

        #Get camera params
        try:
            with open(osp.join(self.root_dir, camera['house_id'], 'camera_params.json'), 'r') as f:
                camera_params = json.load(f)
        except FileNotFoundError as e:
            logging.warning(e)
            return None

        for param, value in camera_params[camera['idx']].items():
            camera[param] = np.array(value)

        # Read GT voxel
        vox_name = osp.join(self.root_dir, camera['house_id'], 'vox', '{:04}.npz'.format(camera['idx']))
        try:
            gt_npz = np.load(vox_name)
        except FileNotFoundError as e:
            logging.warning(e)
            return None

        tsdf_cache_dir = osp.join(self.cache_dir, 'tsdf', camera['house_id'])
        os.makedirs(tsdf_cache_dir, exist_ok = True)
        tsdf_cache_name = osp.join(tsdf_cache_dir, '{:04}.npz'.format(camera['idx']))

        try:
            tsdf_result = np.load(tsdf_cache_name)
            if len(tsdf_result) < 6:
                raise FileNotFoundError
        except (FileNotFoundError, BadZipFile):
            #Make to voxel volume
            vox_bounds,vox_resolution = self._get_voxel_params(camera, gt_npz)
            tsdf_result = self._generate_tsdf(camera, vox_bounds,vox_resolution)
            np.savez_compressed(tsdf_cache_name, **tsdf_result)

        assert gt_npz['voxels'].shape == tsdf_result['tsdf'].shape, 'Ground truth and generated tsdf dimensions does not match'

        sample = {}
        mapped_voxels = self.label_mapping.map(gt_npz['voxels'], dtype = np.int32)
        sample['gt'] = torch.tensor(mapped_voxels, dtype=torch.long)  #Add channel dimension
        for key, data in tsdf_result.items():
            sample[key] = torch.tensor(data[None, ...]) if 'tsdf' in key else torch.tensor(data)
        for key in ['pos', 'up', 'front', 'house_id', 'idx', 'P', 'K']:
            sample['cam_{}'.format(key)] = camera[key]
        for key in ['vox_center', 'vox_min', 'vox_max', 'vox_unit']:
            sample[key] = gt_npz[key]

        #Create sampling mask for data balancing
        sample_mask = (mapped_voxels > 0) & tsdf_result['frustum_mask']
        sample_mask_flat = sample_mask.ravel()
        nbr_occupied = sample_mask.sum()
        all_occluded_empty_index = np.flatnonzero((mapped_voxels == 0) & tsdf_result['frustum_mask'] & ~tsdf_result['visible_free'])
        if len(all_occluded_empty_index) > 2*nbr_occupied:
            occluded_empty_index = np.random.choice(all_occluded_empty_index, size=2*nbr_occupied, replace=False)
        else:
            occluded_empty_index = all_occluded_empty_index
        sample_mask_flat[occluded_empty_index] = True
        sample['loss_mask'] = torch.tensor(sample_mask)

        #Generate TSDF bins and corresponding masks
        if self.val:
            bins = np.linspace(0, 1.0, self.nbr_tsdf_hist_bins)
            bins = torch.tensor([*bins, float('Inf')])
            abs_tsdf = torch.abs(sample['tsdf']).squeeze()
            hist_masks = torch.zeros((self.nbr_tsdf_hist_bins, *abs_tsdf.shape), dtype=torch.bool)
            for i, (lower_bound, upper_bound) in enumerate(zip(bins[:-1], bins[1:])):
                hist_masks[i] = sample['occluded_mask'] & (lower_bound <= abs_tsdf) & (abs_tsdf < upper_bound)
            sample['tsdf_hist_bins'] = bins*sample['tsdf_trunc_margin'] #Convert to actual distance
            sample['tsdf_hist_masks'] = hist_masks


            #Verify mask
            test_coverage = sample['occluded_mask'].clone()
            test_disjunct = torch.ones_like(test_coverage)
            for hm in hist_masks:
                test_coverage &= ~hm
                test_disjunct &= hm
            assert not test_coverage.any(), "Hist masks does not cover occluded_mask"
            assert not test_disjunct.any(), "Hist masks contains overlap"


        return sample

    def get_class_labels(self):
        return self.label_mapping.get_classes()

    def get_class_id(self, name):
        return self.label_mapping.get_class_id(name)

    def get_nbr_classes(self):
        return self.label_mapping.get_nbr_classes()
