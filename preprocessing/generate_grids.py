#!/usr/bin/python3
# Example usage python3 render_suncg.py --min=1 --nc=1
import argparse
import os
import os.path as osp
import sys
import time
import multiprocessing as mp
import queue
import numpy as np
import cv2
import csv
import logging
from rendering import *
from utils import *
from voxel import *
import random
import json
from pywavefront import Wavefront
from ssc.data.suncg import SUNCGLabels
import shutil

from House3D import objrender, create_default_config
from House3D.objrender import Camera, RenderMode, Vec3

from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import binvox_rw
import itertools

suncg_labels = SUNCGLabels()

def image_folder_is_complete(nbr_cameras, res_dir):
    #Check for images
    for ci in range(nbr_cameras):
        for mode_str in image_modes.keys():
            # logger.info('checking {}'.format(osp.join(res_dir, "{:04}_{}.png".format(ci, mode_str))))
            if not osp.isfile(osp.join(res_dir, mode_str, "{:04}.png".format(ci))):
                return False
    return True

def voxel_folder_is_complete(nbr_cameras, res_dir):
    #Check for voxels
    for ci in range(nbr_cameras):
        if not osp.isfile(osp.join(res_dir, 'vox', "{:04}.npz".format(ci))):
            return False
    return True

def render_scene_images(cameras, api, suncg_dir, house_id, result_dir):
    mappingFile = cfg['modelCategoryFile']
    colormapFile = cfg['colorFile']
    modelBlacklistFile = cfg.get('modelBlacklistFile', None)

    #Generate folders for image types
    for mode_str in image_modes.keys():
        os.makedirs(osp.join(result_dir, mode_str), exist_ok = True)

    #Load house
    house_dir = osp.join(suncg_dir,'house',house_id)
    obj_path = osp.join(house_dir,'house.obj')
    make_tmp_obj = not osp.isfile(obj_path)
    # Generate .obj and .mtl if not done already
    if make_tmp_obj:
        logger.warning('Generating .mtl and .obj file, precompute with get_data.py for increased speed')
        gen_house_obj_mtl(house_dir)

    if modelBlacklistFile:
        api.loadSceneNoCache(obj_path,mappingFile, colormapFile, modelBlacklistFile)
    else:
        api.loadSceneNoCache(obj_path,mappingFile, colormapFile)
    cam = api.getCamera()
    # logger.debug('Setup  took: {}ms'.format(int(1e3*(time.time() - start))))
    # start = time.time()

    #Render Cameras
    camera_params = []
    K = constructK()
    for i, cr in enumerate(cameras):
        # Parse camera position
        suncg_cam = SUNCGCamera(cr)
        camera_params.append({'K':K.tolist(), 'P':suncg_cam.P.tolist()})

        pos = Vec3(*suncg_cam.pos)
        front = Vec3(*suncg_cam.front)
        up = Vec3(*suncg_cam.up)

        #Render
        cam.set(pos, front, up)
        imgs = render_images(api, '{:04}'.format(i), result_dir, image_modes)

    with open(osp.join(result_dir, 'camera_params.json'), 'w') as f:
        json.dump(camera_params, f)

    if make_tmp_obj:
        rm_house_obj_mtl(house_dir)
    logger.info('Done rendering {}'.format(house_id))

# Ported from SSC Net.
def generate_scene_voxel_grids(cameras, suncg_dir, house_id, result_dir, obj_cache):
    # debug_dir = '/data/debug2/{}'.format(house_id)
    debug_dir = None
    if debug_dir:
        shutil.rmtree(debug_dir, ignore_errors=True)
        os.makedirs(debug_dir)

    #Generate result dir
    result_dir = osp.join(result_dir, 'vox')
    os.makedirs(result_dir, exist_ok=True)

    #Load house
    house_json = osp.join(suncg_dir,'house',house_id, 'house.json')

    with open(house_json) as f:
        house = json.load(f)

    #config Params
    # voxSize = np.array([20,10,20])
    # voxUnit = 0.3
    # voxSize = np.array([240,144,240])
    # voxUnit = 0.02
    # voxSize = np.array([120,72,120])
    # voxUnit = 0.04
    # voxSize = np.array([60,40,60])
    voxSize = np.array([60,40,60])
    voxUnit = 0.08
    camK = constructK()
    im_w = 640
    im_h = 480

    # Confusing facts:
    # Camera coordinate system has as usual Z forward and Y downward
    # SUNCG coordinate system has Y facing up
    # Output coordinate system has X facing up

    # Select grid based on camera location
    for camera_idx, cr in enumerate(cameras):
        cam = SUNCGCamera(cr)

        if debug_dir:
            cam_debug_dir = osp.join(debug_dir, 'camera{}'.format(camera_idx))
            os.makedirs(cam_debug_dir, exist_ok=True)
        else:
            cam_debug_dir = None

        # Put grid center half the grid length in front of the camera, moving in the XZ-plane
        xz_front = cam.front*np.array([1,0,1])
        xz_front /= np.linalg.norm(xz_front)
        voxOriginWorld = cam.pos + xz_front*voxSize[0]*voxUnit/2

        #Correct box center so we always get some floor.
        if voxOriginWorld[1] + voxUnit/2 > voxUnit*voxSize[1]/2:
            voxOriginWorld[1] = voxUnit*voxSize[1]/2 - voxUnit/2

        voxWorldMin = voxOriginWorld - (voxSize*voxUnit/2)
        voxWorldMax = voxOriginWorld + (voxSize*voxUnit/2)


        gridPtsWorld = np.array(
            np.meshgrid(*[np.linspace(voxWorldMin[i], voxWorldMax[i], voxSize[i]) for i in range(3)], indexing='ij'))

        gridPtsWorldList = gridPtsWorld.view().reshape((3,-1))

        # Create views
        gridPtsWorldXZ = gridPtsWorldList[[0,2],:]
        gridPtsWorldXY = gridPtsWorldList[:2,:]
        gridPtsWorldYZ = gridPtsWorldList[1:,:]
        gridPtsWorldY = gridPtsWorldList[1,:]
        #Output
        gridPtsLabel = np.zeros(gridPtsWorldList.shape[1], dtype=np.uint32)

        xz = [0, 2]
        for houseLevel in house['levels']:
            for node in houseLevel['nodes']:
                if node['type'].lower() != 'room':
                    continue

                #Check if we need the room
                try:
                    bbox_min = np.array(node['bbox']['min'])
                    bbox_max = np.array(node['bbox']['max'])
                except KeyError:
                    continue

                if not boxOverlap(voxWorldMin[xz], voxWorldMax[xz], bbox_min[xz], bbox_max[xz]):
                    continue

                if not sameFloor(voxOriginWorld, bbox_min, bbox_max):
                    continue

                # Find grids in the room
                try:
                    floorObj = Wavefront(osp.join(suncg_dir, 'room', house_id, '{}f.obj'.format(node['modelId'])))
                except IOError as e:
                    floorObj = None

                if False and debug_dir:
                    floor_debug_dir = osp.join(cam_debug_dir, 'floor')
                    os.makedirs(floor_debug_dir, exist_ok=True)
                else:
                    floor_debug_dir = None

                if floorObj:
                    simple_vertices = np.array(floorObj.vertices)
                    inRoom = inPolygon(gridPtsWorldXZ.T, simple_vertices[:,[0,2]], floor_debug_dir)

                    #Early exit if we are not actually in the room.
                    if not inRoom.any():
                        continue

                    #Find floor
                    floorY = np.mean(simple_vertices[:,1])
                    floorMask = inRoom & (np.abs(gridPtsWorldY-floorY) <= voxUnit/2)
                    if 'floor' not in cfg['catBlacklist']:
                        classRootId, _ = suncg_labels.getClassRoot('floor')
                        gridPtsLabel[floorMask] = classRootId
                else:
                    #Need floor object
                    continue


                # Find ceiling
                try:
                    ceilObj = Wavefront(osp.join(suncg_dir, 'room', house_id, '{}c.obj'.format(node['modelId'])))
                except IOError as e:
                    ceilObj = None

                if ceilObj:
                    simple_vertices = np.array(ceilObj.vertices)
                    ceilY = np.mean(simple_vertices[:,1])
                    ceilMask = inRoom & (np.abs(gridPtsWorldY-ceilY) <= voxUnit/2)
                    if 'ceiling' not in cfg['catBlacklist']:
                        classRootId, _ = suncg_labels.getClassRoot('ceiling')
                        gridPtsLabel[ceilMask] = classRootId

                # Find walls
                try:
                    wallObj = Wavefront(osp.join(suncg_dir, 'room', house_id, '{}w.obj'.format(node['modelId'])))
                except IOError as e:
                    wallObj = None


                if False and debug_dir:
                    wall_debug_dir = osp.join(cam_debug_dir, 'wall')
                    os.makedirs(wall_debug_dir, exist_ok=True)
                else:
                    wall_debug_dir = None

                wallMask = np.zeros_like(inRoom)
                if wallObj and ceilObj and floorObj:
                    for m_name, material in wallObj.materials.items():
                        vertices = np.array([[material.vertices[i], material.vertices[i+2]] for i in range(5, len(material.vertices), material.vertex_size)])
                        if vertices.size > 0:
                            wallMask |= inPolygon(gridPtsWorldXZ.T, vertices, wall_debug_dir, convex_hull = True)

                    classRootId, _ = suncg_labels.getClassRoot('wall')
                    wallMask &= gridPtsWorldY < (ceilY-voxUnit/2)
                    wallMask &= gridPtsWorldY > (floorY+voxUnit/2)
                    if 'wall' not in cfg['catBlacklist']:
                        gridPtsLabel[wallMask] = classRootId

                # Visualize
                if False and debug_dir:
                    t = time.time()
                    ax_list = []
                    fig = getFigure()
                    ax = plt.subplot(2,2,1, projection='3d')
                    ax_list.append(ax)
                    plt.title('inRoom, {:0.2f}%'.format(100*np.sum(inRoom)/float(inRoom.size)))
                    ax.plot(gridPtsWorldList[0,inRoom], gridPtsWorldList[1,inRoom], gridPtsWorldList[2,inRoom], 'g.', alpha = 0.8)
                    ax.plot(gridPtsWorldList[0,~inRoom], gridPtsWorldList[1,~inRoom], gridPtsWorldList[2,~inRoom], 'r.', alpha = 0.05)
                    plt.axis('equal')
                    ax = plt.subplot(2,2,2, projection='3d')
                    ax_list.append(ax)
                    plt.title('Wall, {:0.2f}%'.format(100*np.sum(wallMask)/float(inRoom.size)))
                    ax.plot(gridPtsWorldList[0,wallMask], gridPtsWorldList[1,wallMask], gridPtsWorldList[2,wallMask], 'g.', alpha = 0.8)
                    ax.plot(gridPtsWorldList[0,~wallMask], gridPtsWorldList[1,~wallMask], gridPtsWorldList[2,~wallMask], 'r.', alpha = 0.05)
                    plt.axis('equal')
                    ax = plt.subplot(2,2,3, projection='3d')
                    ax_list.append(ax)
                    plt.title('Floor, {:0.2f}%'.format(100*np.sum(floorMask)/float(inRoom.size)))
                    ax.plot(gridPtsWorldList[0,floorMask], gridPtsWorldList[1,floorMask], gridPtsWorldList[2,floorMask], 'g.', alpha = 0.8)
                    ax.plot(gridPtsWorldList[0,~floorMask], gridPtsWorldList[1,~floorMask], gridPtsWorldList[2,~floorMask], 'r.', alpha = 0.05)
                    plt.axis('equal')
                    ax = plt.subplot(2,2,4, projection='3d')
                    ax_list.append(ax)
                    plt.title('Ceiling, {:0.2f}%'.format(100*np.sum(ceilMask)/float(inRoom.size)))
                    ax.plot(gridPtsWorldList[0,ceilMask], gridPtsWorldList[1,ceilMask], gridPtsWorldList[2,ceilMask], 'g.', alpha = 0.8)
                    ax.plot(gridPtsWorldList[0,~ceilMask], gridPtsWorldList[1,~ceilMask], gridPtsWorldList[2,~ceilMask], 'r.', alpha = 0.05)
                    plt.axis('equal')

                    anim = FuncAnimation(fig, lambda i: [ax.view_init(30, i*30) for ax in ax_list], frames=np.arange(0, 12), interval=1000)
                    anim.save(osp.join(cam_debug_dir, 'Room{}.gif'.format(node['modelId'])), dpi=80, writer='imagemagick')
                    # plt.savefig(osp.join(debug_dir, 'Room{}.png'.format(node['modelId'])))
                    plt.close()
                    print('Plots grid room', time.time()-t)

                #Check all objects
                for obj_idx in node.get('nodeIndices', []):
                    obj_node = houseLevel['nodes'][obj_idx]
                    try:
                        model_id = obj_node['modelId'].replace('/', '__')
                    except KeyError:
                        continue

                    if model_id in cfg['modelBlacklist']:
                        continue

                    classRootId, classRoot = suncg_labels.getClassRoot(model_id)

                    obj_bbox_max = np.array([obj_node['bbox']['max']]).T
                    obj_bbox_min = np.array([obj_node['bbox']['min']]).T

                    # Work with voxels around the object
                    objMask = np.all((obj_bbox_min - voxUnit/2 <= gridPtsWorldList) &
                                     (gridPtsWorldList <= obj_bbox_max + voxUnit/2),
                                     axis = 0)

                    if not objMask.any():
                        continue

                    objMaskIdx = np.flatnonzero(objMask)
                    gridPtsWorldNear = gridPtsWorldList[:, objMask]

                    # Get voxels
                    binvox_filename = osp.join(suncg_dir, 'object_vox', 'object_vox_data', model_id, '{}.binvox'.format(model_id))
                    with open(binvox_filename, 'rb') as f:
                        dims, translate, scale = binvox_rw.read_header(f)
                        assert len(set(dims)) == 1, "Voxel grid assumed to be cube"
                        #Adjust t for the standard binvox order
                        t = np.array([[translate[0], translate[2], translate[1]]]).T


                    gridPtsWorldNear_h = np.vstack([gridPtsWorldNear, np.ones([1,gridPtsWorldNear.shape[1]], dtype = gridPtsWorldNear.dtype)])
                    try:
                        T = np.linalg.inv(np.reshape(obj_node['transform'], [4,4], order='F'))
                    except TypeError as e:
                        logger.error(np.reshape(obj_node['transform'], [4,4], order='F'))
                        raise e
                    obj_coords_h = T@gridPtsWorldNear_h
                    obj_coords_h = obj_coords_h/obj_coords_h[-1]
                    obj_coords = obj_coords_h[:3]

                    #Transform world coordinates to voxel coords
                    side_len = dims[0]
                    obj_coords = side_len*(obj_coords - t)/scale

                    # If object is a window or door, clear voxels classified as wall in bbox.
                    if classRootId in (4, 5):
                        gridPtsLabel[wallMask & objMask] = 0

                    # t = time.time()
                    obj_mask = obj_cache.query(obj_coords, model_id, (np.sqrt(3)/2)*side_len*voxUnit/scale)
                    matched_indices = objMaskIdx[obj_mask]
                    # print('Simple grid search for', gridPtsWorldNear.shape[1], 'world points:', time.time()-t)

                    if matched_indices.size == 0:
                        continue

                    gridPtsLabel[matched_indices] = classRootId

                    # if debug_dir and classRoot == 'chair':
                        # rect = np.squeeze(np.array([vert for vert in itertools.product(*zip(obj_bbox_max, obj_bbox_min))]).T)
                        # fig = getFigure()
                        # ax = fig.gca(projection='3d')
                        # ax.plot(rect[0], rect[1], rect[2], '*')
                        # ax.plot(gridPtsWorldNear[0,match_mask], gridPtsWorldNear[1,match_mask], gridPtsWorldNear[2,match_mask], 'g.')
                        # ax.plot(gridPtsWorldNear[0,~match_mask], gridPtsWorldNear[1,~match_mask], gridPtsWorldNear[2,~match_mask], 'r.')
                        # # ax.plot(obj_world_coords[0], obj_world_coords[1], obj_world_coords[2], 'd', zorder=1)
                        # plt.title('Class {}, nbr_matches {}'.format(classRoot, match_mask.sum()))
                        #
                        # anim = FuncAnimation(fig, lambda i: ax.view_init(30, i*30), frames=np.arange(0, 12), interval=2000)
                        # anim.save(osp.join(cam_debug_dir, 'obj_vs_world{}.gif'.format(model_id)), dpi=fig.get_dpi(), writer='imagemagick')
                        # # plt.savefig(osp.join(debug_dir, 'obj_vs_world{}.png'.format(model_id)))
                        # plt.close()
                        #
                        # fig = getFigure()
                        # ax = fig.gca(projection='3d')
                        # vox_x, vox_y, vox_z = np.nonzero(obj_bv.data)
                        # ax.plot(obj_coords[0,match_mask], obj_coords[1,match_mask], obj_coords[2,match_mask], '.b', alpha=0.5)
                        # ax.plot(obj_coords[0,~match_mask], obj_coords[1,~match_mask], obj_coords[2,~match_mask], '.r', alpha=0.5)
                        # ax.plot(vox_x, vox_y, vox_z, '.g', alpha=0.5)
                        # plt.title('Class {}, nbr_matches {}'.format(classRoot, match_mask.sum()))
                        # # plt.savefig(osp.join(cam_debug_dir, 'obj_vs_world_nnsearch{}.png'.format(model_id)))
                        # anim = FuncAnimation(fig, lambda i: ax.view_init(30, i*30), frames=np.arange(0, 12), interval=2000)
                        # anim.save(osp.join(cam_debug_dir, 'obj_vs_world_nnsearch{}.gif'.format(model_id)), dpi=fig.get_dpi(), writer='imagemagick')
                        # plt.close()
                        #
                        # fig = getFigure()
                        # ax = fig.gca(projection='3d')
                        # ax.plot(gridPtsWorldList[0,objMask], gridPtsWorldList[1,objMask], gridPtsWorldList[2,objMask], '.g', alpha=0.5)
                        # ax.plot(gridPtsWorldList[0,~objMask], gridPtsWorldList[1,~objMask], gridPtsWorldList[2,~objMask], '.r', alpha=0.5)
                        # plt.title('Class {}, nbr_matches {}'.format(classRoot, match_mask.sum()))
                        # # plt.savefig(osp.join(cam_debug_dir, 'obj_vs_world_nnsearch{}.png'.format(model_id)))
                        # anim = FuncAnimation(fig, lambda i: ax.view_init(30, i*30), frames=np.arange(0, 12), interval=2000)
                        # anim.save(osp.join(cam_debug_dir, 'obj_vs_world{}.gif'.format(model_id)), dpi=fig.get_dpi(), writer='imagemagick')
                        # plt.close()

                        # fig = getFigure()
                        # plt.plot(nn_dist, '.-')
                        # plt.title('Class {}, threshold {}'.format(classRoot, obj_bv.scale*np.sqrt(3)/2))
                        # plt.savefig(osp.join(debug_dir, 'obj_vs_world_dist{}.png'.format(model_id)))
                        # plt.close()

        # Change coordinate axis (assume Z up) XZY ->  YZX
        gridPtsLabel3D = gridPtsLabel.reshape(voxSize)

        if debug_dir:
            plotVoxelsList(gridPtsLabel, gridPtsWorldList, suncg_labels, save_path = osp.join(debug_dir, 'camera{}.gif'.format(camera_idx)))
            # plotVoxels(gridPtsLabelYZX, suncg_labels, save_path = osp.join(debug_dir, 'camera{}.gif'.format(camera_idx)))

        np.savez_compressed(osp.join(result_dir, '{:04}.npz'.format(camera_idx)),
                            voxels=gridPtsLabel3D,
                            vox_center = voxOriginWorld,
                            vox_unit = voxUnit,
                            vox_min = voxWorldMin,
                            vox_max = voxWorldMax)

    logger.info('Done voxelizing {}'.format(house_id))

'''
Since the C code from House3D simply exits on error instead of throwing exceptions
we must treat exit of process as error during processing and invalidate that house idself.
'''
def worker(device, id_queue, suncg_dir, width, height, result_dir, invalid_houses):
    assert osp.isfile(cfg['modelCategoryFile']) and osp.isfile(cfg['colorFile'])
    #Try allocating device
    api = objrender.RenderAPI(width, height, device=device)
    # api.printContextInfo()
    NN_cache_dir = osp.join(suncg_dir, 'object_vox', 'object_vox_data', '__cache__')
    os.makedirs(NN_cache_dir, exist_ok=True)
    model_dir = osp.join(suncg_dir, 'object_vox', 'object_vox_data')
    obj_cache = NNCache(NN_cache_dir, model_dir)

    while True:
        # start = time.time()
        #Check if we have work to do
        try:
            house_id = id_queue.get_nowait()
        except queue.Empty:
            return

        #Get Cameras
        cam_path = osp.join(suncg_dir,'camera',house_id,'room_camera.txt')
        with open(cam_path) as f:
            csvreader = csv.reader(f, delimiter=' ', quoting=csv.QUOTE_NONNUMERIC)
            cameras = [c for c in csvreader]

        #Create result dir
        id_result_dir = osp.join(result_dir, house_id)
        os.makedirs(id_result_dir, exist_ok = True)

        #If already renderered, skip work
        if image_folder_is_complete(len(cameras), id_result_dir):
            logger.info('Skipping images for {}'.format(house_id))
        else:
            invalid_houses.invalidate(house_id)
            render_scene_images(cameras, api, suncg_dir, house_id, id_result_dir)

        if voxel_folder_is_complete(len(cameras), id_result_dir):
            logger.info('Skipping voxels for {}'.format(house_id))
        else:
            invalid_houses.invalidate(house_id)
            generate_scene_voxel_grids(cameras, suncg_dir, house_id, id_result_dir, obj_cache)

        invalid_houses.validate(house_id)
    #Free memory
    del api
    # logger.debug('Rendering took: {}ms'.format(int(1e3*(time.time() - start))))

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Render images from SUNCG using PBRS camera files and House3D rendering')
    parser.add_argument('suncg_dir', type=str, help='Path to suncg dir')
    parser.add_argument('--result-dir', type=str, default=None)
    parser.add_argument('--nbr-proc', type=int, default=1, help='Number of processes per GPU')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--devices', type=int, default=[0], nargs='+', help='List of GPU devices to use')
    parser.add_argument('--logfile', type=str, default=None)
    parser.add_argument('--model-blacklist', type=str, default=None, help='Path to file listing models and categories to ignore')
    parser.add_argument('--nbr-houses', type=int, default=None, help='Nbr houses to sample from')
    parser.add_argument('--datasets', type=str, nargs='*', help='Specify dataset JSON files to generate from, overrides nbr houses argument')
    parser.add_argument('--modes', type=str, default=['rgb', 'semantic', 'invdepth'], nargs='+',
                        help='List of image modes, supported: [{}]'.format(', '.join(modes_str2obj.keys())))
    args = parser.parse_args()

    # Check if voxelization exists
    voxel_dir = osp.join(args.suncg_dir, 'object_vox', 'object_vox_data')
    if not osp.exists(voxel_dir):
        print('Object voxelizations are missing in {}'.format(voxel_dir))
        sys.exit()

    global image_modes
    image_modes = mode_str2map(args.modes)

    result_dir = args.result_dir if args.result_dir else osp.join(args.suncg_dir, 'scene_comp')

    if args.datasets:
        pbrs_houses = set()
        for dset_path in args.datasets:
            with open(dset_path, 'r') as f:
                dset = json.load(f)
            dset_houses = {cam['house_id'] for cam in dset}
            pbrs_houses |= dset_houses
        pbrs_houses = sorted(pbrs_houses)
    else:
        pbrs_houses = sorted(os.listdir(osp.join(args.suncg_dir, 'camera')))

    if args.nbr_houses and args.nbr_houses < len(pbrs_houses):
        pbrs_houses = pbrs_houses[:args.nbr_houses]

    global cfg
    cfg = create_default_config('.', colormap='fine')
    if args.model_blacklist:
        cfg['modelBlacklistFile'], cfg['catBlacklist'], cfg['modelBlacklist'] = gen_blacklist(args.model_blacklist, cfg['modelCategoryFile'])
    else:
        cfg['catBlacklist'] = cfg['modelBlacklist'] = set()


    global logger
    logger = setup_mp_logger(args.logfile)

    invalid_houses = InvalidHouses(osp.join(result_dir, 'invalid.json'))
    manager, invalid_houses_proxy = invalid_houses.get_proxy()

    #Run rendering
    run_mp_house3d(devices = args.devices,
                   nbr_proc = args.nbr_proc,
                   work_list = pbrs_houses,
                   worker = worker,
                   args = (args.suncg_dir, args.width, args.height, result_dir, invalid_houses_proxy))
    invalid_houses.store()
