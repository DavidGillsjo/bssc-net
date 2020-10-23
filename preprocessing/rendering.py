import time
import os
import os.path as osp
import numpy as np
import cv2
import yaml
import csv
import tempfile
import subprocess

import _init_house3d
from House3D.objrender import RenderMode

SCRIPT_DIR = osp.dirname(osp.realpath(__file__))

modes_str2obj = {'rgb': RenderMode.RGB,
                 'semantic': RenderMode.SEMANTIC,
                 'instance': RenderMode.INSTANCE,
                 'depth': RenderMode.DEPTH,
                 'invdepth': RenderMode.INVDEPTH}
suncgtb_scn2scn = osp.abspath(osp.join(SCRIPT_DIR, '..','libs','SUNCGtoolbox','gaps','bin','x86_64','scn2scn'))

class SUNCGToolboxError(Exception):
    def __init__(self, returncode, message):
        self.returncode = returncode
        self.message = message

    def __str__(self):
        return self.message


def mode_str2map(str_list):
    mode_map = {}
    for mode_str in str_list:
        m_lower = mode_str.lower()
        mode_map[m_lower] = modes_str2obj[m_lower]
    return mode_map

def glvec2np(vec):
    return np.array([vec.x, vec.y, vec.z])

def render_images(api, prefix, res_dir, image_modes, return_copy = False):
    imgs = {}
    for mode_str, mode in image_modes.items():
        api.setMode(mode)
        mat = np.array(api.render(), copy=return_copy)
        if mode == RenderMode.DEPTH:
            img_basename = "{}.png".format(prefix)
            res_path = osp.join(res_dir, mode_str, img_basename)
            cv2.imwrite(res_path, mat[:,:,0])
            img_basename = "{}_inf.png".format(prefix)
            res_path = osp.join(res_dir, mode_str, img_basename)
            cv2.imwrite(res_path, mat[:,:,1])
        else:
            img_basename = "{}.png".format(prefix)
            res_path = osp.join(res_dir, mode_str, img_basename)
            cv2.imwrite(res_path, mat[:,:,::-1])

        if return_copy:
            imgs[mode_str] = {'img': mat, 'name': img_basename}
    return imgs

def gen_blacklist(yaml_path, modelmapping_path):
    with open(yaml_path, 'r') as f:
        blk_cat_model = yaml.safe_load(f)


    blk_model = set(blk_cat_model.get('models', []))
    blk_cat = set(blk_cat_model.get('categories',[]))
    with open(modelmapping_path, 'r') as f:
        csvr = csv.reader(f)
        next(csvr)
        for r in csvr:
            if r[2] in blk_cat or r[3] in blk_cat:
                blk_model.add(r[1])
                blk_cat.add(r[2])
                blk_cat.add(r[3])


    blk_file = tempfile.NamedTemporaryFile(mode='w',delete=False)
    blk_file.write('\n'.join(sorted(blk_model)))
    blk_file.close()
    return blk_file.name, blk_cat, blk_model

def gen_house_obj_mtl(house_dir):
    c_str = 'cd {} && {} house.json house.obj'.format(osp.abspath(house_dir),suncgtb_scn2scn)
    r = subprocess.run(c_str, shell=True)
    if r.returncode !=0:
        raise SUNCGToolboxError(returncode = r, message = 'SUNCGToolbox scn2scn exited with return code {}, make sure you have compiled SUNCGtoolbox'.format(r))

def rm_house_obj_mtl(house_dir):
    try:
        os.remove(osp.join(house_dir, 'house.obj'))
    except OSError:
        pass
    try:
        os.remove(osp.join(house_dir, 'house.mtl'))
    except OSError:
        pass
