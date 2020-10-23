"""Add house3d to PYTHONPATH

Usage:
    import _init_house3d
    import House3D
"""

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(osp.realpath(__file__))
house3d_dir = osp.abspath(osp.join(this_dir, '..', 'libs', 'House3D'))

# Add lib to PYTHONPATH
add_path(house3d_dir)
