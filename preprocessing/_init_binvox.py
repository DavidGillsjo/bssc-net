"""Add binvox_rw to PYTHONPATH

Usage:
    import _init_binvox
    import binvox
"""

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(osp.realpath(__file__))
binvox_dir = osp.abspath(osp.join(this_dir, '..', 'libs', 'binvox-rw-py'))

# Add lib to PYTHONPATH
add_path(binvox_dir)
