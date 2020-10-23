"""Add tsdf-fusion-python to PYTHONPATH

Usage:
    import _init_tsdf
    import fusion
"""

import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(osp.realpath(__file__))
tsdf_dir = osp.abspath(osp.join(this_dir, '..', '..', 'libs', 'tsdf-fusion-python'))

# Add lib to PYTHONPATH
add_path(tsdf_dir)
