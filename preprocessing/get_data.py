#! /usr/bin/python3
import argparse
import zipfile
import os
import os.path as osp
from urllib.request import urlretrieve
import logging
import sys
import io
import threading
import time

def dl_progress(count, blockSize, totalSize):
      counts_freq = int(totalSize/(blockSize*100))
      if count % counts_freq == 0:
          percent = count*blockSize*100/totalSize
          print('\rDownloading...{:0.2f}%'.format(percent))
      sys.stdout.flush()

def get_suncg(data_dir):
    my_dir = osp.join(data_dir, 'suncg')
    if osp.exists(my_dir):
        print('SUNCG folder already exists, do nothing')
        return

    # Download zipfile if we have not
    zip_target = osp.join(data_dir, 'suncg_data.zip')
    suncg_url = 'http://suncg.cs.princeton.edu/data/suncg_data.zip'
    if not osp.exists(zip_target):
        print('Getting the SUNCG data')
        urlretrieve(suncg_url, zip_target, reporthook=dl_progress)

    # Unzip nested
    os.mkdir(my_dir)
    with zipfile.ZipFile(zip_target, mode='r') as my_zip:
        for nested_zip_name in my_zip.infolist():
            print('Extracting {}'.format(nested_zip_name.filename))
            nested_zip = io.BytesIO(my_zip.read(nested_zip_name))
            with zipfile.ZipFile(nested_zip, mode='r') as my_zip2:
                my_zip2.extractall(path = my_dir)

def get_pbrs(data_dir):
    my_dir = osp.join(data_dir, 'suncg', 'camera')
    if osp.exists(my_dir):
        print('PBRS camera folder already exists, do nothing')
        return

    # Download zipfile if we have not
    zip_target = osp.join(data_dir, 'camera_v2.zip')
    pbrs_url = 'http://pbrs.cs.princeton.edu/pbrs_release/data/camera_v2.zip'
    if not osp.exists(zip_target):
        print('Getting the PBRS camera data')
        urlretrieve(pbrs_url, zip_target, reporthook=dl_progress)

    print('Unzipping the PBRS camera data')
    with zipfile.ZipFile(zip_target, mode='r') as my_zip:
        my_zip.extractall(path = my_dir)


class convThread(threading.Thread):
    def __init__(self, c, house_dirs, exec_file, nbr_threads):
        threading.Thread.__init__(self)
        self.c = c
        self.house_dirs = house_dirs
        self.exec_file = exec_file
        self.nbr_threads = nbr_threads

    def run(self):
        for hd in self.house_dirs[self.c::self.nbr_threads]:
            if not (osp.isfile(osp.join(hd, 'house.obj')) and
                    osp.isfile(osp.join(hd, 'house.mtl'))):
                # start_t = time.time()
                os.system('cd {} && {} house.json house.obj'.format(hd, self.exec_file))
                # print('House ID {} took {}s'.format(hd, time.time() - start_t))

def parse_args():
    script_dir = osp.dirname(osp.realpath(__file__))
    parser = argparse.ArgumentParser(description='Get data for AL')
    parser.add_argument('--data-dir', type=str, default=osp.abspath(osp.join(script_dir,'..','data')),
                        help='Data folder. (default: %(default)s)')
    parser.add_argument('--suncg', action='store_true', default=False,
                        help='Get the SunCG dataset')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.suncg or args.suncg_house3D:
        get_suncg(args.data_dir)
        get_pbrs(args.data_dir)
