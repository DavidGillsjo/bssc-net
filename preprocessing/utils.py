import time
import sys
import multiprocessing as mp
import queue
import logging
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp
import json

from multiprocessing.managers import BaseManager


class InvalidHousesMemory:
    def __init__(self, houses = None, logger = None):
        self.houses = set(houses) if houses else set()
        self.lock = mp.Lock()

    def set(self, houses):
        self.houses = set(houses)

    def get(self):
        return self.houses

    def invalidate(self, house_id):
        with self.lock:
            self.houses.add(house_id)

    def validate(self, house_id):
        with self.lock:
            self.houses.discard(house_id)

    def is_valid(self, house_id):
        return house_id not in self.houses


class InvalidHouses:
    def __init__(self, filepath, logger = None):
        self.houses = InvalidHousesMemory()
        self.filepath = filepath
        try:
            self._read_file()
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def get_proxy(self):
        '''
        manager: A Multprocessing manager handling proxy object. mp.Manager
        '''
        manager = RenderManager()
        manager.register('invalidhouses', InvalidHousesMemory, exposed = ['invalidate', 'validate', 'get'])
        manager.start()

        ih_proxy = manager.invalidhouses(self.houses.get())
        self.houses = ih_proxy
        return manager, ih_proxy

    def _read_file(self):
        with open(self.filepath, 'r') as f:
            self.houses.set(json.load(f))

    def _write_file(self):
        with open(self.filepath, 'w') as f:
            json.dump(sorted(self.houses.get()), f, indent=0)

    def invalidate(self, house_id):
        self.houses.invalidate(house_id)

    def validate(self, house_id):
        self.houses.validate(house_id)

    def is_valid(self, house_id):
        return self.houses.is_valid(house_id)

    def store(self):
        self._write_file()

class RenderManager(BaseManager):
    pass

def getFigure():
    dpi = 200.0
    resolution = np.array([1920, 1080], dtype=np.float)
    return plt.figure(figsize=resolution/dpi, dpi = dpi)

class LoggerProcess(mp.Process):
    def __init__(self, logger, id_queue, interval):
        mp.Process.__init__(self)
        self.id_queue = id_queue
        self.interval = interval
        self.avg_rate = 0
        self.alpha = 0.3
        self.prev_queue_len = None
        self.logger = logger

    def run(self):
        while not self.id_queue.empty():
            if self.prev_queue_len:
                rate = (self.prev_queue_len - self.id_queue.qsize())/self.interval
                self.avg_rate = (1-self.alpha)*self.avg_rate  + self.alpha*rate

            queue_len = self.id_queue.qsize()
            avg_rate_min = self.avg_rate * 60
            if avg_rate_min > 0:
                rem_min = int(queue_len/avg_rate_min)
                self.logger.info('{} House ids remaining. Rate: {} ids/min, Left: {} h, {} min'.format(queue_len,
                                                                                     int(avg_rate_min),
                                                                                     int(rem_min / 60),
                                                                                     rem_min % 60))
            self.prev_queue_len = queue_len
            time.sleep(self.interval)


# NOTE: First args in worker must be (device,  queue) and not included in args
def run_mp_house3d(devices, nbr_proc, work_list, worker, args):

    work_queue = mp.Queue()
    for w in work_list:
        work_queue.put(w)

    status_proc = LoggerProcess(mp.get_logger(), work_queue, 60)
    status_proc.start()

    #No need for MP if one process only
    if nbr_proc == 1:
        try:
            worker(*((devices[0], work_queue) + args))
        finally:
            status_proc.terminate()
        return

    proc_map = {d:[] for d in devices}
    time.sleep(0.1) #Give queue time to write
    while not work_queue.empty():
        for d, procs in proc_map.items():
            # Create new process list
            alive_p = [p for p in procs if p.is_alive()]
            # Fill up with missing
            if len(alive_p) < nbr_proc:
                for new_i in range(nbr_proc - len(alive_p)):
                        p = mp.Process(target=worker, args=(d, work_queue) + args)
                        alive_p.append(p)
                        p.start()
                        time.sleep(0.3) # Give it some time for GPU allocation
                proc_map[d] = alive_p
        time.sleep(1)

    #Wait for processes to terminate
    for d, procs in proc_map.items():
        for p in procs:
            p.join()
    status_proc.terminate()


class StreamToLogger(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   """
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''

   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())

   def flush(self):
       pass


def setup_mp_logger(logfile = None):
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)
    if logfile:
        file_h = logging.FileHandler(logfile, mode='w')
        file_h.setFormatter( logging.Formatter('[%(asctime)s][%(levelname)s/%(processName)s] %(message)s'))
        logger.addHandler(file_h)
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)
    else:
        stream_h = logging.StreamHandler(sys.stdout)
        stream_h.setFormatter( logging.Formatter('[%(asctime)s][%(levelname)s/%(processName)s] %(message)s'))
        logger.addHandler(stream_h)

    return logger
