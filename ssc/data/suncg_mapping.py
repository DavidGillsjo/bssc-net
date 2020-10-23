import ssc.data.suncg as suncg
import yaml
import numpy as np
from collections import OrderedDict
import os.path as osp
from multiprocessing.managers import BaseManager

def setup_yaml():
  """ https://stackoverflow.com/a/8661021 """
  represent_dict_order = lambda self, data:  self.represent_mapping('tag:yaml.org,2002:map', data.items())
  yaml.add_representer(OrderedDict, represent_dict_order)

class MappingManager(BaseManager):
    pass

#Store as many variables as np arrays as possible to reduce copy on access memory leak during multiprocessing.
class SUNCGMapping:
    def __init__(self, yaml_map=None):
        self.all_labels = np.array(suncg.SUNCGLabels().getClasses())

        self.wrapper = yaml_map is None
        if self.wrapper:
            return

        with open(osp.join(osp.dirname(suncg.__file__), 'mappings', '{}.yaml'.format(yaml_map))) as f:
            y = yaml.safe_load(f)
            labels = y['labels']
            mapping = y['mapping']


        self.labels = np.array(labels)
        self.int_mapping = np.array([labels.index(mapping[suncg_label]) for suncg_label in self.all_labels])


    @classmethod
    def create_proxy(cls, yaml_map = None):
        manager = MappingManager()
        manager.register('SUNCGMapping', cls, exposed = ['map', 'get_nbr_classes', 'get_classes', 'get_class_id'])
        manager.start()
        return manager, manager.SUNCGMapping(yaml_map)

    def map(self, np_array, dtype = None):
        if self.wrapper:
            return np_array.astype(dtype) if dtype else np_array

        new_array = np.zeros_like(np_array, dtype = dtype)
        for old_int, new_int in enumerate(self.int_mapping):
            new_array[np_array == old_int] = new_int

        return new_array

    def get_nbr_classes(self):
        return len(self.get_classes())

    def get_class_id(self, name):
        if self.wrapper:
            return np.flatnonzero(self.all_labels==name)[0]
        else:
            return np.flatnonzero(self.labels==name)[0]

    def get_classes(self):
        if self.wrapper:
            return self.all_labels
        else:
            return self.labels


# Just generate a template if called as script
if __name__ == '__main__':
    sl = suncg.SUNCGLabels()
    labels = sl.getClasses()

    all_NYU_mappings = sl.get_NYU_mapping()
    NYU_labels = ['free', *all_NYU_mappings['elevenClass']]
    NYU_mapping = ['free', *all_NYU_mappings['map36to11']]

    mapping = OrderedDict.fromkeys(labels)
    for li, l in enumerate(labels):
        mapping[l] = NYU_mapping[li]

    setup_yaml()
    with open('suncg11.yaml', 'w') as f:
        yaml.dump({'labels': NYU_labels, 'mapping': mapping}, f)
