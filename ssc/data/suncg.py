#!/usr/bin/python3
import os.path as osp
from scipy.io import loadmat

script_dir = osp.dirname(osp.realpath(__file__))

class SUNCGLabels:
    def __init__(self):
        self.index2name = []
        self.name2index = {}

        # classMapping = loadmat(osp.join(script_dir, 'ClassMapping.mat'))
        objcategory_struct = loadmat(osp.join(script_dir, 'suncgObjcategory.mat'),squeeze_me=True)
        objcategory = objcategory_struct['objcategory']
        # self.class_root_id = objcategory['classRootId'].item()
        self.all_classes = objcategory['allcategories'].item()
        self.all_labeled_obj = objcategory['all_labeled_obj'].item()
        self.class_id = objcategory['classid'].item()
        # self.class_root = objcategory['categoryRoot'].item()
        # self.class_NYU40 = objcategory['classNYU40'].item()

        #Setup subcategory mapping
        oh = objcategory['object_hierarchical'].item()
        self.class2class_root_id = {}
        self.class2class_root = {}
        #ID 0 are free pixels
        self.class_root_id2class_root = ['free']
        for cat_id, node in enumerate(oh):

            cat = node['categoryname'].item()
            children = node['clidern'].item()

            self.class_root_id2class_root.append(cat)

            #Check if string
            if hasattr(children, 'strip'):
                children = [children]

            for c in children:
                self.class2class_root_id[c] = cat_id + 1 #Since we added background first
                self.class2class_root[c] = cat

        # Index mapping
        self.model2index = {model:idx for idx,model in enumerate(self.all_labeled_obj)}

    def getClass(self, model_name):
        try:
            idx = self.model2index[model_name]
            class_id = self.class_id[idx]
            class_name = self.all_classes[class_id - 1] #MATLAB indexing
        except KeyError:
            class_id = -1
            class_name = model_name

        return class_id, class_name

    def getClassRootFromRootID(self, class_id):
        return self.class_root_id2class_root[class_id]

    def getClassRootIDFromRoot(self, class_name):
        return self.class_root_id2class_root.index(class_name)

    def getClassRoot(self, model_name):
        _, class_name = self.getClass(model_name)
        try:
            class_root_id = self.class2class_root_id[class_name]
            class_root = self.class2class_root[class_name]
        except KeyError:
            class_root_id = len(self.class_root_id2class_root) - 1
            class_root = self.class_root_id2class_root[class_root_id]
        return class_root_id, class_root

    def getClasses(self):
        return self.class_root_id2class_root

    def getNbrClasses(self):
        return len(self.class_root_id2class_root)

    def get_NYU_mapping(self):
        mapping = loadmat(osp.join(script_dir, 'ClassMapping.mat'),squeeze_me=True)
        print(mapping.keys())
        return mapping
