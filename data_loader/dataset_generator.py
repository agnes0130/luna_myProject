from data_loader.annotation_parser import AnnotatedInfo, parseAnnotationXML, loadAnnotations, updateLabels, loadDatasetInfo
from scipy.ndimage.interpolation import rotate
import random
import pyarrow as pa
import cv2
import numpy as np
import shutil
import os
import pickle
import lmdb
# process raw image to obtain standard dataset
# raw image is too big, only very little local area is required for training and test


class DatasetGenerator:
    def __init__(self, benign_data_path, malignant_data_path, benign_label_path, malignant_label_path):
        self.benign_data_path = benign_data_path
        self.malignant_data_path = malignant_data_path
        self.benign_label_path = benign_label_path
        self.malignant_label_path = malignant_label_path
        self.info_dict = []
        self.benign_names = []
        self.malignant_names = []
        self.load()

    def load(self):
        benign_files = os.listdir(self.benign_data_path)
        for benign_file in benign_files:
            self.benign_names.append(os.path.join(self.benign_data_path, benign_file))
        picknumber = len(self.benign_names)
        malignant_files = os.listdir(self.malignant_data_path)
        malignant_data_names = []
        for malignant_file in malignant_files:
            malignant_data_names.append(os.path.join(self.malignant_data_path, malignant_file))
        self.malignant_names = random.sample(malignant_data_names, picknumber)
        print(len(self.benign_names), len(self.malignant_names))

    
    @staticmethod
    def cropROI(images, info, half_size, rot_deg, translation):
        pass

    @staticmethod
    def dumps_pyarrow(obj):
        """
        Serialize an object.
        Returns:
            Implementation-dependent bytes-like object
        """
        return pickle.dumps(obj, protocol=2)

    def generateDataElement(self, name, half_size, half_depth, extend_num, max_rot_deg, max_trans):
        pass

    def getDataElement(self, name):
        element_roi = np.load(name)
        if element_roi is not None:
            return element_roi
        else: raise AssertionError()


    def commitDataset(self, last_ind, txn, roi_list, label_list):
        # shuffle the dataset
        indices = list(range(len(roi_list)))
        random.shuffle(indices)
        for i in indices:
            txn.put(u'{}'.format(last_ind).encode('ascii'), self.dumps_pyarrow((roi_list[i], label_list[i])))
            last_ind += 1
        txn.commit()
        txn = self.db_env.begin(write=True)
        return last_ind, txn
            
    def generateDataset(self, save_name, benign_names, malignant_names, half_size, half_depth, max_rot_deg, max_trans):
        # create a dataset with LMBD format
        
        if os.path.exists(save_name):
            print('Erasing previously created LMDB at "{}"'.format(save_name))
            shutil.rmtree(save_name)
        self.db_env = lmdb.open(save_name, map_size=half_size*2*half_size*2*(half_depth*2+1) * len(benign_names) * 2 * 10)
        beign_size = len(self.benign_names)
        malign_size = len(self.malignant_names)
        total_len = beign_size + malign_size
        
        generated_beign_size = 0
        generated_malign_size = 0
        txn = self.db_env.begin(write=True)
            
        data_indices = list(range(total_len))
        random.shuffle(data_indices)
        roi_list = []
        label_list = []
        last_ind = 0
        for i in data_indices:
            if i < beign_size:
                name = benign_names[i]
                lable = 0
                generated_beign_size += 1
            else:
                name = malignant_names[i-beign_size]
                lable = 1
                generated_malign_size += 1
            roi_list += self.getDataElement(name)
            label_list += [lable]
            

            if len(roi_list) > 1000:
                last_ind, txn = self.commitDataset(last_ind, txn, roi_list, label_list)
                roi_list, label_list = [], []
            print("generated {}/{}".format(generated_beign_size + generated_malign_size, len(benign_names)* 2))
        if len(roi_list) != 0:
            last_ind, txn = self.commitDataset(last_ind, txn, roi_list, label_list)
            roi_list, label_list = [], []
            print('generate rest data')
        print("write length")
        txn.put(b'__len__', self.dumps_pyarrow(last_ind + 1))
        txn.commit()
        print("Flushing database ...")
        self.db_env.sync()
        self.db_env.close()

if __name__ == '__main__':
    data_path = "../data"
    label_path = "../label"
    benign_label_path = label_path + "/benignlabel"
    malignant_label_path = label_path + "/malignantlabel"
    benign_data_path = data_path + "../dataset/train_node_real40"
    malignant_data_path = data_path + "../dataset/train_node_fake40"
    generator = DatasetGenerator(benign_data_path, malignant_data_path, benign_label_path, malignant_label_path)
    
    train_ratio = 0.8
    benign_train_num = int(len(generator.benign_names) * train_ratio)
    malignant_train_num = int(len(generator.malignant_names) * train_ratio)
    
    generator.generateDataset("dataset/train", generator.benign_names[0:benign_train_num], generator.malignant_names[0:malignant_train_num], half_size=32,  half_depth=5, each_class_size=10000, 
                            max_rot_deg=180, max_trans=10)
    generator.generateDataset("dataset/test", generator.benign_names[benign_train_num:], generator.malignant_names[malignant_train_num:], half_size=32,  half_depth=5, each_class_size=2000, 
                            max_rot_deg=180, max_trans=10)
    