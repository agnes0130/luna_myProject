import lmdb
import pickle
import msgpack
import tqdm
import pyarrow as pa

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
import os
class LMDBDataSet(data.Dataset):
    def __init__(self, db_path, use_min_ratio, use_max_ratio):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=os.path.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            byteflow = txn.get(b'__len__')
            self.length = pickle.loads(byteflow)
        self.min_ind = int(self.length * use_min_ratio)
        self.max_ind = int(self.length * use_max_ratio) - 1
        self.length = self.max_ind - self.min_ind

    def __getitem__(self, index):
        env = self.env
        if index > self.length:
            assert "Index {} is out of boundary".format(self.min_ind + index)
        with env.begin(write=False) as txn:
            byteflow = txn.get(u'{}'.format(self.min_ind + index).encode('ascii'))
        data, label = pickle.loads(byteflow)


        return data, label

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'