import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import numpy
import torch.utils.data as data
import os
from sampler import ImbalancedDatasetSampler
import sys

def default_loader(path):
    return Image.open(path).convert('RGB')

mytransform = transforms.Normalize(mean= (-300,), std= (700,))

train_positive_length = 0
train_negative_length = 0


class myImageFolder(data.Dataset):
    def __init__(self, positive_root, negative_root, transform = None, loader = default_loader):
        all_imgs = []
        # positive_files = os.listdir(positive_root)
        # negative_files = os.listdir(negative_root)
        # print(positive_files)
        for neg_fn in os.listdir(negative_root):
            all_imgs.append((neg_fn, int(0)))
        for fn in os.listdir(positive_root):
            all_imgs.append((fn, int(1)))
        self.positive_root = positive_root
        self.negative_root = negative_root
        self.imgs = all_imgs
        self.transform = transform
        self.loader = loader
        if 'train' in positive_root:
            global train_positive_length
            global train_negative_length
            train_positive_length = len([x for x in all_imgs if x[1] == 1])
            train_negative_length = len([x for x in all_imgs if x[1] == 0])

    def __getitem__(self, index): 
        fn, label = self.imgs[index]
        if label == 1:
            img = torch.from_numpy(numpy.load(os.path.join(self.positive_root, fn)).transpose((2,0,1)))
        elif label == 0:
            img = torch.from_numpy(numpy.load(os.path.join(self.negative_root, fn)).transpose((2,0,1)))

        if self.transform is not None:
            img = self.transform(img)
        # print(fn, img.shape)
        return img.float(), torch.tensor(label).view(-1)

    def __len__(self):
        return len(self.imgs)

train_dataset = myImageFolder(positive_root = './train_node_real', negative_root = './train_node_fake', transform= mytransform)
test_dataset = myImageFolder(positive_root = './test_node_real', negative_root = './test_node_fake', transform= mytransform)

BATCH_SIZE = 1024
NUM_WORKERS = 16


weight_sampler = False
if weight_sampler == True:
    target = np.hstack((np.zeros(int(len([x for x in os.listdir('./train_node_real') if os.path.isfile(os.path.join('./train_node_real', x))])), dtype=np.int32), np.ones(int(len([x for x in os.listdir('./train_node_fake') if os.path.isfile(os.path.join('./train_node_fake', x))])), dtype=np.int32)))
    print('{}, {}'.format(len(np.where(target == 0)[0]), len(np.where(target == 1)[0])))
    class_sample_count = np.unique(target, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])

    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    # print(len(samples_weight))
    sampler = data.WeightedRandomSampler(samples_weight, len(np.where(target == 0)[0]) * 2, replacement=False)
    print(samples_weight)


sampler = ImbalancedDatasetSampler(train_dataset, train_positive_length, train_negative_length)
# print(list(sampler))
# sys.exit(0)


train_loader = data.DataLoader(train_dataset, batch_size= BATCH_SIZE, sampler= sampler, num_workers= NUM_WORKERS)
test_loader = data.DataLoader(test_dataset, batch_size= BATCH_SIZE, num_workers= NUM_WORKERS)