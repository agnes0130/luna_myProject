import torch
import torch.utils.data
import torchvision
import numpy
import random


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, train_positive_length, train_negative_length, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(train_positive_length + train_negative_length))

        self.return_indices = list()
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        # print(label_to_count)
        positive_indices = [pos_idx for pos_idx in self.indices if self._get_label(dataset, pos_idx) == 1]       
        # print(positive_indices) 
        negative_indices = [neg_idx for neg_idx in self.indices if self._get_label(dataset, neg_idx) == 0]
        # print(negative_indices)
        sort_negative_indices = numpy.random.choice(negative_indices, size= train_positive_length, replace= False)
        self.return_indices = positive_indices + list(sort_negative_indices)
        random.shuffle(self.return_indices)
        # print(self.return_indices)


    def _get_label(self, dataset, idx):
        return dataset.imgs[idx][1]
                
    def __iter__(self):
        # return (self.indices[i] for i in torch.multinomial(
            #  self.num_samples, replacement=True))
        return iter(self.return_indices)

    def __len__(self):
        return self.num_samples