
import os
import numpy as np
import math
from jhML.compute import cp

#######################################################
# Base class for dataset
#######################################################
class Dataset:
    r"""An abstract class representing a :class:`Dataset`.
    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    the default options of :class:`jhML.utils.DataLoader`.
    :class:`jhML.utils.DataLoader`` by default constructs a index sampler that yields integral indices. 
    """
    def __init__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    # Lack of Default `__len__` in Python Abstract Base Classe



#######################################################
# Data loader class
#######################################################
class Dataloader:

    def __init__(self, dataset, batch_size, shuffle=True, gpu=False, drop_last=True):
        r""" Return mini-batched data from dataset
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_len = len(dataset)
        self.max_iter = int(self.data_size / batch_size) if drop_last else math.ceil(self.data_size / batch_size)
        self.gpu = gpu
        self.reset()


        # Validate custom dataset
        if  self.data_len <= 0:
            raise ValueError("Invalid dataset : len(dataset) > 1")
        if isinstance (self.dataset[0], tuple) and not isinstance (self.dataset[0][0], np.array):
            raise ValueError("Dataset sholud return numpy.ndarray")
        elif not isinstance (self.dataset[0], np.array):
            raise ValueError("Dataset sholud return numpy.ndarray")

        if isinstance (self.dataset[0], tuple):
            self.num_return = len(self.dataset[0])
        else:
            self.num_return = 1
        

    def reset(self):
        self.iter = 0

        if self.shuffle:
            self.idx = np.random.permutation(self.data_len)
        else:
            self.idx = np.arange(self.data_len)


    def __iter__(self):
        return self

    def __next__(self):
        if self.iter >= self.max_iter:
            self.reset()
            raise StopIteration
            
        i = self.iter
        batch_idx = self.idx[i*self.batch_size : (i+1)*batch_size]

        data = ([[] for _ in range(self.num_return)])
        for idx in batch_idx:
            example = self.dataset[idx]
            if self.num_return == 1:
                data[0].append(example)
            else:
                for e in range(self.num_return):
                    data[e].append(example[e])

        if gpu:
            data = cp.array(data)
        else:
            data = np.array(data, dtype=object)
            
        return data

