import os
import gzip

import numpy as np
import jhML
from jhML import Dataset
from jhML.utils.download import get_file

class MNIST(Dataset):

    num_class = 10

    def __init__(self, train=True, flatten=False):

        self.train = train
        self.flatten = flatten

        self._download_data()
        self.len = len(self.data)

    def __getitem__(self, index):

        return self.data[index].ravel(), self.label[index]

    def __len__(self):
        return self.len

    def _download_data(self):
        url = 'http://yann.lecun.com/exdb/mnist/'
        train_files = {'target': 'train-images-idx3-ubyte.gz',
                       'label': 'train-labels-idx1-ubyte.gz'}
        test_files = {'target': 't10k-images-idx3-ubyte.gz',
                      'label': 't10k-labels-idx1-ubyte.gz'}

        files = train_files if self.train else test_files
        data_path = get_file(url + files['target'])
        label_path = get_file(url + files['label'])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)
        
    def _load_label(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data
