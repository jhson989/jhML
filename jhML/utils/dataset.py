
import os
import numpy as np


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