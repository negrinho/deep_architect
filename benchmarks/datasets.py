import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

# Simple Datasets from the UCI Machine Learning Repository

class IRIS(data.Dataset):
    """`Iris <https://archive.ics.uci.edu/ml/datasets/Iris>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``iris.data`` exists or will be saved to if download is set to True.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    filename = "iris.data"
    md5_checksum = '42615765a885ddf54427f12c34a0a070'

    def __init__(self, root, download=True):
        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        fp = os.path.join(root, self.filename)
        # Take first four columns as data
        # Convert last column of string labels into integer labels
        def iris_labels(label):
            if label == b'Iris-setosa':
                return 0
            elif label == b'Iris-versicolor':
                return 1
            elif label == b'Iris-virginica':
                return 2
        self.data = np.loadtxt(fp, delimiter=',', usecols=[0,1,2,3])
        self.labels = np.loadtxt(fp, delimiter=',', usecols=[4], converters={4: iris_labels})

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (features, target) where target is index of the target class.
        """
        features, target = self.data[index], int(self.labels[index])
        return features, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        fpath = os.path.join(root, self.filename)
        if not check_integrity(fpath, self.md5_checksum):
            return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.md5_checksum)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

class WINE(data.Dataset):
    """`Wine <https://archive.ics.uci.edu/ml/datasets/Wine>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``wine.data`` exists or will be saved to if download is set to True.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    filename = "wine.data"
    md5_checksum = '3e584720e6718d28509f86f05b7885a1'

    def __init__(self, root, download=True):
        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        fp = os.path.join(root, self.filename)
        # First column is label (int)
        # Next 13 columns are features
        self.data = np.loadtxt(fp, delimiter=',', usecols=np.arange(1, 14))
        self.labels = np.loadtxt(fp, delimiter=',', usecols=[0])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (features, target) where target is index of the target class.
        """
        features, target = self.data[index], int(self.labels[index])
        return features, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        fpath = os.path.join(root, self.filename)
        if not check_integrity(fpath, self.md5_checksum):
            return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.md5_checksum)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

class CANCER(data.Dataset):
    """`Breast Cancer <https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``wdbc.data`` exists or will be saved to if download is set to True.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    filename = "wdbc.data"
    md5_checksum = 'c6dd5a2909808f3a901cf00cfd8dfff0'

    def __init__(self, root, download=True):
        self.root = os.path.expanduser(root)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        fp = os.path.join(root, self.filename)
        # First column is an ID number, we throw away this
        # Second column is the label
        # Next 30 columns are features
        # Next 13 columns are features
        def cancer_labels(label):
            if label == b'M': # Malignant
                return 0
            elif label == b'B': # Benign
                return 1
        self.data = np.loadtxt(fp, delimiter=',', usecols=np.arange(2, 32))
        self.labels = np.loadtxt(fp, delimiter=',', usecols=[1], converters={1: cancer_labels})

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (features, target) where target is index of the target class.
        """
        features, target = self.data[index], int(self.labels[index])
        return features, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        fpath = os.path.join(root, self.filename)
        if not check_integrity(fpath, self.md5_checksum):
            return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        root = self.root
        download_url(self.url, root, self.filename, self.md5_checksum)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str
