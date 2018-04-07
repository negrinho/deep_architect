import os
import os.path
import numpy as np
import sys

import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision.datasets.utils import download_url, check_integrity

# Simple Datasets from the UCI Machine Learning Repository
# These datasets are simple and stored in one file. We must
# split them into a train and test file.
# TODO: Split them into a train and test file. Currently train=True does nothing.

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

    def __init__(self, root, download=True, train=True, one_hot=True):
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
        self.labels = np.loadtxt(fp, delimiter=',', usecols=[4], converters={4: iris_labels}, dtype=int)
        if one_hot:
            self.labels = np.eye(3, dtype=int)[self.labels]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (features, target) where target is index of the target class.
        """
        features, target = self.data[index], self.labels[index]
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

    def __init__(self, root, download=True, train=True, one_hot=True):
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
        self.labels = np.loadtxt(fp, delimiter=',', usecols=[0], dtype=int) - 1
        if one_hot:
            self.labels = np.eye(3, dtype=int)[self.labels]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (features, target) where target is index of the target class.
        """
        features, target = self.data[index], self.labels[index]
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

    def __init__(self, root, download=True, train=True, one_hot=True):
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
        self.labels = np.loadtxt(fp, delimiter=',', usecols=[1], converters={1: cancer_labels}, dtype=int)
        if one_hot:
            self.labels = np.eye(2, dtype=int)[self.labels]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (features, target) where target is index of the target class.
        """
        features, target = self.data[index], self.labels[index]
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

class MODEL_EVAL_LOGS(data.Dataset):
    """ Model Evaluation Log Dataset: A torch dataset based on logged evaluations from darch.search_logger
    Unfinished
    """
    def __init__(self, root):
        from darch import search_logging as sl
        self.root = os.path.expanduser(root)
        self.raw_data = sl.read_search_folder(self.root)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (features, target) where target is index of the target class.
        """
        eval_log = self.raw_data[index]
        return eval_log['features'], eval_log['results']

    def __len__(self):
        return len(self.raw_data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        return fmt_str

class SubDataset(data.Dataset):
    def __init__(self, dataset, indices):
        assert(max(indices) < len(dataset))
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
    def __len__(self):
        return len(self.indices)
    def __repr__(self):
        fmt_str = 'Dataset ' + self.dataset.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.dataset.root)
        return fmt_str

def train_val_split(dataset, valid_size=0.1):
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    split = int(valid_size * len(dataset))
    train_idx, valid_idx = indices[split:], indices[:split]
    return SubDataset(dataset, train_idx), SubDataset(dataset, valid_idx)

class TorchInMemoryDataset:
    """ Wrapper around a pytorch dataset to create an InMemoryDataset
    InMemoryDataset is a wrapper around a dataset defined by in memory objects X and y
    So we've come full circle...
    """
    def __init__(self, dataset, shuffle_at_epoch_begin, batch_size=32, to_numpy=True):
        self.dataset = dataset
        self.shuffle_at_epoch_begin = shuffle_at_epoch_begin
        
        self.dataloader = data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle_at_epoch_begin)
        self._iterator = self.dataloader.__iter__()
        self.to_numpy = to_numpy

    def get_num_examples(self):
        return len(self.dataset)
    
    def check_size(self, batch_size):
        if self.dataloader.batch_size != batch_size:
            self.dataloader = data.DataLoader(self.dataset, batch_size=batch_size, shuffle=self.shuffle_at_epoch_begin)
            self._iterator = self.dataloader.__iter__()
            return False
        return True

    # Hacking around util.DataLoader to match InMemorydataset. This introduces some overhead
    # In trying and catching exceptions, but it's price we to have to pay for compatibility...
    def next_batch(self, batch_size):
        try:
            self.check_size(batch_size)
            batch = self._iterator.__next__()
        except StopIteration:
            if self.check_size:
                self._iterator = self.dataloader.__iter__()
            batch = self._iterator.__next__()
        if self.to_numpy:
            batch = (batch[0].numpy(), batch[1].numpy())
        return batch

# TODO: Move this into an appropriate area

class ToNumpy(object):
    def __call__(self, image):
        return np.asarray(image)

class ToOneHot(object):
    def __init__(self, num_classes):
        self.eye = torch.eye(num_classes)
    def __call__(self, label):
        return self.eye[label]

# TODO: DELETE THIS ONCE TORCH 0.4 COMES OUT
class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]