import numpy as np
import os
import sys
import struct
import deep_architect.contrib.useful.datasets.augmentation as au
if sys.version_info[0] == 2:
    import cPickle as pickle # pylint: disable=E0401
else:
    import pickle

def load_mnist(data_dir, flatten=False, one_hot=True, normalize_range=False):
    from tensorflow.examples.tutorials.mnist import input_data
    # print data_dir
    mnist = input_data.read_data_sets(data_dir, one_hot=one_hot, reshape=flatten)

    def _extract_fn(x):
        X = x.images
        y = x.labels

        if not normalize_range:
            X *= 255.0
        return (X, y)

    Xtrain, ytrain = _extract_fn(mnist.train)
    Xval, yval = _extract_fn(mnist.validation)
    Xtest, ytest = _extract_fn(mnist.test)

    return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

def load_cifar10(data_dir, flatten=False, one_hot=True, normalize_range=False,
        whiten_pixels=True, border_pad_size=0, data_format='NHWC'):
    """Loads all of CIFAR-10 in a numpy array.
    Provides a few options for the output formats. For example,
    normalize_range returns the output images with pixel values in [0.0, 1.0].
    The other options are self explanatory. Border padding corresponds to
    upsampling the image by zero padding the border of the image.
    """
    train_filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4']
    val_filenames = ['data_batch_5']
    test_filenames = ['test_batch']

    # NOTE: this function uses some arguments from the outer scope, namely
    # flatten, one_hot, normalize_range, and possibly others once added.
    def _load_data(fpath):
        with open(fpath, 'rb') as f:
            if sys.version_info > (3, 0):
                # Python3
                d = pickle.load(f, encoding='latin1')
            else:
                # Python2
                d = pickle.load(f)

            # for the data
            X = d['data'].astype('float32')

            # reshape the data to the format (num_images, height, width, depth)
            num_images = X.shape[0]
            num_classes = 10
            X = X.reshape((num_images, 3, 32, 32))
            if data_format == 'NHWC':
                X = X.transpose((0,2,3,1))
            X = X.astype('float32')

            # transformations based on the argument options.
            if normalize_range:
                X = X / 255.0

            if flatten:
                X = X.reshape((num_images, -1))

            # for the labels
            y = np.array(d['labels'])

            if one_hot:
                y_one_hot = np.zeros((num_images, num_classes), dtype='float32')
                y_one_hot[np.arange(num_images), y] = 1.0
                y = y_one_hot

            return (X, y)

    # NOTE: this function uses some arguments from the outer scope.
    def _load_data_multiple_files(fname_list):

        X_parts = []
        y_parts = []
        for fname in fname_list:
            fpath = os.path.join(data_dir, fname)
            X, y = _load_data(fpath)
            X_parts.append(X)
            y_parts.append(y)

        X_full = np.concatenate(X_parts, axis=0)
        y_full = np.concatenate(y_parts, axis=0)

        return (X_full, y_full)

    Xtrain, ytrain = _load_data_multiple_files(train_filenames)
    Xval, yval = _load_data_multiple_files(val_filenames)
    Xtest, ytest = _load_data_multiple_files(test_filenames)

    if whiten_pixels:
        mean = Xtrain.mean(axis=0)[None, :]
        std = Xtrain.std(axis=0)[None, :]
        Xtrain = (Xtrain - mean) / std
        Xval = (Xval - mean) / std
        Xtest = (Xtest - mean) / std

    # NOTE: the zero padding is done after the potential whitening
    if border_pad_size > 0:
        Xtrain = au.zero_pad_border(Xtrain, border_pad_size)
        Xval = au.zero_pad_border(Xval, border_pad_size)
        Xtest = au.zero_pad_border(Xtest, border_pad_size)

    return (Xtrain, ytrain, Xval, yval, Xtest, ytest)

# load mnist, but for DyNet 
def read_mnist(dataset, path):
    if dataset is "training":
        fname_img = os.path.join(path, "train-images-idx3-ubyte")
        fname_lbl = os.path.join(path, "train-labels-idx1-ubyte")
    elif dataset is "testing":
        fname_img = os.path.join(path, "t10k-images-idx3-ubyte")
        fname_lbl = os.path.join(path, "t10k-labels-idx1-ubyte")
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, "rb") as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, "rb") as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        images = np.multiply(
            np.fromfile(fimg, dtype=np.uint8).reshape(len(labels), rows*cols),
            1.0 / 255.0)

    get_instance = lambda idx: (labels[idx], images[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(labels)):
        yield get_instance(i)

def download_examples(path):
    if not os.path.exists(path): 
        os.makedirs(path)
    import gzip
    from six.moves import urllib
    baseurl = "http://yann.lecun.com/exdb/mnist/"
    for elem in ["train-images-idx3-ubyte.gz",
                 "train-labels-idx1-ubyte.gz",
                 "t10k-images-idx3-ubyte.gz",
                 "t10k-labels-idx1-ubyte.gz"]:
        print("downloading " + elem + " ...")
        data = urllib.request.urlopen(baseurl + elem)
        file_path = os.path.join(path, elem)
        with open(file_path, 'wb') as f:
            f.write(data.read())
        with open(file_path.replace('.gz', ''), "wb") as f, \
                gzip.GzipFile(file_path) as zip_f: 
            f.write(zip_f.read())
