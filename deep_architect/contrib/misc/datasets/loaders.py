import numpy as np
import os
import sys
import pickle
import deep_architect.contrib.misc.datasets.augmentation as au


def load_mnist(flatten=False, one_hot=True, validation_frac=0.1):
    from keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    if flatten:
        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))
    if one_hot:
        y_train = au.idx_to_onehot(y_train, 10)
        y_test = au.idx_to_onehot(y_test, 10)
    num_train = int((1.0 - validation_frac) * X_train.shape[0])
    X_train, X_val = X_train[:num_train], X_train[num_train:]
    y_train, y_val = y_train[:num_train], y_train[num_train:]
    return (X_train, y_train, X_val, y_val, X_test, y_test)


def load_cifar10(data_dir,
                 flatten=False,
                 one_hot=True,
                 normalize_range=False,
                 whiten_pixels=True,
                 border_pad_size=0,
                 data_format='NHWC'):
    """Loads all of CIFAR-10 in a numpy array.
    Provides a few options for the output formats. For example,
    normalize_range returns the output images with pixel values in [0.0, 1.0].
    The other options are self explanatory. Border padding corresponds to
    upsampling the image by zero padding the border of the image.
    """
    train_filenames = [
        'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4'
    ]
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
                X = X.transpose((0, 2, 3, 1))
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