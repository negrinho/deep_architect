import numpy as np


def onehot_to_idx(y_onehot):
    y_idx = np.where(y_onehot > 0.0)[1]
    return y_idx


def idx_to_onehot(y_idx, num_classes):
    num_images = y_idx.shape[0]
    y_one_hot = np.zeros((num_images, num_classes), dtype='float32')
    y_one_hot[np.arange(num_images), y_idx] = 1.0
    return y_one_hot


def center_crop(X, out_height, out_width):
    _, in_height, in_width, _ = X.shape
    assert out_height <= in_height and out_width <= in_width

    start_i = (in_height - out_height) / 2
    start_j = (in_width - out_width) / 2
    out_X = X[:, start_i:start_i + out_height, start_j:start_j + out_width, :]

    return out_X


# random crops for each of the images.
def random_crop(X, out_height, out_width):
    num_examples, in_height, in_width, _ = X.shape
    # the ouput dimensions have to be smaller or equal that the input dimensions.
    assert out_height <= in_height and out_width <= in_width

    start_is = np.random.randint(in_height - out_height + 1, size=num_examples)
    start_js = np.random.randint(in_width - out_width + 1, size=num_examples)
    out_X = []
    for ind in range(num_examples):
        st_i = start_is[ind]
        st_j = start_js[ind]

        out_Xi = X[ind, st_i:st_i + out_height, st_j:st_j + out_width, :]
        out_X.append(out_Xi)

    out_X = np.array(out_X)
    return out_X


def random_flip_left_right(X, p_flip):
    num_examples = X.shape[0]
    out_X = X.copy()
    flip_mask = np.random.random(num_examples) < p_flip
    out_X[flip_mask] = out_X[flip_mask, :, ::-1, :]
    return out_X


def per_image_whiten(X):
    """ Subtracts the mean of each image in X and renormalizes them to unit norm.
    """
    num_examples = X.shape[0]
    X_flat = X.reshape((num_examples, -1))
    X_mean = X_flat.mean(axis=1)
    X_cent = X_flat - X_mean[:, None]
    X_norm = np.sqrt(np.sum(X_cent * X_cent, axis=1))
    X_out = X_cent / X_norm[:, None]
    X_out = X_out.reshape(X.shape)
    return X_out


def zero_pad_border(X, pad_size):
    n, height, width, num_channels = X.shape
    X_padded = np.zeros(
        (n, height + 2 * pad_size, width + 2 * pad_size, num_channels),
        dtype='float32')
    X_padded[:, pad_size:height + pad_size, pad_size:width + pad_size, :] = X
    return X_padded
