# NOTE: let us say that it is a binary matrix.

from deep_architect.contrib.misc.calibration_utils import generate_indices, argsort
import deep_architect.utils as ut

import numpy as np
import matplotlib.pyplot as plt


def show_image(X):
    plt.imshow(X)
    plt.show()


def permute_rows(X, indices):
    assert X.shape[0] == indices.shape[0]
    return X[indices]


def permute_columns(X, indices):
    assert X.shape[1] == indices.shape[0]
    return X.T[indices].T


def subset_rows(X, indices):
    return X[indices]


def subset_columns(X, indices):
    return X.T[indices].T


def sort_columns(X, col_values, increasing):
    assert X.shape[1] == col_values.shape[0]
    indices = np.argsort(col_values)
    if increasing:
        indices = indices[::-1]
    return permute_columns(X, indices)


def sort_rows(X, row_values, increasing):
    assert X.shape[0] == row_values.shape[0]
    indices = np.argsort(row_values)
    if increasing:
        indices = indices[::-1]
    return permute_rows(X, indices)


def sort_rows_and_columns(X, row_values, col_values, increasing):
    return sort_columns(
        sort_rows(X, row_values, increasing), col_values, increasing)


X = np.random.binomial(1, 0.1, (16, 8))
X = sort_rows_and_columns(X, X.sum(axis=1), X.sum(axis=0), True)
plt.imshow(X)
plt.show()


# NOTE: a good thing to have here is to maek sure that I can label the positions
# of the elements. e.g., the examples that it is refering to, and the models
# that were used.
def visualize(mat):
    pass


def visualize_with_metric(mat, maximizing):
    pass
