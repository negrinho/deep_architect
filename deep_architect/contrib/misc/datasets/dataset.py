
import numpy as np
import tensorflow as tf

class TFDataset:
    def __init__(self, X, y, shuffle_at_epoch_begin, batch_size=128, batch_transform_fn=None):
        if X.shape[0] != y.shape[0]:
            assert ValueError("X and y the same number of examples.")

        self.dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if shuffle_at_epoch_begin:
            self.dataset = self.dataset.shuffle(45000)
        if batch_transform_fn is not None:
            self.dataset = self.dataset.map(batch_transform_fn)
        self.dataset = self.dataset.batch(batch_size)
        self.dataset = self.dataset.cache().prefetch(batch_size)

        self.num_examples = X.shape[0]
        self.epoch = 0
        self.examples_in_cur_epoch = 0
        self.iterator = self.dataset.make_one_shot_iterator()

    def get_num_examples(self):
        return self.num_examples

    def next_batch(self):
        X, y = self.iterator.next()
        self.examples_in_cur_epoch += X.shape[0]
        if self.examples_in_cur_epoch == self.num_examples:
            self.iterator = self.dataset.make_one_shot_iterator()
            self.examples_in_cur_epoch = 0
            self.epoch += 1
            return (X, y), True
        else:
            return (X, y), False

class InMemoryDataset:
    """Wrapper around a dataset for iteration that allows cycling over the
    dataset.
    This functionality is especially useful for training. One can specify if
    the data is to be shuffled at the end of each epoch. It is also possible
    to specify a transformation function to applied to the batch before
    being returned by next_batch.
    """

    def __init__(self, X, y, shuffle_at_epoch_begin, batch_transform_fn=None):
        if X.shape[0] != y.shape[0]:
            assert ValueError("X and y the same number of examples.")

        self.X = X
        self.y = y
        self.shuffle_at_epoch_begin = shuffle_at_epoch_begin
        self.batch_transform_fn = batch_transform_fn
        self.iter_i = 0

    def get_num_examples(self):
        return self.X.shape[0]

    def next_batch(self, batch_size):
        """Returns the next batch in the dataset.
        If there are fewer that batch_size examples until the end
        of the epoch, next_batch returns only as many examples as there are
        remaining in the epoch.
        """

        n = self.X.shape[0]
        i = self.iter_i

        # shuffling step.
        if i == 0 and self.shuffle_at_epoch_begin:
            inds = np.random.permutation(n)
            self.X = self.X[inds]
            self.y = self.y[inds]

        # getting the batch.
        eff_batch_size = min(batch_size, n - i)
        X_batch = self.X[i:i + eff_batch_size]
        y_batch = self.y[i:i + eff_batch_size]
        self.iter_i = (self.iter_i + eff_batch_size) % n

        # transform if a transform function was defined.
        if self.batch_transform_fn != None:
            X_batch_out, y_batch_out = self.batch_transform_fn(X_batch, y_batch)
        else:
            X_batch_out, y_batch_out = X_batch, y_batch

        return (X_batch_out, y_batch_out)
