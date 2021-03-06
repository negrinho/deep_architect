# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CIFAR-10 data set.
See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os

import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3


class Cifar10DataSet:
    """Cifar10 data set.
  Described by http://www.cs.toronto.edu/~kriz/cifar.html.
  """

    def __init__(self, data_dir, subset='train', use_distortion=True):
        self.data_dir = data_dir
        self.subset = subset
        self.use_distortion = use_distortion

    def get_filenames(self):
        if self.subset in ['train', 'validation', 'eval']:
            return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'image':
                                               tf.FixedLenFeature([],
                                                                  tf.string),
                                               'label':
                                               tf.FixedLenFeature([], tf.int64),
                                           })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([DEPTH * HEIGHT * WIDTH])

        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(
            tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]),
            tf.float32)
        label = tf.cast(features['label'], tf.int32)

        # # Custom preprocessing.
        # image = self.preprocess(image)

        return image, label

    def make_batch(self, batch_size):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames()
        # Repeat infinitely.
        dataset = tf.data.TFRecordDataset(filenames)

        # Parse records.
        dataset = dataset.map(self.parser, num_parallel_calls=2).cache()

        # Potentially shuffle records.
        if self.subset == 'train':
            # Ensure that the capacity is sufficiently large to provide good random
            # shuffling.
            dataset = dataset.apply(
                tf.data.experimental.shuffle_and_repeat(
                    Cifar10DataSet.num_examples_per_epoch(self.subset)))
        else:
            dataset = dataset.repeat()

        # Batch it up.
        dataset = dataset.map(self.preprocess, num_parallel_calls=2)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(batch_size)

        return dataset

    def preprocess(self, image, label):
        """Preprocess a single image in [height, width, depth] layout."""
        image = tf.image.per_image_standardization(image)
        if self.subset == 'train' and self.use_distortion:
            # Pad 4 pixels on each dimension of feature map, done in mini-batch
            image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
            image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
            image = tf.image.random_flip_left_right(image)
        return image, label

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return 512
            return 40000
        elif subset == 'validation':
            return 10000
        elif subset == 'eval':
            return 10000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)
