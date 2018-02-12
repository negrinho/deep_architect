# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import darch.searchers as se
import examples.tensorflow.mnist.evaluator as ev
import examples.tensorflow.mnist.search_space as ss

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_FREQUENCY = 100  # Number of steps between evaluations.

FLAGS = None

def maybe_download(filename):
  """Download the data from Yann's website, unless it's already here."""
  if not tf.gfile.Exists(WORK_DIRECTORY):
    tf.gfile.MakeDirs(WORK_DIRECTORY)
  filepath = os.path.join(WORK_DIRECTORY, filename)
  if not tf.gfile.Exists(filepath):
    filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
    with tf.gfile.GFile(filepath) as f:
      size = f.size()
    print('Successfully downloaded', filename, size, 'bytes.')
  return filepath

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def main():

  # Get the data.
  train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
  train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
  test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
  test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

  # Extract it into numpy arrays.
  train_data = extract_data(train_data_filename, 60000)
  train_labels = extract_labels(train_labels_filename, 60000)
  test_data = extract_data(test_data_filename, 10000)
  test_labels = extract_labels(test_labels_filename, 10000)

  # Generate a validation set.
  validation_data = train_data[:VALIDATION_SIZE, ...]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_data = train_data[VALIDATION_SIZE:, ...]
  train_labels = train_labels[VALIDATION_SIZE:]
  train_size = train_labels.shape[0]

  # TODO: probably pass the data to this model.
  evaluator = ev.Evaluator(train_data, train_labels, 
    validation_data, validation_labels, test_data, test_labels, 
    BATCH_SIZE, IMAGE_SIZE, NUM_CHANNELS, NUM_LABELS, NUM_EPOCHS)
  # searcher 
  (inputs, outputs, hs) = ss.get_search_space_fn(NUM_LABELS)()
  se.random_specify(outputs.values(), hs.values())
  evaluator.eval(inputs, outputs, hs)

if __name__ == '__main__':
  main()

  # # get search space.
  # for _ in xrange():
  #   (inputs, outputs, hs, v_hist, cfg_d) = searcher.sample()
  #   val = evaluator.eval(inputs, outputs, hs)
  #   searcher.update(val, cfg_d)
