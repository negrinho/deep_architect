

###${MARKDOWN}
# # First steps through a complete example
#
# In essence, DeepArchitect is a framework to search over general computational graphs.
# The implementation is not tied ot any particular framework, i.e., you can
# use it with your favorite deep learning framework, or any other domain that
# fits into the ideas of DeepArchitect.
# There are many concepts that come together to make DeepArchitect possible.
# We believe that a good initial first step is to show how to write a simple
# example in Keras from the ground up.
# Even if not all components are clear, it will give the reader a sense of the
# concepts involved in coding a complete example.
# We start by adapting [one of the MNIST examples](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
# in the Keras repo.
# Keras is a nice framework for a first example because it hides away much of the
# complexity related to training the model, allows us to focus on writing
# down the search space.
#
# We assume that the reader is familiar with Keras.
# Adapting an existing example in a well-known framework allows us to very
# clearly explain the steps necessary for adapting an existing implementation
# to use DeepArchitect for architecture search.
# Now that we explained the rationale behind this tutorial, let us move into the
# code.
#
# Let us start by looking at the original example:
# '''python
# '''Trains a simple convnet on the MNIST dataset.

# Gets to 99.25% test accuracy after 12 epochs
# (there is still a lot of margin for parameter tuning).
# 16 seconds per epoch on a GRID K520 GPU.
# '''

# from __future__ import print_function
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K

# batch_size = 128
# num_classes = 10
# epochs = 12

# # input image dimensions
# img_rows, img_cols = 28, 28

# # the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=keras.optimizers.Adadelta(),
#               metrics=['accuracy'])

# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
# '''
#
# Starting from the code above, we maintain exactly the same code that
# loads the data.

###${CODE}

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

###${MARKDOWN}
# Now we focus on defining the search space. The previous model is defined
# as
# '''python
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))
# '''
#
# This defines a  to a single fixed model.
# DeepArchitect provides a language to write down search spaces over architectures
# in a compositional and modular manner.
# We will show that this does not represent a substantial amount of work over
# the common workflow presented above.

