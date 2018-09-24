

A quite important part of DeepArchitect is the attention paid to logging
and visualization.
Architecture search provides a tremendous opportunity for visualization for the
problem at hand.
A large number of models can be evaluated in the same setting, meaning that
some broad insights about the results can be extracted from the results of the
search.
Just compare this with the current non-architecture search workflow in
machine learning.
In such a workflow, it is hard to evaluate search over very diverse models
as the description of the search space is very ad-hoc, with only a few simple
hyperparameters that we can express in this ad-hoc manner.

In DeepArchitect, we can express a search space in a standard way which
is largely shared for all the frameworks and domains that we consider
(only necessary to reimplement the simple modules, with the abstract modules
such as substitution modules being directly applicable in the new domain).
In this tutorial, we will talk a bit about how to use the logging functionalities
in place for DeepArchitect.
The logging functionality has a component that is fixed and a componet that
can be specified by the user.
# NOTE: probably, this would be specified in a better way.
# probably, just remove.

# talk about setting up logging.
# talk about the post visualization.
# save a bunch of information like the number of parameters and what not.

Another aspect that is show cased in this tutorial is how easily is to convert
an existing example in Keras to one in architecture search that allows us to
keep information about the evaluations that were done.
The main changes are writing down a search space and setting up the logging
functionality.

# eval, eval_start, eval_step,

# run a visualization on the code base, run on command line. the following line.

# TODO: finish the logging and visualization example

Let us start with the MNIST example in Keras adapted to DeepArchitect.
This was copied from the Keras website.


'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

This is the simplest MNIST example that we can get from the Keras examples
website.
It takes the MNIST examples in flattened form and applies a two layer multi-layer
perceptron.
We will adapt this example, but also, set up logging and visualization to

It does not make sense to use the test data as validation data, so we will
create a small validation set out of training set and use the test set only
to evaluate the best architecture that we will found.

import deep_architect.core as co
from keras.layers import Input

### NOTE: make this a simple adaptation of the previous model.
class Evaluator:
    def __init__(self, batch_size, epochs):
        batch_size = 128
        num_classes = 10
        epochs = 20

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        num_val = 10000

    def eval(self, inputs, outputs, hs):
        x = Input((784,) dtype='float32')
        co.forward({inputs["In"] : x})
        y = outputs["Out"].val
        model = Model(inputs=x, outputs=y)

        model.summary()

        model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(),
                    metrics=['accuracy'])

        history = model.fit(x_train, y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=1,
                            validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        results = {
            "train_accuracy" : ,
            "validation_accuracy" : ,
            "test_accuracy" : ,
            "num_parameters" : ,
        }

# TODO: using the information about the model. what can be done here?
import deep_architect.helpers.keras as hke
import deep_architect.hyperparameters as hp
import deep_architect.searchers.common as sco
import deep_architect.modules as mo
from keras.layers import Dense, Dropout, BatchNormalization

D = hp.Discrete

# the summary description of the search space it is the repetition of
# a cell

def cell(h_opt_drop, h_opt_batchnorm, h_drop_rate, h_activation, h_permutation):
    h_units = D([128, 256, 512])
    return siso_sequential([
        mo.siso_permutation([
            lambda: km(Dense, {"units" : h_units, "activation" : h_activation})),
            lambda: mo.siso_optional(lambda: km(Dropout, {"rate" : h_drop_rate})),
            lambda: mo.siso_optional(lambda: km(BatchNormalization, {}))
        ], h_permutation)
    ])

km = hke.siso_keras_module_from_keras_layer_fn

h_opt_drop = D([0, 1])
h_opt_batchnorm = D([0, 1])
h_permutation = hp.OneOfKFactorial(3)
fn = lambda: cell(h_opt_drop, h_opt_batchnorm, D([0.0, 0.2, 0.5, 0.8]))
search_space_fn = lambda: mo.siso_sequential([
    mo.siso_repeat(fn, D([1, 2, 4]),
    km(Dense, {
        "units" : D([num_classes]),
        "activation" : D["softmax"]}))])


import deep_architect.search_logging as sl

sl.create_search_folderpath('logs', 'logging_tutorial',
    delete_if_exists=True, create_parent_folders=True)

This create an initial folder structure that will progressively be filled by
each of the evaluations.
The basic architecture search loop with a single process is

from deep_architect.searchers.random import RandomSearcher

searcher = RandomSearcher()


# Each evaluation that we call to the user, the searcher returns a model in
the search space that is then passed to the evaluator for training and
computing the validation performance.

The logging funcitonality also gives nice support for managing the folders to
store user information for each evaluation.
For example, let us assume that the user wanted to keep the evaluated models
for each of the evaluations, i.e., for each of the architectures that were done
by the searcher.
This is easily possible, as each evaluation folder contains a folder
dedicated for the user to store whatever might be useful from that evaluation.




# we encourage the reader to

# TODO: some of these things are a bit repeated. I think that it would be a
# good idea.

# NOTE: that there is some redundancy on the search space in the
# sense that some of the transformations may not show up in the model.

# the visualization functionality is very preliminary at this stage, but we
# hope to keep adding more functionality as we work on the project.