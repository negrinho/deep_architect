###${MARKDOWN}
# This tutorial demonstrates how to train a Keras model that is specified using
# the DeepArchitect framework.

import keras
from keras import Model

from deep_architect import core as co
from deep_architect import modules as mo
from deep_architect.searchers.common import random_specify
from deep_architect.hyperparameters import Discrete as D
from deep_architect.contrib.misc.datasets.loaders import load_mnist

from deep_architect.contrib.deep_learning_backend.keras_ops import (
    input_node, conv2d, max_pool2d, relu, batch_normalization,
    global_pool2d, fc_layer)

# First, we load the data. For this tutorial, we will used the MNIST dataset.
X_train, y_train, X_val, y_val, X_test, y_test = load_mnist('data/mnist', normalize_range=True)

# Now we create the keras search space. In addition to the actual model
# components, we need to add an input module at the beginning to accomodate for
# training in keras. (Note, aside from the input module, every module used in
# the search space has the same signature as those from the framework agnostic
# modules. This allows us to be able to easily change the framework being used.)
in_node = input_node()
ins, outs = mo.siso_sequential([
    in_node,
    conv2d(D([32, 64]), D([3, 5]), D([1, 2]), D([True, False])),
    max_pool2d(D([3, 5]), D([1, 2])),
    relu(),
    batch_normalization(),
    global_pool2d(),
    fc_layer(D([10]))
])

# We will randomly specify the hyperparameters for this architecture and compile
# it.
random_specify(outs.values())
co.forward({ins['In'] : X_train.shape[1:]})
_, input_layer = in_node
model = Model(
    inputs=[inp.val for inp in input_layer.values()],
    outputs=[out.val for out in outs.values()])
model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

# Now, we can train the model as normal in Keras.
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    shuffle=True, batch_size=32, epochs=10)
