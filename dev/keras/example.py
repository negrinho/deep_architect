import keras
from keras import Model

from darch import core as co
from darch import modules as mo
from darch.searchers import random_specify
from darch.hyperparameters import Discrete as D
from darch.contrib.useful.datasets.loaders import load_mnist

from dev.keras.keras_ops import (
    input_node, conv2d, max_pool2d, relu, batch_normalization, 
    global_pool2d, fc_layer)

def train_keras():
    #load the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist('data/mnist', normalize_range=True)

    # create the search space
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

    # specify the hyperparameters of the search space
    # and create the relevant layers
    random_specify(outs.values())
    co.forward({ins['In'] : X_train.shape[1:]})


    # compile the model
    _, input_layer = in_node
    model = Model(
        inputs=[inp.val for inp in input_layer.values()],
        outputs=[out.val for out in outs.values()])
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    # train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        shuffle=True, batch_size=32, epochs=10)

if __name__ == '__main__':
    train_keras()
