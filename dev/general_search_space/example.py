import darch.modules as mo
import darch.core as co
from darch.hyperparameters import Discrete as D
from dev.general_search_space.general_ops import *
from dev.general_search_space import backend
from darch.searchers import random_specify
import darch.visualization as viz
from darch.contrib.useful.datasets.loaders import load_cifar10
from darch.contrib.useful.datasets.dataset import InMemoryDataset
from dev.keras.keras_ops import input_node

def get_search_space(num_classes):
    return mo.siso_sequential([
        conv2d(D([32, 64]), D([3, 5]), D([1, 2]), D([True, False])),
        relu(),
        batch_normalization(),
        conv2d(D([32, 64]), D([3, 5]), D([1, 2]), D([True, False])),
        max_pool2d(D([3, 5]), D([1, 2])),
        relu(),
        batch_normalization(),
        conv2d(D([32, 64]), D([3, 5]), D([1, 2]), D([True, False])),
        relu(),
        batch_normalization(),
        global_pool2d(),
        fc_layer(D([num_classes]))
    ])

def main():
    backend.set_backend(backend.KERAS)
    ins, outs = get_search_space(10)
    random_specify(outs.values())
    viz.draw_graph(outs.values())
    if backend.get_backend() == backend.PYTORCH:
        _, _, _, _, X, y = load_cifar10('data/cifar10/cifar-10-batches-py/', data_format='NCHW')
    else:
        _, _, _, _, X, y = load_cifar10('data/cifar10/cifar-10-batches-py/')
    dataset = InMemoryDataset(X, y, False)
    in_dim = list(dataset.next_batch(1)[0].shape[1:])
    X_batch, y_batch = dataset.next_batch(16)

    if backend.get_backend() == backend.TENSORFLOW:
        import tensorflow as tf
        import darch.helpers.tensorflow as htf
        X_pl = tf.placeholder("float", [None] + in_dim)
        y_pl = tf.placeholder("float", [None, 10])
        co.forward({ins['In'] : X_pl})
        logits = outs['Out'].val
        train_feed, _ = htf.get_feed_dicts(outs.values())
        train_feed.update({X_pl: X_batch, y_pl: y_batch})
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            logs = sess.run(logits, feed_dict=train_feed)
            print logs
    elif backend.get_backend() == backend.TENSORFLOW_EAGER:
        import tensorflow as tf
        co.forward({ins['In'] : tf.constant(X_batch)})
        logs = outs['Out'].val
        print logs
    elif backend.get_backend() == backend.PYTORCH:
        import torch
        co.forward({ins['In'] : torch.Tensor(X_batch)})
        logs = outs['Out'].val
        print logs
    elif backend.get_backend() == backend.KERAS:
        import keras
        in_node = input_node()
        ins, outs = mo.siso_sequential([in_node, (ins, outs)])
        _, input_layer = in_node
        co.forward({ins['In'] : X.shape[1:]})
        model = keras.Model(
            inputs=[inp.val for inp in input_layer.values()],
            outputs=[out.val for out in outs.values()])
        model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
        logs = model.predict(X_batch)
        print logs

if __name__ == '__main__':
    main()