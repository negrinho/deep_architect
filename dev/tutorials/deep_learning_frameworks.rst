Framework Agnostic Specification
--------------------------------

One problem that often arises with architecture search research is that in papers
publishing results, the search spaces are underspecified or crucial implementation
details are left out. DeepArchitect is meant to address that problem by allowing
researchers to ground their search space in code that can then be published.
The framework allows the user to clearly outline their search space, rather
than have it be mixed in with the rest of their code for the search.

Another step towards clearly defined search spaces and easily transferable
research code is the concept of specifying the search spaces in a framework
agnostic manner. This allows other researchers to immediately be able to plug
in your search space into their existing code base.

This tutorial outlines how to create and use a framework agnostic search
space, as well as how to use a sampled architecture from the framework agnostic
search space in a specific deep learning framework.

This is a simple specification of a search space using only framework agnostic
parts. Each module used is single input, single output, and chained together
to form the architecture. Note, in this example, only framework agnostic modules
are used, but it is possible to mix framework agnostic and framework specific
modules. You just need to make sure that the backend being used matches the
framework of the framework specific modules. The only prohibited use is that
of framework specific modules from different frameworks.

.. code:: python
    import deep_architect.core as co
    import deep_architect.modules as mo
    import deep_architect.visualization as viz
    from deep_architect.contrib.deep_learning_backend import backend
    from deep_architect.contrib.deep_learning_backend.general_ops import *
    from deep_architect.contrib.misc.datasets.dataset import InMemoryDataset
    from deep_architect.contrib.misc.datasets.loaders import load_cifar10
    from deep_architect.hyperparameters import Discrete as D
    from deep_architect.searchers.common import random_specify


    def get_search_space(num_classes):
        return mo.siso_sequential([
            conv2d(D([32, 64]), D([3, 5]), D([1, 2]), D([True, False])),
            relu(),
            batch_normalization(),
            conv2d(D([32, 64]), D([3, 5]), D([1, 2]), D([True, False])),
            max_pool2d(D([3, 5]), D([1, 2])),
            relu(),
            batch_normalization(),
            dropout(D([.7, .9])),
            conv2d(D([32, 64]), D([3, 5]), D([1, 2]), D([True, False])),
            relu(),
            batch_normalization(),
            global_pool2d(),
            fc_layer(D([num_classes]))
        ])


First, you must set the backend framework to be used. DeepArchitect simply
uses the module implementations specific to the framework being used. If a
module is not implemented for a given framework, a RuntimeError is raised.
The implementations for the framework specific modules are
`here <https://github.com/negrinho/deep_architect/blob/master/deep_architect/contrib/deep_learning_backend>`__.
The four backends currently supported are TENSORFLOW, TENSORFLOW_EAGER,
PYTORCH, and KERAS. In the following example, we use TENSORFLOW_EAGER.

.. code:: python

    backend.set_backend(backend.TENSORFLOW_EAGER)

Now, you need to sample from the search space and specify the hyperparameters
to create a specific architecture. Here, we assign the hyperparameters randomly.

.. code:: python

    ins, outs = get_search_space(10)
    random_specify(outs.values())

You can now view the fully specified architecture.

.. code:: python

    viz.draw_graph(outs.values())

The next step is to load the data. Since CIFAR-10 is small enough, we'll load
it into memory. (Note, Pytorch expects the data to be formatted
channels first)

.. code:: python
    if backend.get_backend() == backend.PYTORCH:
        _, _, _, _, X, y = load_cifar10('data/cifar-10-batches-py/',
                                        data_format='NCHW')
    else:
        _, _, _, _, X, y = load_cifar10('data/cifar-10-batches-py/')
    dataset = InMemoryDataset(X, y, False)

The rest of the tutorial demonstrates how to run a batch of data through the
architecture you just sampled.

.. code:: python

    X_batch, y_batch = dataset.next_batch(16)
    logit_vals = None

Tensorflow
^^^^^^^^^^

First the Tensorflow graph framework.

.. code:: python

    if backend.get_backend() == backend.TENSORFLOW:
        in_dim = list(X_batch.shape[1:])
        import tensorflow as tf
        import deep_architect.helpers.tensorflow_support as htf

    # In order to feed the data through, you need to create placeholders for the
    # data and compile the graph with the placeholders assigned to the input nodes
    # of the graph. The logits tensor is placed in the output dictionary by the
    # framework after the graph is compiled.
    X_pl = tf.placeholder("float", [None] + in_dim)
    y_pl = tf.placeholder("float", [None, 10])
    co.forward({ins['In']: X_pl})
    logits = outs['Out'].val

    # This gets all of the other placeholders needed during training, such as
    # indicators for batch norm and dropout layers
    train_feed, _ = htf.get_feed_dicts(outs.values())
    train_feed.update({X_pl: X_batch, y_pl: y_batch})

    # Now, simply run the graph as you normally would.
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        logit_vals = sess.run(logits, feed_dict=train_feed)

Tensorflow Eager
^^^^^^^^^^^^^^^^

The Tensorflow Eager framework is much simpler to use compared to the normal
Tensorflow framework. Simply assign the input values to the input node of the
architecture, set the architecture to use training mode, and call `co.forward()`.

.. code:: python

    elif backend.get_backend() == backend.TENSORFLOW_EAGER:
        import tensorflow as tf
        import deep_architect.helpers.tensorflow_eager_support as htfe
        tf.enable_eager_execution()
        htfe.set_is_training(outs.values(), True)
        co.forward({ins['In']: tf.constant(X_batch)})
        logit_vals = outs['Out'].val

PyTorch
^^^^^^^

The usage of the PyTorch framework is almost identical to that of the Tensorflow
Eager framework.

.. code:: python

    elif backend.get_backend() == backend.PYTORCH:
        import torch
        import deep_architect.helpers.pytorch_support as hpy
        hpy.train(outs.values())
        co.forward({ins['In']: torch.Tensor(X_batch)})
        logit_vals = outs['Out'].val

Keras
^^^^^
Finally, the Keras framework. This framework requires adding a special input
node to the start of the search space. This input node what is given to the
keras model builder. Also, note that this an example of mixing framework
agnostic and framework specific modules.

.. code:: python

    elif backend.get_backend() == backend.KERAS:
        import keras
        from deep_architect.contrib.deep_learning_backend.keras_ops import input_node
        in_node = input_node()
        ins, outs = mo.siso_sequential([in_node, (ins, outs)])
        _, input_layer = in_node
        co.forward({ins['In']: X.shape[1:]})
        model = keras.Model(inputs=[inp.val for inp in input_layer.values()],
                            outputs=[out.val for out in outs.values()])
        model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])
        logit_vals = model.predict(X_batch)

The logits are stored in `logit_vals`

.. code:: python

    print(logit_vals)
