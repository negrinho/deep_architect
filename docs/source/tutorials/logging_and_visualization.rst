
Logging and visualization
-------------------------

Logging and visualization are important aspects of an architecture search workflow.
Architecture search provides tremendous opportunity to create insightful visualizations
based on architecture search results.

The logging functionality in DeepArchitect allows us to create a folder for
a search experiment.
This search log folder contains a folder for each evaluation done during search.
Each evaluation folder contains a fixed component and a component
that is specified by the user.
The fixed component contains a file with the hyperparameters values that define
the architecture in the search space that is being considered in the search
experiment and a file with the results obtained for that architecture.
Both of these files are represented as JSON files in disk.
The user component allows the user to store any information that may be of
interest for each particular architecture, e.g., parameters for the evaluated model
or model predictions on examples of the validation set.
The logging functionality makes it convenient to manage this folder structure
with a single subfolder per evaluation.
The log folder for the whole search experiment keeps user information
at the search level. For example, if we want to keep checkpoints on the searcher
as the search progresses.

We point the reader to
`examples/mnist_with_logging <https://github.com/negrinho/darch/blob/master/examples/mnist_with_logging/main.py>`__
for an example use of logging, and to
`deep_architect/search_logging.py <https://github.com/negrinho/darch/blob/master/deep_architect/search_logging.py>`__
for the API definitions.
We recommend the reader to go through these references
to get a better grasp of the logging functionality.

Starting Keras example to adapt for logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us start with a Keras MNIST example (copied from the Keras website),
and adapt to get a DeepArchitect example using the logging functionality.

.. code:: python

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
    epochs = 1

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
    print(y_train.shape, y_test.shape)
    model.summary()

    model.compile(
        loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

This is the simplest MNIST example that we can get from the Keras examples
website.

Evaluator and search space definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We will adapt this example to create a DeepArchitect example showcasing some of
the logging and visualization functionalities.

It does not make sense to use the test data as validation data, so we will
create a small validation set out of training set.

.. code:: python

    import deep_architect.core as co
    from keras.layers import Input
    from keras.models import Model


    class Evaluator:

        def __init__(self, batch_size, epochs):
            self.batch_size = 128
            self.num_classes = 10
            self.epochs = 1

            # the data, split between train and test sets
            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            x_train = x_train.reshape(60000, 784)
            x_test = x_test.reshape(10000, 784)
            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            num_val = 10000
            x_train, x_val = (x_train[:num_val], x_train[num_val:])
            y_train, y_val = (y_train[:num_val], y_train[num_val:])
            self.x_train = x_train
            self.y_train = y_train
            self.x_val = x_val
            self.y_val = y_val
            self.x_test = x_test
            self.y_test = y_test
            self.last_model = None

        def eval(self, inputs, outputs):
            x = Input((784,), dtype='float32')
            co.forward({inputs["In"]: x})
            y = outputs["Out"].val
            model = Model(inputs=x, outputs=y)

            model.summary()

            model.compile(
                loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])

            history = model.fit(
                self.x_train,
                self.y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=1)
            self.last_model = model
            train_metrics = model.evaluate(self.x_train, self.y_train, verbose=0)
            val_metrics = model.evaluate(self.x_val, self.y_val, verbose=0)
            test_metrics = model.evaluate(self.x_test, self.y_test, verbose=0)
            return {
                "train_loss": train_metrics[0],
                "validation_loss": val_metrics[0],
                "test_loss": test_metrics[0],
                "train_accuracy": train_metrics[1],
                "validation_accuracy": val_metrics[1],
                "test_accuracy": test_metrics[1],
                "num_parameters": model.count_params(),
            }


    import deep_architect.helpers.keras as hke
    import deep_architect.hyperparameters as hp
    import deep_architect.searchers.common as sco
    import deep_architect.modules as mo
    from keras.layers import Dense, Dropout, BatchNormalization

    D = hp.Discrete

    km = hke.siso_keras_module_from_keras_layer_fn


    def cell(h_opt_drop, h_opt_batchnorm, h_drop_rate, h_activation, h_permutation):
        h_units = D([128, 256, 512])
        return mo.siso_sequential([
            mo.siso_permutation(
                [
                    lambda: km(Dense, {
                        "units": h_units,
                        "activation": h_activation
                    }),  #
                    lambda: mo.siso_optional(
                        lambda: km(Dropout, {"rate": h_drop_rate}), h_opt_drop),
                    lambda: mo.siso_optional(  #
                        lambda: km(BatchNormalization, {}), h_opt_batchnorm)
                ],
                h_permutation)
        ])


    def model_search_space():
        h_opt_drop = D([0, 1])
        h_opt_batchnorm = D([0, 1])
        h_permutation = hp.OneOfKFactorial(3)
        h_activation = D(["relu", "tanh", "elu"])
        fn = lambda: cell(h_opt_drop, h_opt_batchnorm, D([0.0, 0.2, 0.5, 0.8]),
                          h_activation, h_permutation)
        return mo.siso_sequential([
            mo.siso_repeat(fn, D([1, 2, 4])),
            km(Dense, {
                "units": D([num_classes]),
                "activation": D(["softmax"])
            })
        ])


    search_space_fn = mo.SearchSpaceFactory(model_search_space).get_search_space

Main search loop with logging
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This create an initial folder structure that will be progressively filled by
each of the evaluations. The basic architecture search loop with a single process
is as follows:

.. code:: python

    from deep_architect.searchers.mcts import MCTSSearcher
    import deep_architect.search_logging as sl
    import deep_architect.visualization as vi
    import deep_architect.utils as ut

    searcher = MCTSSearcher(search_space_fn)
    evaluator = Evaluator(batch_size, epochs)
    num_samples = 3

    sl.create_search_folderpath(
        'logs',
        'logging_tutorial',
        delete_if_exists=True,
        create_parent_folders=True)

    for evaluation_id in range(num_samples):
        (inputs, outputs, hyperp_value_lst, searcher_eval_token) = searcher.sample()
        results = evaluator.eval(inputs, outputs)
        eval_logger = sl.EvaluationLogger(
            'logs', 'logging_tutorial', evaluation_id, abort_if_exists=True)
        eval_logger.log_config(hyperp_value_lst, searcher_eval_token)
        eval_logger.log_results(results)
        user_folderpath = eval_logger.get_evaluation_data_folderpath()
        vi.draw_graph(
            outputs.values(),
            draw_module_hyperparameter_info=False,
            out_folderpath=user_folderpath)
        model_filepath = ut.join_paths([user_folderpath, 'model.h5'])
        evaluator.last_model.save(model_filepath)
        searcher.update(results["validation_accuracy"], searcher_eval_token)

The above code samples and evaluates three architectures from the search space.
The results, the corresponding graph, and the saved models are logged to each of the evaluation
folders. Typically, we may not want to save the weights for all the architectures
sampled during training as this will lead to large amount of data being kept,
with only a few ones being of interest to the user, then perhaps different logic
should be used to maintain these models.

After running this code, we ask the reader to explore the resulting
log folder to get a sense of the information that is kept.
We made the resulting folder available
`here <https://www.cs.cmu.edu/~negrinho/deep_architect/logging_tutorial.zip>`__ in case the reader does not
wish to run the code locally, but still wishes to inspect the resulting
search log folder.

Concluding remarks
^^^^^^^^^^^^^^^^^^

Log folders are useful for visualization.
Architecture search allows us to try many of the
different architectures and explore different characteristics on each of them.
We may set the search space with the goal of exploring what
characteristics lead to better performance. Architecture search, and
more specifically, DeepArchitect and the workflow that we suggest allows us to
formulate many of these questions easily and explore the results to gain insight.
We encourage users of DeepArchitect to think about interesting visualizations
that can be constructed using architecture search workflows.
