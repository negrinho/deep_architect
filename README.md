# Overview

<p align="center">
  <img width="400" height="400" src="docs/source/logo/logo.svg">
</p>

*DeepArchitect: Architecture search so easy you'll think it's magic!*

## Introduction

DeepArchitect is a framework for automatically searching over
computational graphs in arbitrary domains.
DeepArchitect was designed with a focus on
**modularity**, **ease of use**, **reusability**, and **extensibility**.
It should be easy for the user to implement new
search spaces and searchers in DeepArchitect.
The core APIs were carefully designed and their interaction carefully thought out.
The design is such that most interfaces (e.g., for search
spaces, searchers, and evaluators) are only lightly coupled, each of them
having well-defined responsibilities.

A machine learning expert has to make a large number of choices to design
a deep learning model for a new problem.
With the development of DeepArchitect, we aim to impact the workflows of
machine learning expert (both researchers and practitioners)
by allowing them to easily express this
design ambiguity and offload it to an automatic search algorithm.
DeepArchitect uses composable and modular operators to express a search
space over computational graphs.
A search algorithm can then sample architectures from the search space with the
goal of maximizing a desired performance metric.

<!-- what is there in store for both researchers and practicioners -->
For researchers, DeepArchitect aims to make architecture search research more
reusable and reproducible by providing them with a
modular framework that they can use to implement
new search algorithms and new search spaces while reusing a large amount of
existing code.
For practitioners, DeepArchitect aims to augment their workflow by providing them
with a tool allowing them to easily write a search space encoding
a large number of desired design choices
and use a search algorithm to automatically find a good architecture.

DeepArchitect has the following **main components**:

* a language for writing composable and expressive search spaces over computational
graphs in arbitrary domains (e.g., Tensorflow, Keras, Pytorch, and even non
deep learning frameworks such as scikit-learn and preprocessing pipelines);
* search algorithms that can be used for arbitrary search spaces;
* logging functionality to easily keep track of the results of a search;
* visualization functionality to explore and inspect logging information
resulting from a search experiment.

<!-- main differences between different tools. -->
Compared to existing work for hyperparameter optimization and architecture
search, DeepArchitect has several differentiating features that makes it suitable
for augmenting existing machine learning workflows.
The main difference between DeepArchitect and current hyperparameter architecture
search tools is that DeepArchitect has better integration, having the hyperparameters
directly related to the computational elements.
This reduces the arbitrary ad-hoc compilation steps where the values of the collection
of hyperparameters have to be mapped to a specific computational graph by the expert.
The main difference between DeepArchitect and other existing research in architecture
search is that DeepArchitect is explicitly concerned about
extensibility, ease of use, and programability,
e.g., we designed a composable and expressive language to write search spaces.
Existing work on architecture search does not have a principled way of accomplish
this, relying on ad-hoc encodings of search spaces, therefore making them
hard to adapt and reuse them in new settings.

## Installation

Run the following code snippet for a local installation:

```
git clone git@github.com:negrinho/darch.git deep_architect
cd deep_architect
pip install -e .
```

After installing DeepArchitect, attempt to run one of the examples to check that no
dependencies are missing, e.g.,
`python examples/framework_starters/main_keras.py`
or
`python examples/mnist_with_logging/main.py --config_filepath examples/mnist_with_logging/configs/debug.json`.
We omitted many of the deep learning framework dependencies to avoid
installing unnecessary software that will not be used by a particular user.

We have included [utils.sh](https://github.com/negrinho/darch/blob/master/utils.sh)
with some useful functionality to develop for DeepArchitect, e.g., to build
documentation, extract code from documentation files, and
build Singularity containers with the development dependencies.

## A minimal DeepArchitect example with Keras

Consider the following short example that we minimally adapt from
[this Keras example](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
by defining a search space of models and sampling a random model from it.
The original example considers a single fixed three-layer neural network
with ReLU activations in the hidden layers and dropout with rate equal to *0.2*.
We constructed a search space by relaxing the number of layers that the network
can have, choosing between sigmoid and ReLU activations, and the number of units that
each dense layer can have. Check the following minimal search space:

```python

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop

import deep_architect.helpers.keras_support as hke
import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
import deep_architect.core as co
import deep_architect.visualization as vi
from deep_architect.searchers.common import random_specify

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

# model = Sequential()
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(num_classes, activation='softmax'))

D = hp.Discrete


def dense(h_units, h_activation):
    return hke.siso_keras_module_from_keras_layer_fn(Dense, {
        'units': h_units,
        'activation': h_activation
    })


def dropout(h_rate):
    return hke.siso_keras_module_from_keras_layer_fn(Dropout, {'rate': h_rate})


def cell(h_units, h_activation, h_rate, h_opt_drop):
    return mo.siso_sequential([
        dense(h_units, h_activation),
        mo.siso_optional(lambda: dropout(h_rate), h_opt_drop)
    ])


def model_search_space():
    h_activation = D(['relu', 'sigmoid'])
    h_rate = D([0.0, 0.25, 0.5])
    h_num_repeats = D([1, 2, 4])
    return mo.siso_sequential([
        mo.siso_repeat(
            lambda: cell(
                D([256, 512, 1024]), h_activation, D([0.2, 0.5, 0.7]), D([0, 1])
            ), h_num_repeats),
        dense(D([num_classes]), D(['softmax']))
    ])


(inputs, outputs) = mo.SearchSpaceFactory(model_search_space).get_search_space()
random_specify(outputs.values())
inputs_val = Input((784,))
co.forward({inputs["In"]: inputs_val})
outputs_val = outputs["Out"].val
vi.draw_graph(outputs.values(), draw_module_hyperparameter_info=False)
model = Model(inputs=inputs_val, outputs=outputs_val)
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

```

<!-- comments on the example. -->
This example is introductory and it is meant to show how to introduce the
absolute minimal architecture search capabilities in an existing Keras example.
In this case, we see that we can compactly express a reasonable
number of structural transformations for the computational graph.
Our search space essentially says that our network will be composed of a
sequential connection of *1*, *2*, or *4* cells, followed by a final dense module that
outputs probabilities over classes.
Each cell is a sub-search space (again, exhibiting the modularity and composability
properties of DeepArchitect). The choice of the type of activation for the dense layer
in the cell search space is shared among all cell search spaces used.
All other hyperparameter of the cell search space are chosen independently for
each occurrence of the cell search space in the sequential connection.

We left the original search space commented out in the code above for the reader
to get a sense of how little code conceptually we need to add to support a
nontrivial search space.
We encourage the reader to think about
how to support such a search space using current
hyperparameter optimization tools or in an ad-hoc manner. For example,
using existing tools,
how much code would be required to encode the search space and sample a random
architecture from it.


<!-- suggestions on going forward. -->
We have not yet discussed some important aspects of DeepArchitect.
For example, more complex searchers are able to explore the search
space in a more purposeful and sample efficient manner, and the logging
functionality is useful to keep a record of the performance of different architectures.
These and other aspects are better covered in existing tutorials.
We recommend looking at the tour of the repository for deciding what to read next.
[This](https://github.com/negrinho/darch/blob/master/examples/mnist_with_logging/main.py)
slightly more complex example shows the use of the search and logging functionalities.
The [framework starters](https://github.com/negrinho/darch/tree/master/examples/framework_starters)
are minimal architecture search examples
in DeepArchitect across different deep learning frameworks.
These should be straightforward to adapt to implement your custom examples.

## Framework components

In this section, we briefly cover the principles that guided the design of
DeepArchitect.
Some of the main concepts that we deal with in DeepArchitect are:

* **Search spaces**: Search spaces are constructed by putting together modules
(both basic and substitution modules) and
hyperparameters (independent and dependent).
Modules are composed of inputs, outputs, and hyperparameters.
The search spaces are often passed around as a dictionary of inputs and a dictionary
of outputs, allowing us to seamlessly deal with search spaces with multiple modules
and easily combine them. In designing substitution modules, we make
extensive use of ideas of delayed evaluation.
An important concept in search spaces is the notion of graph transitions with
value assignments to the independent hyperparameters.
Good references to peruse to get acquainted with these ideas are
[deep_architect/core.py](https://github.com/negrinho/darch/blob/master/deep_architect/core.py) and
[deep_architect/modules.py](https://github.com/negrinho/darch/blob/master/deep_architect/modules.py).

* **Searchers**:  Searchers interact with search spaces through a simple API.
A searcher samples a model from the search space by assigning values to each of the independent
hyperparameters, until there are no more unassigned independent hyperparameters.
A searcher object is instantiated with a search space.
The base API for the searcher has two methods `sample`, which samples
an architecture from the search space, and `update`, which takes the results
for a particular sampled architecture and updates the state of the searcher.
The reader can look at
[deep_architect/searchers/common.py](https://github.com/negrinho/darch/blob/master/deep_architect/searchers/common.py),
[deep_architect/searchers/random.py](https://github.com/negrinho/darch/blob/master/deep_architect/searchers/random.py), and
[deep_architect/searchers/smbo.py](https://github.com/negrinho/darch/blob/master/deep_architect/searchers/smbo.py)
for examples of the common API.
It is also worth to look at
[deep_architect/core.py](https://github.com/negrinho/darch/blob/master/deep_architect/core.py) and
for some of the traversal functionality that allows to traverse the independent
hyperparameters in the graph.

* **Evaluators**:  Evaluators take a sampled architecture from the search space and
compute some notion of performance for that particular architecture. Evaluators
often have a single method named `eval` that
takes the definition of an architecture to evaluate and
returns a dictionary with the evaluation results.
In the simplest case, there is a single performance metric of interest.
See [here](https://github.com/negrinho/darch/blob/master/deep_architect/contrib/misc/evaluators/tensorflow/classification.py)
for an example implementation of an evaluator.

* **Logging**:  When we run an architecture search workload, we evaluate
multiple architectures in the search space. To keep track of the generated
results, we designed a folder structure that maintains a single folder per
evaluation. This structure allows us to keep the information about the
configuration evaluated, the results for that configuration, and additional
information that the user may wish to maintain for that configuration, e.g.,
example predictions or the model checkpoints.
Most of the logging functionality can be found in
[deep_architect/search_logging.py](https://github.com/negrinho/darch/blob/master/deep_architect/search_logging.py).
A simple example
of how the logging functionality is used can be found
[here](https://github.com/negrinho/darch/blob/master/examples/mnist_with_logging/main.py).

* **Visualization**:  The visualization functionality allows us to inspect the
structure of a search space and to visualize graph transitions resulting from
assigning values to the independent hyperparameters.
These visualization can be useful for debugging, e.g., checking if the search
space is encoding the expected design choices.
There are also visualizations to callibrate the necessary
evaluation effort to recover the correct performance ordering for architectures
in the search space, e.g., how many epochs do we need to invest to identify the best
architecture or make sure that the best architecture is at least in the top 5.
Good references for this functionality can be found in
[deep_architect/visualization.py](https://github.com/negrinho/darch/blob/master/deep_architect/visualization.py).

## Main folder structure

The most important source files in the repository live in
the [deep_architect folder](https://github.com/negrinho/darch/tree/master/deep_architect),
excluding the contrib folder, which contains auxiliary code
to the framework that is potentially useful, but that we do not necessarily want
to maintain. We recommend the user to peruse it. We also recommend the user
to read the tutorials as they cover much of the information needed to extend the
framework.
See below for a high-level tour of the repo.

* [core.py](https://github.com/negrinho/darch/blob/master/deep_architect/core.py):
    Most important classes to define search spaces.
* [hyperparameters.py](https://github.com/negrinho/darch/blob/master/deep_architect/hyperparameters.py):
    Basic hyperparameters and auxiliary hyperaprameter sharer class.
* [modules.py](https://github.com/negrinho/darch/blob/master/deep_architect/modules.py):
    Definition of substitution modules along with some auxiliary
abstract functionality to connect modules or construct larger search spaces
from simpler search spaces.
* [search_logging.py](https://github.com/negrinho/darch/blob/master/deep_architect/search_logging.py):
    functionality to keep track of the results of the architecture
search process, allowing to maintain structured folders for each  search
experiment.
* [utils.py](https://github.com/negrinho/darch/blob/master/deep_architect/utils.py):
    general utility functions that are not directly related to architecture
search, but are useful in many contexts such as logging and visualization.
* [visualization.py](https://github.com/negrinho/darch/blob/master/deep_architect/visualization.py):
    Simple visualization functionality used to visualize search spaces as graph
    or a sequence of graphs.

We also have a few folders in the deep_architect folder.

* [communicators](https://github.com/negrinho/darch/tree/master/deep_architect/communicators):
    simple functionality to communicate between master and worker
processes to relay the evaluation of an architecture and retrieve the results
once finished.
* [contrib](https://github.com/negrinho/darch/tree/master/deep_architect/contrib):
    useful functionality that uses DeepArchitect and that it will not
be necessarily maintained over time, but that users of DeepArchitect may
find useful in their own examples. Contributions by the community will live in this folder.
See [here](https://github.com/negrinho/darch/blob/master/CONTRIBUTING.md)
for an in-depth explanation for the rationale behind the project organization and
the contrib folder.
* [helpers](https://github.com/negrinho/darch/tree/master/deep_architect/helpers):
    helpers for different frameworks that we support. This allows us to take
the base functionality defined in
[core.py](https://github.com/negrinho/darch/blob/master/deep_architect/core.py)
and expand it to provide compilation
functionality for computational graphs in different frameworks. It should be
instructive to compare support for different frameworks. One file in the folder
per framework.
* [searchers](https://github.com/negrinho/darch/tree/master/deep_architect/searchers):
    different searchers that can be used on top of the search spaces
defined in DeepArchitect. One searcher per file to maintain a reasonable degree
of separation.
* [surrogates](https://github.com/negrinho/darch/tree/master/deep_architect/surrogates):
    different surrogate functions over architectures in the search space.
searchers based on sequential model based optimization are used frequently in
DeepArchitect.

## Roadmap for the future

Going forward, the core authors of DeepArchitect expect to continue
extending and maintaining the codebase and use it for it their own research in architecture
search and deep learning.

The community will have a fundamental role in extending DeepArchitect,
for example, authors from existing architecture search algorithms
can implement their algorithms in DeepArchitect.
This will allow the community to compare algorithms on the same footing and
to measure progress more reliably.
New search spaces for new tasks can be implemented and made available in DeepArchitect,
allowing users to use them (either directly or in the construction of new search spaces),
in their own experiments. New evaluators can be implemented,
making sure that it possible to use the same evaluator for the other search
spaces for the same task.
The visualization functionality can be improved by
adding new, more interesting, visualizations leveraging the fact that
architecture search workloads train a large number of models.
Ensembling capabilities may be added to
DeepArchitect to easily construct essembles from the many models that were
explored as a result of the architecture search workload.

The reusability, composability, and extensibility properties of DeepArchitect
will be fundamental going forward. We ask willing contributors to check the
[contributing guide](https://github.com/negrinho/darch/blob/master/CONTRIBUTING.md).
We recommend using GitHub issues to engage with the authors
of DeepArchitect and ask clarification and usage questions.
Please, check if your question has already been answered before creating a new issue.

## Reaching out

You can reach the main researcher behind of DeepArchitect at negrinho@cs.cmu.edu.
If you tweet about DeepArchitect, use the tag `#DeepArchitect` and/or
mention me ([\@rmpnegrinho](https://twitter.com/rmpnegrinho)) in the tweet.
For bug reports, questions, and suggestions, use
[Github issues](https://github.com/negrinho/darch/issues).

## License

DeepArchitect is licensed under the MIT license as found
[here](https://github.com/negrinho/darch/blob/master/LICENSE.md).
Contributors agree to license their contributions under the MIT license.

## Contributors and acknowledgments

The main researcher behind DeepArchitect is [Renato Negrinho](https://www.cs.cmu.edu/~negrinho/).
[Daniel Ferreira](https://github.com/dcferreira) played an important role in designing the
APIs of DeepArchitect through discussions and contributions.
This work benefited immensely from the involvement and contributions of talented
undergraduate students at CMU
([Darshan Patil](https://github.com/dapatil211),
[Max Le](https://github.com/lethenghia18),
[Kirielle Singajarah](https://github.com/ksingarajah),
[Zejie Ai](https://github.com/aizjForever),
[Yiming Zhao](https://github.com/startrails98),
[Emilio Arroyo-Fang](https://github.com/negrinho/darch/commits?author=fizzxed)).
This work benefited greatly from discussions with both faculty (Geoff Gordon, Matt
Gormley, Graham Neubig, Carolyn Rose, Ruslan Salakhutdinov, Eric Xing, and Xue Liu), and
fellow PhD students (Zhiting Hu, Willie Neiswanger, Christoph Dann, and Matt Barnes).
