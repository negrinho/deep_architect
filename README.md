# DeepArchitect

<!-- NOTE: some of the goals of this model is to show that it is possible
to develop a framework. -->

<!-- point to examples of reimplementations of search spaces in this framework -->
<!-- show that searchers can use this model very frequently. -->

<!-- TODO: mention programmability of architecture search . -->

<!-- talk about how github issues are the right place to address some of the
comments about representation. -->

<!-- allows the user to easily experiments over complex architecture search spaces.  -->

*Architecture search so easy that you'll think it's magic!*

DeepArchitect is a carefully designed framework for automatically searching over computational graphs.
DeepArchitect is made of components such as a language for writing composable and
expressive search spaces over computational graphs in arbitrary domains
(e.g., Tensorflow, Keras, Pytorch, and even non deep learning frameworks, such as Scikit Learn and preprocessing pipelines),
general search algorithms that can be used in arbitrary domains,
logging functionality to easily keep track of the results of a search,
visualization functionality to explore and inspect the logs that resulted from a search.
DeepArchitect aims to impact the workflows of both researchers and practioneers
by reducing the burden resulting from the large number of arbitrary choices that
have to be made to design a deep learning model for a problem.
As it is most often currently done, instead of writing down a single model
(or a search space in an ad-hoc manner),
DeepArchitect uses composable and modular operators to express a search
space that is then passed to a search algorithm that samples
architectures from it with the goal of maximizing a desired performance metric.
The modularity and composability properties make code written using DeepArchitect
extremely reusable, often leading to the user only having to write a small amount of
code for the desired use case.

<!-- main differences between different tools. -->
Compared to existing work for hyperparameter optimization and architecture
search, DeepArchitect has several differentiating features that make it suitable
to augment or replace existing tools.
The main differences between DeepArchitect and hyperparameter optimization
tools is that DeepArchitect is more integrated, having the hyperparameters
directly related to the computational elements. This means that there is less for
users to develop their own correspondence between encoding of search spaces and
resulting compiled models.
The main difference between DeepArchitect and other research in architecture
search is that most of this research is not highly concerned about extensibility,
ease of use, and programability, e.g., the search spaces are often hard-coded and hard to
adapt to or reuse them for new settings. This is not the case with DeepArchitect, allowing the
user, for example, to run the same searcher on arbitrary search spaces, and run the
same search space on arbitrary searchers.

<!-- what is there in store for both researchers and practicioners -->
For researchers, DeepArchitect aims to make architecture search research more
reusable by providing researchers with a framework that they can use to implement
new search algorithms while reusing a large amount of code.
For practioners, DeepArchitect aims to ease the burden resulting from
number of arbitrary choices that the practioner has to make to write down a model
over architectures.
For practicioners, DeepArchitect aims to augment their workflow by providing
a tool that removes a large fraction of this overwhelming arbitrariness by
encoding these choices directly in a search space and letting a search algorithm
take care of finding an architecture.

DeepArchitect was designed with modularity, ease of use, resusability, and
extensibility in mind. It should be easy for the user to implement a new
search space or a new searcher in the framowork. The core APIs were carefully
designed and their interaction carefully thought out. The design is such that
most interfaces (e.g., for search spaces, searchers, and evaluators) are only
lightly coupled and are mostly orthogonal, meaning that each of them as
well defined responsibilities and concerns.

DeepArchitect mostly stays out of your way, allowing you to use a subset of the
framework and expand it as needed.
For example, it is very simple to set up an example with random search
over an arbitrary search space.
The main focus of DeepArchitect is on programability; it should be easy
for the user to express complex search spaces over computational graphs
using
Search spaces and searchers are very loosely tied to any particular computational
framework, as the language to write down search spaces over architectures
works across different computational frameworks

Consider a very minimal functionality exemplifying how to minimally adapt a
basic Keras examples to sample a model from a search space rather than using
a single fixed model upfront.

# A minimal DeepArchitect example by adapting a Keras example

Consider the following short example that we minimally adapt from
[this Keras example](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)
by defining a search space of models and sampling a random model from it.
The original example considers a single fixed three-layer neural network
with ReLU activations in the hidden layers and dropout with rate equal to 0.2.
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

import deep_architect.helpers.keras as hke
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
absolute minimal architecture search capabilities in an existing Keras examples.
In this case, we see that the changes necessary to express a reasonable
number of structural transformations to the computational graph is very compact.
Our search space essentially says that our network will be composed of a
sequential connection of 1, 2, or 4 cells, followed by a final dense module that
outputs probabilities over classes.
Each cell is a subsearch space (again, justifying the modularity and composability
properties of DeepArchitect). The choice of the activation for the dense layer
in the cell search space is shared

Each of the cell search spaces is composed of a
dense layer for which we cho

<!-- suggestions on going forward. -->
There are many aspects that we have not exemplified here, such as logging,
searching, multiple input multiple output modules, and so forth.
This is example is meant to give the reader a taste of easy is to augment
existing examples with architecture search capabilities. We point the
reader to the example for a more concrete details on how a typical
example using the framework looks like. This example still uses a single process
(i.e., both the searcher and the evaluators), which should be a reasonable computational setting to start using the
<!-- check thi... -->


We left the original search space commented out in the code above for the reader
to get a sense of how little code conceptually we need to add to support a
nontrivial search space. It may be worth to take a minute to think about how
would the user go about supporting such a search space using current
hyperparameter optimization tools or in an ad-hoc manner. For example, if
we just wanted to sample a random architecture from this search space, how
much code would this entail if we had to encode the search space using typical existing tools.

There are a few important aspects of the framework that are not represented but
that are useful for the reader to eventually get acquainted with them.
For example, more sophisticated searchers are useful to explore the search
space in a more purposeful and sample efficient manner, and the logging
functionality is useful to keep a record of the performance of different architectures.
These and other aspects are better covered in existing tutorials.
We recommend looking at the tour of the repository for deciding what to read next.
[This](...) slighly more complex example also includes the usage of the search and
logging functionality.

# Design

In this section, we briefly cover the principles that guided the design of
DeepArchitect.
Some of the main concepts that we deal in DeepArchitect are

* Search spaces: search spaces are constructed by putting together modules (both basic and substitution) and
hyperparameters (independent and dependent). These are represented through the
connections of the modules, which have inputs, outputs, and hyperparameters.
The search spaces are often passed around a dictionary of inputs and a dictionary
of outputs, allowing us to abstract whether the search space is composed of a
single or multiple modules. Working directly with these dictionaries allows us
to compose search spaces easily too. In designing substitution modules, we make
extensive use of ideas of delayed evaluation.
A important concept in search spaces is the notion of graph transitions with
value assignments to the independent hyperparameters.
Good references to peruse to
get more acquainted with these ideas are (core.py, modules.py).

* Searchers: searchers interact with a search through a very simple interface.
Regardless of the each search space that a searcher is dealing with, the searcher
samples a model from the search by assigning values to each of the independent
hyperparameter of the search, until there are no more independent hyperparameters to
assign. A searcher object is created with a specific search space.
The base API for the searcher has two methods `sample`, which samples
an architecture from the search space, and `update`, which takes the results
for a particular sampled architecture and updates the state of the searcher to
take the results of this architecture into consideration.
The reader can look at (point to searchers.common, searchers.random, searchers.smbo
for some examples of the common API. It is also worth to look at core.py for
some of the traversal functionality that allows to traverse the independent
hyperparameters in the graph. Any

* Evaluators: evaluators take a sampled architecture from the search space and
compute some notion of performance for that particular architecture. These
contain a single method that returns a dictionary with characteristics for the
architecture evaluated. In the simplest case, there is a single metric that
can be used as a performance metric.

* Logging: When we run an architecture search workload, we are going to evaluate
multiple architectures in the search space. To keep track of the generated
information, we designed a folder structure that maintains a single folder per
evaluation. This structure allows us to keep the information about the
configuration evaluated, the results for that configuration, but also additional
information that the user may wish to maintain for that architecture, e.g.,
example predictions on the validation set or the save model. Most of the
logging functionality can be found in search_logging.py. A simple example
of how the logging functionality is used can be found mnist_with_logging/main.py.

* Visualization: Visualization functionality allows us to visualize the structure
of the search space and to visualize graph transitions that result from
assigning values to the independent hyperparameters. This allows the user to
get a sense if the search space that was defined is encoding the expected
search space or not. Another possible visualizations of interest is related to
calibration of the effort necessary to determine an appropriate ordering of
the architectures, e.g., how many epochs to we need to invest to identify the best
architecture or make sure that the best architecture is at least in the top 5.
Good references about this functionality live in the visualization.py and
contrib/misc/callibration_utils.py.


The most important source files in the repository live in deep_architect subtree
excluding deep_architet/contrib. deep_architect/contrib contains code auxiliary
to the framework, which is potentially useful, but we do not necessarily want
to maintain. We recommend the user to peruse it. See below for a high-level tour
of the repo.

* core.py: Most important classes to define search spaces.
* hyperparameters.py: Basic hyperparameters and auxiliary hyperaprameter sharer class.
* modules.py: Definition of subsitution modules along with some auxiliary
abstract functionality to connect modules or construct more larger search spaces
from simpler search spaces.
* search_logging.py: functionality to keep track of the results of the architecture
search process, allowing to maintain strucutured folders for each architecture search
process.
* utils.py: general utility functions that are not directly related to architecture
search, but are useful in many contextes such as logging and visualization.
* visualization.py: Simple visualization functionality used to visualize
search spaces in graph form.

We also have a few folders in the deep_architect folder.
* communicators: simple functionality to communicate between master and worker
processes to relay the evaluation of an architecture and retrieve the results
once done.

* contrib: useful functionality that uses DeepArchitect and that it will not
be necessarily maintained over time, but that users of DeepArchitect may
find useful in their own examples. Contributions by the community will live in this folder.
See (TODO: link to contributing) for an
in-depth explanation for the rationale behind the project organization and
the contrib folder.

* helpers: helpers for different frameworks that we support. This allows to take
the base functionality defined in core.py and expand it to provide compilation
functionality for computational graphs in different frameworks. It should be
instructive to compare support for different frameworks. One file in the folder
per framework.

* searchers: different searchers that can be used in top of the search spaces
defined in DeepArchitect. One searcher per file to maintain a reasonable degree
of separation.

* surrogates: different surrogate function over architectures in the search space.
searchers based on sequential model based optimization are used frequently in
DeepArchitect.


Outside the deep_architect folder, be sure to also check the tutorials and read
the API documentation.

<!-- Looking into the future -->
# Looking into the future

Going into the future, the core authors of DeepArchitect expect to continue
extending and maintaining the codebase and use it for it their own research in architecture
search, and deep learning in general.

The community will have a fundamental role to play in moving forward with
DeepArchitect, for example, authors from existing architecture search algorithms
can implement their algorithms in DeepArchitect such that we can compare them
with existing algorithms to measure progress more reliably. New search
spaces for new tasks can be implemented and can be made available in DeepArchitect,
allowing users to build on them by composing new search spaces from them,
or just using them in their own experiments. New evaluators can be implemented,
making sure that it possible to use the same evaluator for the other search
spaces in the same problem. The visualization functionality can be improved by
adding new visualizations leveraging the fact that architecture search
workloads try many many models. Essembling capabilities may be added to
DeepArchitect to easily construct essembles from the many models that were
explored as a result of the architecture search workload.

The reusability, composability, and extensibility properties of the DeepArchitect
will be fundamental going forward. We ask the willing contributor to check the
contributing guide (TODO: put a link to it). There we detail some of the rationale behind the folder
organization of the project and where will future contributions live.
We recommend using GitHub issues to engage with the authors
of DeepArchitect and ask clarification and usage questions. Please, check your
question has already been answered before asking it.

<!-- # Social Media

You can reach the main architect of DeepArchitect on Twitter at . If you tweet about
DeepArchitect, use the tag #DeepArchitect and/or mention me in the tweet
@rmpnegrinho. Looking forward to feedback about the framework. I

This work benefited imensely from a group of excited

Graham Neubig,

This work was done in part when the main contributor

This project benefited from the discussions with many people: Geoff Gordon, Matt
Gormley, Willie Neiswanger, Ruslan Salakutinov, Eric Xing, Xue Liu, ...

Daniel Ferreira
Max Le
Darshan Patil
Kirielle Singajarah

Geoff Gordon
Matt Gormley
Yiming Zhao
Emilio Arroyo-Fang -->

# License

MIT license

TODO: add some fast pointers.

We made a consistent effort to make the code very readable, so we recommend
users of DeepArchitect to read the source when looking for specific about what
the code is doing.

TODO: say that feedback about implementations should be done in the model.

TODO: if there is no tutorial for a particular aspect of the framework, the
best way of getting acquainted with it is to read the source for an existing
example to get an idea of what is happening and then, if necessary, open a
issue about the use case. Additional tutorials and documentation will be created
as we go along.

If you want to support my PhD and/or support DeepArchitect with compute
and/or money, reach out to negrinho@cs.cmu.edu.


# Why use DeepArchitect

Recently, there was a stream of research in architecture search.
Unfortunately, much of this research builds their own systems from scratch,
severely reducing the possibility of building on them.
In DeepArchitect, we designed a careful API to build modular architecture search systems.
Our framework allows the researcher to identify a specific aspect of architecture
search that the researcher wants to explore and focus on it while reusing
available components from other models.
This simultaneously improves reusability and reproducibility of the researcher's
work, making it readily available to the community and allowing fairer and
easier comparisons to previous work and future work.

- Well designed APIs:
When designing DeepArchitect, we paid special attention to the design of the
different APIs to make sure that the programmer would express concepts at the
right level of abstraction. This means that compared

- Only wrapper code:
- Modular:
- Easy to define search spaces:

