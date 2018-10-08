# DeepArchitect

<!-- NOTE: some of the goals of this model is to show that it is possible
to develop a framework. -->

DeepArchitect is a carefully designed framework for automatically searching over computational graphs.
DeepArchitect is made of components such as a language for writing composable and
expressive search spaces over computational graphs in arbitrary domains
(e.g., Tensorflow, Keras, Pytorch, or even non deep learning frameworks, such as Scikit Learn and preprocessing pipelines),
general search algorithms that can be used in arbitrary domains,
logging functionality to easily keep track of the results of a search,
visualization functionality to explore and inspect the logs that resulted from a search.
DeepArchitect aims to impact the workflows of both researchers and practioneers
by reducing the burden resulting from the large number of arbitrary choices that
have to be made to design a deep learning model for a problem.
As it is most often currently done, instead of writing down a single model
(or a search space or an hyperparameter space in an ad-hoc manner),
DeepArchitect uses composable and modular operators to express a search
space that is then passed to a search algorithm that samples
architectures from it with the goal of maximizing some desired performance metric.
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


We left the original search space





Before starting, we recommend that you look at a fully featured example.

*TODO*

Architecture Search Engine

We present an open-source architecture search engine.
This implementation is mo

A tour of the repo:



examples:
    We have examples for Tensorflow and Pytorch. As always, the MNIST example
    is a good place to start at.

To best get initiated with this repo, you should first ask yourself what is the
outcome that you want out of the reading these tutorials.
We identified various levels of the


Features:


* darch: Most of the framework code lies in this folder.
    * deep_architect.core: This file contains the main code regarding the implementation.
* deep_architect.contrib.misc: The contrib folder is used to keep code that is potentially useful
    for the people using the framework, but for which we will not make a coherent
    effort to maintain it.
*

* deep_architect.helpers
This folder contains code to support search over multiple domains. We show that
it is possible to search over architectures in arbitrary deep learning frameworks,
we provide a number of pytorch and tensorflow examples. A keras example, and a
scikit-learn example.
We encourage people to get creative in what domain are they going to use
the architecture search framework.
An architecture search framework is broadly done in this case.

For now, all code is to be run from

# have links to this folder. this is going to be nice.


Design principles

This project follows the inspiration of the initial DeepArchitect project.
Compared to the initial DeepArchitect code, this project is much more fully
featured.

Similarities with DeepArchitect (NOTE: not necessary.)

Features:
* Modularity


TODO: the readme should be after keras.
TODO: add a few issues to get contributions started.
TODO: do a recommended reading section where we suggest how to get up to speed
with the framework.
TODO: Recommendations for understanding. this is going to be nice to make sure
that we can do something interesting.

Install the discourse page.

both models have to be serializable for the multi-gpu information. both the
searcher token and the value list. keep it simple.

Dual purpose of creating the search spaces and writing to it.

----



todo: this file will go over the design decisions and the motivations to work
over some of these problems.

core.py
Core functionality to construct search spaces. Contains the definitions of ...
For high level usage of the library, only a very small level of understanding of
how the functionality is implemented is necessary.

Helps define what is a simple search space.

modules.py:
Modules contain mostly substitution modules, that are useful to construct new
search spaces through composition.
CamelCase functions signify that the functions return modules while lower case
functions in the typical format means that the functions return the inputs and
outputs of the module directly.
These functions are useful to implement complex search spaces through composition.
siso means that there is a single input and a single output.
mimo means that there are potentially multiple inputs and multiple outputs.

The standard followed to name hyperparameters is to prefix them with h_.

The substitution modules are implemented through lazy evaluation.

TODO: perhaps add a few examples of how this can be implemented here.

Substitution modules implement a different form of delayed evaluation for the
modules. One canonical

TODO: pointers to these documentation. this is going to be interesting.

Working directly with dictionaries of inputs and outputs is a substantial idea
that allows us to write expressions more concisely.

Relies heavily on delayed evaluation.

Sharing through passing the same hyperparameter.

The most important Python modules to understand are.

In core.py you will find the necessary functionality.


TODO: say something about if you find comments missing, looking for an example
of usage of the model is the best way to go.

NOTE: I think that the tutorials should be linked here.
the actual generated tutorials should be hold somewhere
TODO: check where to find a place that can hold tutorials.

TODO: write tutorials and examples as they make a large difference for adoption.


The main problems for architecture search are the difficulty of encoding many
computational paths.
This is solved by the introduction of the domain specific language to
write down search spaces over computational architectures.

TODO: DeepArchitect can be used for writing programs over architectures.


TODO: do the MCTS optimizer for the logo.


NOTE: most of the examples are given in Keras, but due to the abstraction
of DeepArchitect, they are very easy to transfer to new settings.


TODO: say that
why use DeepArchitect:
* Expressive: express complex search spaces.
* Visualization tools: make the most out of the spend computation by exploring
your results through interactive visualizations.

add some formattting elements to the model.

mention contribuions.

TODO: Contributors.





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


<!-- acknowledgments and contributors. -->

This project benefited from the discussions with many people: Geoff Gordon, Matt
Gormley, Willie Neiswanger, Ruslan Salakutinov, Eric Xing, Xue Liu, ...




We

<!-- requests for contributions. -->

# Getting yourself familiar with DeepArchitect

After going through this document, you should have an approximate idea of what
is DeepArchitect and what it aims to do.


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
contributing guide. There we detail some of the rationale behind the folder
organization of the project and where will future contributions live.
We recommend using GitHub issues to engage with the authors
of DeepArchitect and


