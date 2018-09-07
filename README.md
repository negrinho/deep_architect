# darch






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



