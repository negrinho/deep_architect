
###${MARKDOWN}
# <!--- high-level ideas that go into DeepArchitect -->
# DeepArchitect uses multiple ideas and concepts to achieve the
# desired level of modularity, extensibility, and abstraction.
# In this tutorial, we aim to convey to the reader ideas that make
# DeepArchitect tick and explain API design decisions.
# After going through this tutorial, the reader should have a good
# understanding of how the different components in DeepArchitect are inter-related.
#
# A reasonably simple example is a good starting point to start thinking about
# what can be represented with the search space representation language.
# We will use a runnable example in Keras, but keep in mind that DeepArchitect
# does not commit to any particular framework, deep learning or otherwise.
# The core codebase is composed mainly of wrappers and it is fairly trivial
# to extend it to other domains (e.g., scikit-learn pipelines).
# Consider the following Keras example using Keras functional API pulled
# verbatim from https://keras.io/getting-started/functional-api-guide/
# TODO: point to other examples that use different frameworks.
# TODO: fix the links.

from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model

inputs = Input(shape=(784,))
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=predictions)
model.summary()
plot_model(model, show_shapes=True)
# NOTE: I think that these can be put in the write model.
# I can write these in assets.

# The above code is an example of a fixed two-layer perceptron defined in Keras.
# The problem with the above code is that it requires the expert to commit to
# a single model.
# There are many opportunities in the code above to be less specific about the
# model.
# For example, we commit to a neural network with two layers, each with 64 hidden
# units and ReLU activations.
# A natural first step is to be less specific about these hyperparameters by
# searching over them.
# For example, by searching over the number of layers and activations of
# each layer.
# We defined a few simple helper functions that, for simple cases, allow us
# to take a function that returns a Keras layer and wraps it in a DeepArchitect
# module.
# A minimal adaptation of the above example in DeepArchitect would look like

import deep_architect.core as co
import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
import deep_architect.helpers.keras as hke
from deep_architect.searchers.common import random_specify

D = hp.Discrete

def dense(h_units, h_activation):
    return hke.siso_keras_module_from_keras_layer_fn(Dense, {
        "units" : h_units, "activation" : h_activation})

def get_search_space():
    co.Scope.reset_default_scope()
    return mo.siso_sequential([
        dense(D([32, 64, 128, 256]), D(["relu", "sigmoid"])),
        dense(D([32, 64, 128, 256]), D(["relu", "sigmoid"])),
        dense(D([10]), D(["softmax"]))
    ])

(inputs, outputs) = get_search_space()

in_model = Input(shape=(784,))
co.forward({inputs["In"] : in_model})
predictions = outputs["Out"].val
model = Model(inputs=in_model, outputs=predictions)
model.summary()
plot_model(model, show_shapes=True)

# The above code defines a search space where the nonlinearities and number of
# hidden units are allowed to vary pre layer.
# In this case, the hyperparameters are independent for each of the modules.
# What we have done is simply defining a search space that captures all possible
# choices for the values of these hyperparameters .
# In the DeepArchitect codebase we have defined some auxiliary tools to
# visualize the search search as a graph.

# TODO: say something about hyperparameter creation.

# NOTE: write about how this is used

import deep_architect.visualization as vi

vi.draw_graph(outputs.values(), draw_module_hyperparameter_info=False)

# Something that can be grasped from the graph is that the connection among
# the modules is fixed.

# In the construction of the search space, all function calls return a dictionary
# of inputs and outputs.
# To sample a model from this search space, we can use some of the common
# functionality defined in searchers.
# Typically, we just use a searcher, but in this case we are just going
# to use a simple function from the search tools that randomly chooses
# values for all hyperparameters of the search space until a single model
# is obtained.

# The rectangles in the graph represent hyperparameters, and the ovals
# represent hyperparameters.
# The edges between the rectangles represent outputs of a module going into
# inputs of other modules.
# The edges between the ovals and the rectangles represent the dependency
# of the module on the value of that hyperparameter.

import deep_architect.searchers.common as seco

vs = seco.random_specify(outputs.values())
print(vs)

# The values randomly chosen are returned by `random_specify`.
# This function simply iterates through the hyperparameters that have not
# been assigned a value yet and chooses a value randomly among the possible ones.

# After choosing all these values, the resulting search space looks like this.

vi.draw_graph(outputs.values(), draw_module_hyperparameter_info=False)

# We see that the edges between hyperapameters and modules have been labeled
# with the values that have been chosen for the hyperparameters.
# The search process iterates over the hyperparameter that have not
# been assigned a value yet and picks a value at random among the possible
# values that can be assigned to that hyperparameter.
# The graph transitions with each assignment.
# We have a function that allows to visualize these graph transitions as a
# sequence of frames.

inputs, outputs = get_search_space()
vi.draw_graph_evolution(outputs.values(), vs, '.', draw_module_hyperparameter_info=False)

# TODO: I think that it is perhaps a good idea to talk about hyperparameters
# here.

# We see that we start with the initial graph

# This graph is still very simple.
# The functionality to visualize the transitions between graphs will be more
# insightful once we start using more complex graph operators.
# The hyperparameter were chosen independently for each of the layers.
# If we wanted to tie the hyperparameter across the different parts of the
# search space, e.g., tie the value of the value of the nonlinearity,
# we simply have to instantiate a single hyperparamter object and use it in
# multiple places.
# Adapting the first search space to reflect this change is straightforward.

def get_search_space_hp_shared():
    co.Scope.reset_default_scope()
    h_activation = D(["relu", "sigmoid"])
    return mo.siso_sequential([
        dense(D([32, 64, 128, 256]), h_activation),
        dense(D([32, 64, 128, 256]), h_activation),
        dense(D([10]), D(["softmax"]))
    ])
(inputs, outputs) = get_search_space_hp_shared()

vi.draw_graph(outputs.values(), draw_module_hyperparameter_info=False)

# Redrawing the initial graph for the search space (i.e., after having
# made any choices for hyperparameters), we see that that now there exists
# a single hyperparameter associated to the local hyperparameter value for the
# activation of each of the dense modules.

# Another useful language feature that we have implemented in our framework
# is the notion of dependent hyperparameters.
# Sometimes, we wish to have an hyperparameter that takes a value that is a
# function of the values of some other hyperparameters.
# For example, we will adapt our running example for writing a search space
# where the value of the number of hidden units for the second layer of the
# network is twice as many as the number of hidden units for the first dense
# layer.

def get_search_space_hp_dependent():
    co.Scope.reset_default_scope()
    h_activation = D(["relu", "sigmoid"])
    h_units = D([32, 64, 128, 256])
    h_units_dep = co.DependentHyperparameter(
        lambda units: 2 * units, {"units" : h_units})

    return mo.siso_sequential([
        dense(h_units, h_activation),
        dense(h_units_dep, h_activation),
        dense(D([10]), D(["softmax"]))
    ])
(inputs, outputs) = get_search_space()

vi.draw_graph(outputs.values(), draw_module_hyperparameter_info=False)

# TODO: draw the hyperparameter evolution here.

# As we can see in the graph, there is an edge going from the independent
# hyperparameter to the hyperparameter that it depends on.
# This edge represents the dependency of one of these hyperparameters on the
# other one.
# Dependent hyperparameters can depend on other dependent hyperparameters,
# as long as no directed cycles are formed.
# One may question why introduce dependent hyperparameters in such a language.
# While independent hyperparameters can be used to express a superset of
# what can be done with dependent hyperparameters, it is useful to have the
# possibility of expressing depedent hyperaparameters to restrict the search
# space to transformations that are of interest rather than only being able
# to consider search spaces that have undue flexibility.
# NOTE: I'm not sure if I want to include this in the model

# So far we talked about modules and hyperparameters.
# For hyperparameters, we distinguish between independent hyperparameters
# (hyperparameters whose value is set independently of any other hyperparameters),
# and dependent hyperparameters (hyperparameters whose value is computed
# as a function of the values of some other hyperparameters).
# For modules, we distinguish between basic modules and substitution modules.
# So far, we have only concerned ourselves with basic modules (i.e., the dense
# module that we used in the example search spaces above).
# Basic modules are used to represent computations eventual computations, i.e.,
# after values for all the hyperparameters of the module and values for the
# inputs are available, the module implements some specific computation.
# In contrast, we can have modules whose sole purpose is to serve as a placeholder
# until some property is determined.
# The purpose of these modules is not to implement a specific computation but
# to delay the choice of a specific property (i.e., the choice of values for
# specific hyperperameter that capture this structural transformation).
# The fundamental concepts to express these transformations is the notion of
# a substitution module.
# Substitution modules rely heavily on the ideas of delayed evaluation.
# We have implemented many structural transformations as subsitution modules in
# DeepArchitect.
# A very important property of substitution modules is that they are
# completely indendent of the underlying framework used for the modules (i.e.,
# they work without requiring any adaptation for Keras, Tensorflow, or Scikit-Learn).
# Let us consider an example for a substitution module that implements
# the optional operation which either includes a submodule or not.

def get_search_space():
    co.Scope.reset_default_scope()
    h_activation = D(["relu", "sigmoid"])
    h_units = D([32, 64, 128, 256])
    h_units_dep = co.DependentHyperparameter(
        lambda units: 2 * units, {"units" : h_units})
    h_opt = D([0, 1])

    return mo.siso_sequential([
        dense(h_units, h_activation),
        mo.siso_optional(lambda: dense(h_units_dep, h_activation), h_opt),
        dense(D([10]), D(["softmax"]))
    ])
(inputs, outputs) = get_search_space()

# A few things to keep in mind about the example above.
# The

# TODO: show that it should be interesting. how does the other hyperparameter
# appear.

# Another simple subsitution module is the one that repeats

# Substitution modules can be nested, for example,
def get_search_space():
    co.Scope.reset_default_scope()
    h_activation = D(["relu", "sigmoid"])
    h_units = D([32, 64, 128, 256])
    h_units_dep = co.DependentHyperparameter(
        lambda units: 2 * units, {"units" : h_units})
    h_opt = D([0, 1])

    return mo.siso_sequential([
        dense(h_units, h_activation),
        mo.siso_optional(lambda: dense(h_units_dep, h_activation), h_opt),
        dense(D([10]), D(["softmax"]))
    ])
(inputs, outputs) = get_search_space()

# All the language features so far have been focused on simple search spaces
# where the module connections are fixed upfront.
# One could argue that these search spaces are easily expressible th



# In this case, to get the Keras model, we do

# NOTE: the model operates over these predictions.
# this s a good way of going about it.

# explain the transition mechanism and what not.





# We will use DeepArchitect to write search spaces over architectures effortlessly.
# The first concept that we introduce is that of a module.
# There is a very direct correspondence between each layers in a neural network
# and a module.
#
# Modules are the main building blocks for writing search spaces over
# computational graphs.
# Consider the signature of the class definition in deep_architect/core.py
# A module is composed of inputs, outputs, and hyperparameters.
# Given values for the inputs and values for the hyperparameters the module
# implements some computation that can be a function of both values.







# <!--- NOTE: it might be a better idea to talk about a generic example. -->
# In essence, DeepArchitect is a framework to search over arbitrary computational
# graphs.
# The graphs can be in a lot of different domains as we will see.
# TODO: link to a different tutorial on applying to different domains.

# # Modules

#
# For pedagogical purposes, we will dive deep into the internals of DeepArchitect.
# From the point of view of the user, much of these internals will be hidden,
# e.g., because the user is concerned mainly with a specific framework for
# which many of these components are already implemented.

# As a toy illustrative example, consider that we define two types of
# modules in an arbitrary framework.
# The goal here is simply to show how can we define search spaces over
# these graphs.
# We will have many chances to look at deep learning examples.



# Modules can be connected to other modules to form networks.
# The computation done by a module is entirely up to the user,
# leading to the modularity and extensibility of the framework.
# New modules can be implemented easily.


# NOTE: perhaps start with a simple example. I think that it is going to be
# better to show some of the functionality.

import deep_architect.core as co

def get_module(name, input_names, output_names, name_to_hyperp):
    m = co.Module(name=name)
    m._register(input_names, output_names, name_to_hyperp)
    return m.get_io()

def a(h1, h2):
    return get_module('A', ["In"], ["Out"], {"h1" : h1, "h2" : h2})

def b(h):
    return get_module('B', ["In"], ["Out0", "Out1"], {"h" : h})

def c(h):
    return get_module('C', ["In0", "In1"], ["Out"], {"h" : h})

# A few observations about the above code,

# We simply defined a few modules.
# This is sufficient to write a few computational graphs, but they do not have
# any defined computational behavior.
# Nonetheless, it is possible use them to write them search spaces.
# For example, a very basic search space is

import deep_architect.hyperparameters as hp

D = hp.Discrete

x = a(D([0, 1]), D([16, 32, 64, 128]))

# NOTE: cover more information about dealing with these models.



# Inputs and outputs.
# hyperparameters
# use of the main scope.
#


# Using the model, it is possible to write a few search spaces.



# NOTE: I should talk about building blocks somewhere in the model.



# NOTE: how do I put into context the notion of compilation and forward with
# the different models.


# contrary to other examples, this is going



# Consider an arbitrary domain which has two types of modules A, and B.
# Modules of type A take two hyperparameters




# Hyperparameter

# We already hinted at how to use hyperparameters, but in this section
# we will go into more detail.
#


# Substitution Modules

# NOTE: substitution modules are domain independent.
# all


# Dependent Hyperparameters

#

# Weight sharing


#



# <!--- -->


# Summary

# Recap all the important points of working with the framework
# contextualualize on the model






# TODO: move this somewhere else.
# When thinking about DeepArchitect, you should always have the three main
# components of the framework in mind: search space, searcher, and evaluator.
# Thinking about where each of the different aspects fits will help the reader
# put the information in context of the framework as a whole.
# To get acquainted with the main building blocks of the framework we recommend
# the reader to peruse the code in deep_architect/core.py.
# TODO: add a link to it.


# TODO: make a recommendation about the use of auxiliary functions to put
# together some of these search spaces.

# TODO: say that siso stands for single input single output.

# NOTE: after going through this tutorial, we seriously recommend the reader
# to peruse

# TODO: set the values of the operators.

# TODO: what does it need to be done for

# TODO:
# most important parts of the model to read are

### NOTE: this is nice but it is mostly about the visualization functionality.

# NOTE: For large complex graphs where the number of hyperparameters can
# lead to too much clutter, we can use the visualization functionality that

# TODO: I think that I should mention everything in the beginning that there
# are these different ways of working with these models.

# TODO: change the model to make it nicer. I really can't define functions that
# lead to conflicts.

# NOTE: something that I like in writing down search spaces is that I can provide
# a textual description of what the search space is going to accomplish and then
# go ahead and actually show it in code. the nice thing is that the code is a lot
# shorter than the actual definition, which I think should help.

# NOTE: I will need to compile the model into a keras model just to show
# that is working as appropriately.

# TODO: I will have to explain the notion of

# SearchSpaceFactory; just takes a function that returns inputs and output.
# I think that it is fairly decent.

# NOTE: explanation of what is the scope.

# NOTE: discuss some of the common questions that come when comparing to
# existing hyperparameter optimization algorithms.

# the scope is used for naming stuff.
# TODO: talk about an important concept in the definition of the language.

# TODO: recommend the reader to look at the helper to get a better
# sense of what we are doing here.
# TODO: check for models vs modules.
# TODO: check for subsitution. is it correct or not.
# TODO: perhaps cover the hyperparameter sharer because this will allow it
# to show the model
# NOTE: maybe put all imports near the top.
# NOTE: overall, I we believe that the language has multiple opportunities for
# composition.
# TODO: write something about understanding substitution modules.
# there is the code but there is also the behavior that this implements.
# I think that I can show both.

