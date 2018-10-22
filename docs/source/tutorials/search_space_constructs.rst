
Search space constructs
-----------------------

DeepArchitect uses multiple ideas and concepts to achieve the
desired level of modularity, extensibility, and abstraction.
In this tutorial, we aim to convey to the reader ideas that make
DeepArchitect tick and explain API design decisions.
After going through this tutorial, the reader should have a good
understanding of how the different components in DeepArchitect are inter-related.

A simple example is a good starting point to start thinking about
what can be represented with the search space representation language.
We will use a runnable example in Keras, but keep in mind that DeepArchitect
does not commit to any particular framework, deep learning or otherwise.
The core codebase is composed mainly of wrappers and it is straightforward
to extend it to other domains (e.g., scikit-learn pipelines).
Consider the following Keras example using the Keras functional API pulled
verbatim from `here <https://keras.io/getting-started/functional-api-guide/>`_.

.. code:: python

    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.utils import plot_model

    inputs = Input(shape=(784,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)


The above code is an example of a fixed two-layer perceptron defined in Keras.
The problem with the above code is that it requires the expert to commit to
specific values for the number of units and the type of activations.
A natural first step is to be less specific about these hyperparameters by
searching over them, e.g., by searching over the number of layers and
activations of each layer.
We defined a few simple helper functions that, for simple cases, allow us
to take a function that returns a Keras layer and wraps it in a DeepArchitect module.
See below for a minimal adaptation of the above example in DeepArchitect.

.. code:: python

    import deep_architect.core as co
    import deep_architect.modules as mo
    import deep_architect.hyperparameters as hp
    import deep_architect.helpers.keras as hke
    from deep_architect.searchers.common import random_specify, specify

    D = hp.Discrete


    def dense(h_units, h_activation):
        return hke.siso_keras_module_from_keras_layer_fn(Dense, {
            "units": h_units,
            "activation": h_activation
        })


    def search_space():
        return mo.siso_sequential([
            dense(D([32, 64, 128, 256]), D(["relu", "sigmoid"])),
            dense(D([32, 64, 128, 256]), D(["relu", "sigmoid"])),
            dense(D([10]), D(["softmax"]))
        ])


    (inputs, outputs) = mo.SearchSpaceFactory(search_space).get_search_space()


The above code defines a search space where the nonlinearities and number of
units are chosen from a set of possible values rather than being fixed upfront.
In this case, the hyperparameters are independent for each of the modules.
What we have done is simply defining a search space that captures all possible
choices for the values of these hyperparameters .
In DeepArchitect, we have implemented some auxiliary tools to
visualize the search search as a graph.

.. code:: python

    import deep_architect.visualization as vi
    vi.draw_graph(outputs.values(), draw_module_hyperparameter_info=False)


The connections between the modules in the graph are fixed.
In the construction of the search space, all function calls return a dictionary
of inputs and outputs.
Typically, we just use a searcher, but in this case we are just going
to use a simple function from the search tools that randomly chooses
values for all hyperparameters of the search space until values for all the

The rectangles in the graph represent modules, and the ovals
represent hyperparameters.
The edges between the rectangles represent outputs of a modules going into
inputs of other modules.
The edges between the ovals and the rectangles represent the dependency
of the module on the value of that hyperparameter.

.. code:: python

    import deep_architect.searchers.common as seco
    vs = seco.random_specify(outputs.values())
    x = Input(shape=(784,))
    co.forward({inputs["In"]: x})
    y = outputs["Out"].val
    print(vs)


The values randomly chosen are returned by `random_specify`.
This function simply iterates through the hyperparameters that have not
been assigned a value yet and chooses a value randomly among the possible ones.
After choosing all these values, the resulting search space looks like this.

.. code:: python

    vi.draw_graph(outputs.values(), draw_module_hyperparameter_info=False)


We see that the edges between hyperapameters and modules have been labeled
with the values that have been chosen for the hyperparameters.
The search process iterates over the hyperparameter that have not
been assigned a value yet and picks a value at random among the possible
values that can be assigned to that hyperparameter.
The graph transitions with each assignment.
We have a function that allows us to visualize these graph transitions as a
sequence of frames.

.. code:: python

    inputs, outputs = search_space()

    # vi.draw_graph_evolution(
    #     outputs.values(), vs, '.', draw_module_hyperparameter_info=False)


We see that we start with the initial graph with no hyperparameters specified
(i.e., no hyperparameters have been assigned a value), and progressively,
one by one, each hyperparameter is assigned a value.
We ask the reader to pay attention to how the edges connecting hyperparameters
to modules change with each transition.

This graph defining a search space is still very simple.
The functionality to visualize the transitions between graphs will become more
insightful once we start using more complex search space operators.
The hyperparameter values were chosen independently for each of the layers.
If we wished to tie some hyperparameters across different parts of the
search space, e.g., use the same nonlinearity for all modules,
we would simply have to instantiate a single hyperparamter and use it in
multiple places.
Adapting the first search space to reflect this change is straightforward.


.. code:: python

    def search_space():
        co.Scope.reset_default_scope()
        h_activation = D(["relu", "sigmoid"])
        return mo.siso_sequential([
            dense(D([32, 64, 128, 256]), h_activation),
            dense(D([32, 64, 128, 256]), h_activation),
            dense(D([10]), D(["softmax"]))
        ])


(inputs, outputs) = search_space()
vi.draw_graph(outputs.values(), draw_module_hyperparameter_info=False)


Redrawing the initial graph for the search space (i.e., after having
made any choices for hyperparameters), we see that that now there exists
a single hyperparameter associated to activations of all dense modules.

We have implemented another useful language features for hyperparameters,
namely dependent hyperparameters, which allows us to express an hyperparameter
whose value is a function of the value of other hyperparameters.
We will adapt our running example for writing a search space
where the value of the number of hidden units for the second layer of the
network is twice as many as the number of hidden units for the first dense
layer.


.. code:: python

    def search_space():
        co.Scope.reset_default_scope()
        h_activation = D(["relu", "sigmoid"])
        h_units = D([32, 64, 128, 256])
        h_units_dep = co.DependentHyperparameter(lambda units: 2 * units,
                                                {"units": h_units})

        return mo.siso_sequential([
            dense(h_units, h_activation),
            dense(h_units_dep, h_activation),
            dense(D([10]), D(["softmax"]))
        ])


    (inputs, outputs) = search_space()
    vi.draw_graph(outputs.values(), draw_module_hyperparameter_info=False)


As we can see in the graph, there is an edge going from the independent
hyperparameter to the hyperparameter that it depends on.
This edge represents the dependency of one of the hyperparameters on the other one.
Dependent hyperparameters can depend on other dependent hyperparameters,
as long as no directed cycles are formed.
One may question why introduce dependent hyperparameters in such a language.
While independent hyperparameters can be used to express a superset of
what can be done with dependent hyperparameters, it is useful to have the
possibility of writing depedent hyperaparameters to restrict the search
space to transformations that are of interest rather than only being able
to consider search spaces that have excessive flexibility.

It may be informative to observe how does the graph transition with
successive assignments to the values of hyperparameters.

.. code:: python

    vs = seco.random_specify(outputs.values())
    inputs, outputs = search_space()

    # vi.draw_graph_evolution(
    #     outputs.values(), vs, '.', draw_module_hyperparameter_info=False)


By looking at the graph, we see that as soon as a value is a assigned
to the hyperparameter that the dependent hyperparameter depends on, the
the dependent hyperparameter is assigned a value.
The value assignment to the dependent hyperparameter is triggered due to the
fact that all the hyperparameters that the depedent hyperparameter depends
on have been assigned a value.

We have talked about modules and hyperparameters.
For hyperparameters, we distinguish between independent hyperparameters
(hyperparameters whose value is set independently of any other hyperparameters),
and dependent hyperparameters (hyperparameters whose value is computed
as a function of the values of some other hyperparameters).
For modules, we distinguish between basic modules
(modules that stay in place when all hyperparameters that the module depends
on have been assigned values),
and substitution modules
(modules that disappear, giving rise to a new graph fragment in its place
with other modules, when all
hyperparameters that the module depends on have been assigned values).

So far, we have only concerned ourselves with basic modules (e.g., the dense
module that we used in the example search spaces above).
Basic modules are used to represent eventual computations, i.e.,
after values for all the hyperparameters of the module and values for the
inputs are available, the module implements some well-defined computation.
In contrast, we can have modules whose purpose is to serve as a placeholder
until some property is determined.
The purpose of these modules is not to implement computation but
to delay the choice of a specific property (i.e., the choice of values for
specific hyperperameter that capture this structural transformation).
The fundamental concept to express these transformations is the notion of
a substitution module.
Substitution modules rely heavily on the ideas of delayed evaluation from
programming languages.

We have implemented many structural transformations as substitution modules in
DeepArchitect.
A very important property of substitution modules is that they are
completely independent of the underlying framework used for the basic modules (i.e.,
they work without requiring any adaptation for Keras, Tensorflow, Scikit-Learn,
or any other framework).
Let us consider an example search space using a substitution module that implements
an operation that either includes a submodule or not.


.. code:: python

    def search_space():
        co.Scope.reset_default_scope()
        h_activation = D(["relu", "sigmoid"])
        h_units = D([32, 64, 128, 256])
        h_units_dep = co.DependentHyperparameter(lambda units: 2 * units,
                                                {"units": h_units})
        h_opt = D([0, 1])

        return mo.siso_sequential([
            dense(h_units, h_activation),
            mo.siso_optional(lambda: dense(h_units_dep, h_activation), h_opt),
            dense(D([10]), D(["softmax"]))
        ])


    (inputs, outputs) = search_space()


The optional module takes a thunk (this terminology comes from programming
languages) which returns a graph fragment (returned as a dictionary of
input names to inputs and a dictionary of output names to outputs)
which is called if the hyperparameter that determines if the thunk is
to be called or not, takes the value "1" (i.e., the thunk is to be called,
and the resulting graph fragment is to be included in the place of the
substitution module).
The visualization functionality will be more insightful in this case.
Consider the graph evolution for a random sample from this search space.

.. code:: python

    vs = seco.random_specify(outputs.values())
    inputs, outputs = search_space()

    # vi.draw_graph_evolution(
    #     outputs.values(), vs, '.', draw_module_hyperparameter_info=False)


We see that once the hyperparameter that the optional substitution module depends on
is assigned a value, the substitution module disappears and is replaced by a graph
fragment that depends on the value that was assigned to that hyperparameter, i.e.,
if we decide to include it, the thunk is called returning a graph fragment;
if we decide to not include it, an identity module (passes the input to the output without changes)
is substituted in its place.

Another simple substitution module is the one that repeats the graph fragment
in a serial connection multiple times.
In this case, the substitution hyperparameter refers to how many times will
the thunk returning a graph fragment will be called; all repetitions are
connected in a serial connection.


.. code:: python

    def search_space():
        co.Scope.reset_default_scope()
        h_activation = D(["relu", "sigmoid"])
        h_units = D([32, 64, 128, 256])
        h_units_dep = co.DependentHyperparameter(lambda units: 2 * units,
                                                {"units": h_units})
        h_opt = D([0, 1])
        h_num_repeats = D([1, 2, 4])

        return mo.siso_sequential([
            mo.siso_repeat(lambda: dense(h_units, h_activation), h_num_repeats),
            mo.siso_optional(lambda: dense(h_units_dep, h_activation), h_opt),
            dense(D([10]), D(["softmax"]))
        ])


    (inputs, outputs) = search_space()


Note that in the search space above, the hyperparameter respective to the
number of units of the dense modules inside the repeat share the same hyperparameter,
meaning that all these modules will have the same number of units.

.. code:: python

    vs = seco.random_specify(outputs.values())
    inputs, outputs = search_space()

    # vi.draw_graph_evolution(
    #     outputs.values(), vs, '.', draw_module_hyperparameter_info=False)

In the graph evolution, we see that once we assign a value to the hyperparameter
corresponding to the number of repetitions of the graph fragment returned by the
thunk, a graph fragment corresponding to the serial connections of that many
repetitions is substituted in its place.
These example search spaces together with the visualizations of the graph
evolutions as we assign values to hyperparameters should give the
reader a sense about what types of options are expressible in
DeepArchitect with basic and substitution modules, and independent and
dependent hyperparameters.
It should also hint to the reader how the language to represent search spaces
is implemented.

Substitution modules can be used in any place a module is required, meaning that
they can nested without any issues.
For example, consider the following example


.. code:: python

    def search_space():
        co.Scope.reset_default_scope()
        h_activation = D(["relu", "sigmoid"])
        h_units = D([32, 64, 128, 256])
        h_units_dep = co.DependentHyperparameter(lambda units: 2 * units,
                                                {"units": h_units})
        h_opt = D([0, 1])
        h_num_repeats = D([1, 2, 4])

        return mo.siso_sequential([
            mo.siso_repeat(lambda: dense(h_units, h_activation), h_num_repeats),
            mo.siso_optional(
                lambda: mo.siso_repeat(lambda: dense(h_units_dep, h_activation), h_num_repeats),
                h_opt),
            dense(D([10]), D(["softmax"]))
        ])


    (inputs, outputs) = search_space()


Again, given the search space above, the reader should get an expectation of
of what graph evolution to expect.
Take one minute to ponder on what kind of transitions to expect and then run
the code below to generate the visualization for the graph evolution and see if
it matches your expectations.

.. code:: python

    vs = seco.random_specify(outputs.values())
    inputs, outputs = search_space()
    # vi.draw_graph_evolution(
    #     outputs.values(), vs, '.', draw_module_hyperparameter_info=False)


We argue that by using basic modules, substitution modules, independent hyperparameters,
and dependent hyperparameters we are able to represent a large variety of
search spaces in a compact and natural manner.
As the reader becomes more confortable with these concepts, the reader should
find it progressively easier to express search spaces in DeepArchitect and
better appreciate the expressivity and reusability of the language.

We now provide some ending notes for this tutorial, both talking about
minor aspects that we have not paid much attention in this tutorial, and
giving recommendations to the reader on how and what to learn next.
Throughout the definition of the various search spaces, we have seen
this line `co.Scope.reset_default_scope()`.
We use an object that we call the scope to assign unique names to the elements
that show up in the search space (currently, modules, hyperparameters, inputs, and
outputs).
Every time a module, hyperparameter, input, or output is created, we use
the scope to assign a unique name to it.
Every time that we want to start the search from scratch with a new search space,
we should clear the scope to avoid keeping the names and objects from the previous
samples around.
In most cases, the user does not have to be concerned about the scope as it
can just use the default scope.
We also recommend the reader to look into search space factory as it provides
a convenient auxiliary function that directly takes care of these issues.

Besides basic modules and substitution modules, we also use several auxiliary
functions whose purpose is to put arrange multiple graph fragments in different
ways.
They often do not create new modules, but simply use graph fragments or
functions that return graph fragments to create a new graph fragment by using the
arguments in a certain way.
An example of a function of this type is `siso_sequential`, which just connects
the graph fragments (expressed as a dictionary of inputs and a dictionary of outputs),
in a serial connection, which just require us to connect inputs and outputs of the
fragments passed as arguments to the function.
Similarly to substitution modules, these auxiliary functions are framework
independent as they only rely on properties of the module API.
A reasonable way of thinking about these auxiliary functions is that they
are just like substitution modules, but the substitution is done immediately
rather than being postponed to some later stage when some hyperparameters have
been specified.
Using and defining auxiliary functions of this type will help the user have
a more effective and pleasant experience with the framework.
Auxiliary functions of this type are very useful in practice as we can use
them to construct larger search spaces by making complex arrangements from
smaller search spaces.

When implementing support for a new framework, the only concepts that need to
potentially be specialized to the new framework are the basic modules.
We recommend the reader to read `deep_architect.core.py` for extensive information
about the APIs.
This code is the basis of DeepArchitect and has been extensively commented,
meaning that the reader should have a much better understanding on how to
extend the framework after perusing this code and perhaps, experimenting with it.
Everything in `deep_architect.core.py` is framework independent.
To understand more about substitution modules and how they are implemented, we
point the reader to `deep_architect.modules.py`, which is also extensively
commented.
We point the reader to the tutorial about supporting new frameworks for an
explanation of the aspects that come into play when specializing to a
new framework.

For learning more about the framework, please read more tutorials on aspects or
use cases which you may find important and/or hard to understand.
In this tutorial, we only covered expressing search spaces over architectures.
DeepArchitect is composed of many other components such as search, evaluation, logging, visualization
and multiworker, so please read additional tutorials if you wish
to become familiar with these other aspects.

While we have not covered rerouting in this tutorial, it is reasonably
straightforward to think about how to implement rerouting with, either as a
substitution module or simply a basic module.
For example, for a rerouting operation that takes `k` inputs and `k` outputs, and
does a permutation of the inputs and outputs based on the value of an
hyperparameter, if we implement this operation using a basic module,
the basic module simply has to implement the chosen permutation when forward is
called.
If a substitution module is used instead, the module disappears once the value
for the hyperparameter is chosen and the result of rerouting shows up in its
place.
After the user becomes proeficient with the ideas of basic and substitution
modules, the user will realize that oftentimes there are multiple ways of
expressing the same search space.
Our suggestion is that basic modules, substitution modules, independent hyperaparameters
and dependent hyperparameters should be used for maximum effect to express
search spaces very compactly and clearly.
