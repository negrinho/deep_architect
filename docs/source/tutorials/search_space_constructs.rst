
Search space constructs
-----------------------

This tutorial should give the reader a good understanding of how to write search spaces in DeepArchitect. All the visualizations generated in this tutorial can be found `here <https://www.cs.cmu.edu/~negrinho/deep_architect/search_space_constructs/viz/>`__.

Starting with a fixed Keras model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A simple example in Keras is a good starting point for understanding the search space representation language. DeepArchitect can be used with arbitrary frameworks (deep learning or otherwise) as the core codebase is composed mainly of wrappers. Consider the following example using Keras functional API pulled verbatim from `here <https://keras.io/getting-started/functional-api-guide/>`_.

.. code:: python

    from keras.layers import Input, Dense
    from keras.models import Model
    from keras.utils import plot_model

    inputs = Input(shape=(784,))
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    predictions = Dense(10, activation='softmax')(x)


Introducing independent hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code above defines a fixed two-layer network in Keras. Unfortunately, this code commits to specific values for the number of units and the activation type. A natural first step is to be less specific about these hyperparameters by searching over the number of layers and activations of each layer. We defined a few simple helper functions that allow us to take a function that returns a Keras layer and wraps it in a DeepArchitect module.

See below for a minimal adaptation of the above example in DeepArchitect.

.. code:: python

    import deep_architect.core as co
    import deep_architect.modules as mo
    import deep_architect.hyperparameters as hp
    import deep_architect.helpers.keras_support as hke
    from deep_architect.searchers.common import random_specify, specify

    D = hp.Discrete
    wrap_search_space_fn = lambda fn: mo.SearchSpaceFactory(fn).get_search_space


    def dense(h_units, h_activation):
        return hke.siso_keras_module_from_keras_layer_fn(Dense, {
            "units": h_units,
            "activation": h_activation
        })


    def search_space0():
        return mo.siso_sequential([
            dense(D([32, 64, 128, 256]), D(["relu", "sigmoid"])),
            dense(D([32, 64, 128, 256]), D(["relu", "sigmoid"])),
            dense(D([10]), D(["softmax"]))
        ])


    (inputs, outputs) = wrap_search_space_fn(search_space0)()

This code defines a search space where the nonlinearities and number of units are chosen from a set of values rather than being fixed upfront. In this case, the hyperparameters are independent for each of the modules. The search space captures possible values for these hyperparameters.

In DeepArchitect, we have implemented some auxiliary tools to visualize the search search as a graph.

.. code:: python

    import deep_architect.visualization as vi
    vi.draw_graph(
        outputs,
        draw_module_hyperparameter_info=False,
        graph_name='graph0_first')

The connections between modules are fixed. In the construction of the search space, all function calls return dictionaries of inputs and outputs. Typically, we use a searcher, but in this example we will use a simple function from the search tools that randomly chooses values for all unsassigned independent hyperparameters in the search space.

In the graph, modules are represented by rectangles and hyperparameters are represented by ovals. An edge between two rectangles represents the output of a module going into an input of the other modules. An edge between a rectangle and a oval represents a dependency of a module on a hyperparameter. Check `here <https://www.cs.cmu.edu/~negrinho/deep_architect/search_space_constructs/viz/>`__ for all search space visualizations generated in this tutorial.

.. code:: python

    import deep_architect.searchers.common as seco
    vs = seco.random_specify(outputs)
    x = Input(shape=(784,))
    co.forward({inputs["in"]: x})
    y = outputs["out"].val
    print(vs)

:py:func:`deep_architect.searchers.common.random_specify` iterates over independent hyperparameters that have not yet been assigned a value and chooses a value uniformly at random from the set of possible values. After all hyperparameters have been assigned values, we have the following search space:

.. code:: python

    vi.draw_graph(
        outputs,
        draw_module_hyperparameter_info=False,
        graph_name='graph0_last')

Edges between hyperparameters and modules have been labeled with the values chosen for the hyperparameters. The graph transitions with each value assignment to an independent hyperparameter. We can visualize these graph transitions as a frame sequence:

.. code:: python

    (inputs, outputs) = wrap_search_space_fn(search_space0)()

    vi.draw_graph_evolution(
        outputs,
        vs,
        '.',
        draw_module_hyperparameter_info=False,
        graph_name='graph0_evo')

We ask the reader to pay attention to how the edges connecting hyperparameters to modules change with each transition. This search space is very simple. This functionality is more insightful for more complex search spaces.

Sharing hyperparameters across modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the previous search space, the hyperparameter values were chosen independently for each of the layers. If we wished to tie hyperparameters across different parts of the search space, e.g., use the same nonlinearity for all modules, we would have to instantiate a single hyperparameter and use it in multiple places. Adapting the first search space to reflect this change is straightforward.


.. code:: python

    def search_space1():
        h_activation = D(["relu", "sigmoid"])
        return mo.siso_sequential([
            dense(D([32, 64, 128, 256]), h_activation),
            dense(D([32, 64, 128, 256]), h_activation),
            dense(D([10]), D(["softmax"]))
        ])


    (inputs, outputs) = wrap_search_space_fn(search_space1)()
    vi.draw_graph(
        outputs,
        draw_module_hyperparameter_info=False,
        graph_name='graph1_first')

Redrawing the initial graph for the search space, we see that that now there is a single hyperparameter associated to activations of all dense modules.

Expressing dependencies between hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A dependent hyperparameters has its value assigned as a function of the values of other hyperparameters. We will adapt our running example by making the number of hidden units of the second layer of the network twice as large as the number of hidden units of the first layer. This allows us to naturally encode a more restricted search space.


.. code:: python

    def search_space2():
        h_activation = D(["relu", "sigmoid"])
        h_units = D([32, 64, 128, 256])
        h_units_dep = co.DependentHyperparameter(lambda dh: 2 * dh["units"],
                                                 {"units": h_units})

        return mo.siso_sequential([
            dense(h_units, h_activation),
            dense(h_units_dep, h_activation),
            dense(D([10]), D(["softmax"]))
        ])


    (inputs, outputs) = wrap_search_space_fn(search_space2)()
    vi.draw_graph(
        outputs,
        draw_module_hyperparameter_info=False,
        graph_name='graph2_first')

As we can see in the graph, there is an edge going from the independent hyperparameter to the hyperparameter that it depends on. Dependent hyperparameters can depend on other dependent hyperparameters, as long as there are no directed cycles.

See below for the graph transition with successive value assignments to hyperparameters.

.. code:: python

    vs = seco.random_specify(outputs)
    (inputs, outputs) = wrap_search_space_fn(search_space2)()

    vi.draw_graph_evolution(
        outputs,
        vs,
        '.',
        draw_module_hyperparameter_info=False,
        graph_name='graph2_evo')

A dependent hyperparameter is assigned a value as soon as the hyperparameters that it depends on have been assigned values.

Delaying sub-search space creation through substitution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have talked about modules and hyperparameters. For hyperparameters, we distinguish between independent hyperparameters (hyperparameters whose value is set independently of any other hyperparameters), and dependent hyperparameters (hyperparameters whose value is computed as a function of the values of other hyperparameters). For modules, we distinguish between basic modules (modules that stay in place when all hyperparameters that the module depends on have been assigned values), and substitution modules (modules that disappear, giving rise to a new graph fragment in its place with other modules, when all hyperparameters that the module depends on have been assigned values).

So far, we have only concerned ourselves with basic modules (e.g., the dense module in the example search spaces above). Basic modules are used to represent computations, i.e., the module implements some well-defined computation after values for all the hyperparameters of the module and values for the inputs are available. By contrast, substitution modules encode structural transformations (they do not implement any computation) based on the values of their hyperparameters. Substitution modules are inspired by ideas of delayed evaluation from the programming languages literature.

We have implemented many structural transformations as substitution modules in DeepArchitect. Substitution modules are that they are independent of the underlying framework of the basic modules (i.e., they work without any adaptation for Keras, Tensorflow, Scikit-Learn, or any other framework). Let us consider an example search space using a substitution module that either includes a submodule or not.


.. code:: python

    def search_space3():
        h_activation = D(["relu", "sigmoid"])
        h_units = D([32, 64, 128, 256])
        h_units_dep = co.DependentHyperparameter(lambda dh: 2 * dh["units"],
                                                 {"units": h_units})
        h_opt = D([0, 1])

        return mo.siso_sequential([
            dense(h_units, h_activation),
            mo.siso_optional(lambda: dense(h_units_dep, h_activation), h_opt),
            dense(D([10]), D(["softmax"]))
        ])


    (inputs, outputs) = wrap_search_space_fn(search_space3)()

The optional module takes a thunk (this terminology comes from programming languages) which returns a graph fragment (returned as a dictionary of input names to inputs and a dictionary of output names to outputs) which is called if the hyperparameter that determines if the thunk is to be called or not, takes the value "1" (i.e., the thunk is to be called, and the resulting graph fragment is to be included in the place of the substitution module). The visualization functionality is insightful in this case. Consider the graph evolution for a random sample from this search space.

.. code:: python

    vs = seco.random_specify(outputs)
    (inputs, outputs) = wrap_search_space_fn(search_space3)()

    vi.draw_graph_evolution(
        outputs,
        vs,
        '.',
        draw_module_hyperparameter_info=False,
        graph_name='graph3_evo')

Once the hyperparameter that the optional substitution module depends on is assigned a value, the substitution module disappears and is replaced by a graph fragment that depends on the hyperparameter value, i.e., if we decide to include it, the thunk is called returning a graph fragment; if we decide to not include it, an identity module (that passes the input to the output without changes) is substituted in its place.

Another simple substitution module is one that repeats a graph fragment in a serial connection. In this case, the substitution hyperparameter refers to how many times will the thunk returning a graph fragment will be called; all repetitions are connected in a serial connection.


.. code:: python

    def search_space4():
        h_activation = D(["relu", "sigmoid"])
        h_units = D([32, 64, 128, 256])
        h_units_dep = co.DependentHyperparameter(lambda dh: 2 * dh["units"],
                                                 {"units": h_units})
        h_opt = D([0, 1])
        h_num_repeats = D([1, 2, 4])

        return mo.siso_sequential([
            mo.siso_repeat(lambda: dense(h_units, h_activation), h_num_repeats),
            mo.siso_optional(lambda: dense(h_units_dep, h_activation), h_opt),
            dense(D([10]), D(["softmax"]))
        ])


    (inputs, outputs) = wrap_search_space_fn(search_space4)()

In the search space above, the hyperparameter for the
number of units of the dense modules inside the repeat share the same hyperparameter,
i.e., all these modules will have the same number of units.

.. code:: python

    vs = seco.random_specify(outputs)
    (inputs, outputs) = wrap_search_space_fn(search_space4)()

    vi.draw_graph_evolution(
        outputs,
        vs,
        '.',
        draw_module_hyperparameter_info=False,
        graph_name='graph4_evo')

In the graph evolution, we see that once we assign a value to the hyperparameter for the number of repetitions of the graph fragment returned by the thunk, a graph fragment with the serial connection of those many repetitions is substituted in its place. These example search spaces, along with their visualizations, should give the reader a sense about what structural decisions are expressible in DeepArchitect.

Substitution modules can be used in any place a module is needed, e.g., they can nested. For example, consider the following example

.. code:: python

    def search_space5():
        h_activation = D(["relu", "sigmoid"])
        h_units = D([32, 64, 128, 256])
        h_units_dep = co.DependentHyperparameter(lambda dh: 2 * dh["units"],
                                                 {"units": h_units})
        h_opt = D([0, 1])
        h_num_repeats = D([1, 2, 4])

        return mo.siso_sequential([
            mo.siso_repeat(lambda: dense(h_units, h_activation), h_num_repeats),
            mo.siso_optional(
                lambda: mo.siso_repeat(lambda: dense(h_units_dep, h_activation),
                                       h_num_repeats), h_opt),
            dense(D([10]), D(["softmax"]))
        ])


    (inputs, outputs) = wrap_search_space_fn(search_space5)()


Take one minute to think about the graph transitions for this search space; then run the code below to generate the actual visualization.

.. code:: python

    vs = seco.random_specify(outputs)
    (inputs, outputs) = wrap_search_space_fn(search_space5)()
    vi.draw_graph_evolution(
        outputs,
        vs,
        '.',
        draw_module_hyperparameter_info=False,
        graph_name='graph5_evo')

By using basic modules, substitution modules, independent hyperparameters, and dependent hyperparameters we are able to represent a large variety of search spaces in a compact and natural manner. As the reader becomes more comfortable with these concepts, it should become progressively easier to encode search spaces and appreciate the expressivity and reusability of the language.

Minor details
^^^^^^^^^^^^^

We cover additional details not yet discussed in the tutorial.

**Search space wrapper:** Throughout the instantiation of the various search spaces, we have seen this call to :code:`wrap_search_space_fn`, which internally uses :py:class:`deep_architect.modules.SearchSpaceFactory`. :py:class:`deep_architect.modules.SearchSpaceFactory` manages the global scope and buffers the search space to make sure that there are no substitution modules with unconnected inputs or outputs (i.e., at the border of the search space).

**Scope:** We use the global the scope to assign unique names to the elements that show up in the search space (currently, modules, hyperparameters, inputs, and outputs). Every time a module, hyperparameter, input, or output is created, we use the scope to assign a unique name to it. Every time that we want to start the search from scratch with a new search space, we should clear the scope to avoid keeping the names and objects from the previous samples around. In most cases, the user does not have to be concerned with the scope as :py:class:`deep_architect.modules.SearchSpaceFactory` can be used to handle the global scope.

**Details about substitution modules:** The search space cannot have substitution modules at its border as effectively substitution modules disappear once the substitution is done, and therefore references to the module and its inputs and outputs become invalid. :py:class:`deep_architect.modules.SearchSpaceFactory` creates and connects extra identity modules, which are basic modules (as opposed to substitution modules), before (in the case of inputs) or after (in the case of outputs) for each input and output belonging to a substitution module at the border of the search space.

**Auxiliary functions:** Besides basic modules and substitution modules, we also use several auxiliary functions for easily arranging graph fragments in different ways. These auxiliary function often do not create new modules, but use graph fragments or functions that return graph fragments to create a new graph fragment by using the arguments in a certain way. An example of a function of this type is :py:func:`deep_architect.modules.siso_sequential`, which just connects the graph fragments (expressed as a dictionary of inputs and a dictionary of outputs), in a serial connection, which just require us to connect inputs and outputs of the fragments passed as arguments to the function. Similarly to substitution modules, these auxiliary functions are framework independent as they only rely on properties of the module API. Using and defining auxiliary functions will help the user have a more effective and pleasant experience with DeepArchitect. Auxiliary functions are very useful to construct larger search spaces made of complex arrangements of smaller search spaces.

**Supporting other frameworks:** Basic modules are the only concepts that need to be specialized to the new framework. We recommend reading `deep_architect/core.py <https://github.com/negrinho/deep_architect/blob/master/deep_architect/core.py>`__ for extensive information about basic DeepArchitect API components. This code is the basis of DeepArchitect and has been extensively commented. Everything in `deep_architect/core.py <https://github.com/negrinho/deep_architect/blob/master/deep_architect/core.py>`__ is framework-independent. To better understand substitution modules and how they are implemented, read `deep_architect/modules.py <https://github.com/negrinho/deep_architect/blob/master/deep_architect/modules.py>`__ . We also point the reader to the tutorial about supporting new frameworks.

**Rerouting:** While we have not covered rerouting in this tutorial, it is reasonably straightforward to think about how to implement rerouting with, either as a substitution module or a basic module. For example, for a rerouting operation that takes `k` inputs and `k` outputs, and does a permutation of the inputs and outputs based on the value of an hyperparameter, if we implement this operation using a basic module, the basic module has to implement the chosen permutation when forward is called. If a substitution module is used instead, the module disappears once the value for the hyperparameter is chosen and the result of rerouting shows up in its place. After the user becomes proficient with the ideas of basic and substitution modules, the user will realize that oftentimes there are multiple ways of expressing the same search space.

Concluding remarks
^^^^^^^^^^^^^^^^^^

In this tutorial, we only covered basic functionality to encode search spaces over architectures. For learning more about the framework, please read more tutorials on aspects or use cases which you may find important and/or hard to understand. DeepArchitect is composed of many other components such as search, evaluation, logging, visualization and multiworking, so please read additional tutorials if you wish to become familiar with these other aspects.
