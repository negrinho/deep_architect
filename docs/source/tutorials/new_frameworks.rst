
Supporting new frameworks
-------------------------

This tutorial aims to explain to you how to support a new framework in
DeepArchitect. After reading it, you should understand what are
the components involved in supporting a framework in DeepArchitect and go
about implementing the necessary ones yourself.

Implementing support for a framework on DeepArchitect heavily depends on
specializing the definition of module for that particular framework.
For most frameworks that we currently support, the necessary changes are
minimal and result mostly from idiosyncrasies from the specific framework, e.g.,
what information needs to kept around.
DeepArchitect is not limited to the frameworks that we currently support.
Asides from the module, most of the other code in DeepArchitect is general
and can be reused without changes across frameworks, e.g., searchers, logging, and visualization.

To exemplify this, we walk the reader over the implementation of existing
helpers for supported frameworks and a new example in scikit-learn.

The main functions to look at are:

.. code:: python

    # def _compile(self):
    #     """Compile operation for the module.

    #     Called once when all the hyperparameters that the module depends on,
    #     and the other hyperparameters of the search space are specified.
    #     See also: :meth:`_forward`.
    #     """
    #     raise NotImplementedError

    # def _forward(self):
    #     """Forward operation for the module.

    #     Called once the compile operation has been called. See also: :meth:`_compile`.
    #     """
    #     raise NotImplementedError

    # def forward(self):
    #     """The forward computation done by the module is decomposed into
    #     :meth:`_compile` and :meth:`_forward`.

    #     Compile can be thought as creating the parameters of the module (done
    #     once). Forward can be thought as using the parameters of the module to
    #     do the specific computation implemented by the module on some specific
    #     data (done multiple times).

    #     This function can only called after the module and the other modules in
    #     the search space are fully specified. See also: :func:`forward`.
    #     """
    #     if not self._is_compiled:
    #         self._compile()
    #         self._is_compiled = True
    #     self._forward()

After all hyperparameters for a search space are specified, we can finally
compile the modules in the search space.
By the time that all the hyperparameters of the search space have been
assigned a value, all substitution modules ought to have been replaced and
only basic modules will be in place.
This can be seen in the forward function in Module.
When forward is called, _compile is called once and then forward is called
right after.
The compilation stage can be used to instantiate any state that is used for the
module for the forward part, e.g., creation of parameters in the case of
deep learning search spaces.
After compiling is done, the forward function can be called to implement the
computation repeatedly.

Let us look in detail at concrete exmaples for frameworks that are currently supported
in DeepArchitect.
Let us look at the Keras defined in
`deep_architect/helpers/keras.py <https://github.com/negrinho/darch/blob/master/deep_architect/helpers/keras.py>`_

.. code:: python

    import deep_architect.core as co
    import tensorflow as tf
    import numpy as np


    class KerasModule(co.Module):
        """Class for taking Keras code and wrapping it in a darch module.

        This class subclasses :class:`deep_architect.core.Module` as therefore inherits all
        the functionality associated to it (e.g., keeping track of inputs, outputs,
        and hyperparameters). It also enables to do the compile and forward
        operations for these types of modules once a module is fully specified,
        i.e., once all the hyperparameters have been chosen.

        The compile operation in this case creates all the variables used for the
        fragment of the computational graph associated to this module.
        The forward operation takes the variables that were created in the compile
        operation and constructs the actual computational graph fragment associated
        to this module.

        .. note::
            This module is abstract, meaning that it does not actually implement
            any particular Keras computation. It simply wraps Keras
            functionality in a DeepArchitect module. The instantation of the Keras
            variables is taken care by the `compile_fn` function that takes a two
            dictionaries, one of inputs and another one of outputs, and
            returns another function that takes a dictionary of inputs and creates
            the computational graph. This functionality makes extensive use of closures.

            The keys of the dictionaries that are passed to the compile
            and forward function match the names of the inputs and hyperparameters
            respectively. The dictionary returned by the forward function has keys
            equal to the names of the outputs.

            This implementation is very similar to the implementation of the Tensorflow
            helper :class:`deep_architect.helpers.tensorflow.TensorflowModule`.

        Args:
            name (str): Name of the module
            name_to_hyperp (dict[str,darch.core.Hyperparameter]): Dictionary of
                hyperparameters that the model depends on. The keys are the local
                names of the hyperparameters.
            compile_fn ((dict[str,object], dict[str,object]) -> (dict[str,object] -> dict[str,object])):
                The first function takes two dictionaries with
                keys corresponding to `input_names` and `output_names` and returns
                a function that takes a dictionary with keys corresponding to
                `input_names` and returns a dictionary with keys corresponding
                to `output_names`. The first function may also return
                two additional dictionaries mapping Tensorflow placeholders to the
                values that they will take during training and test.
            input_names (list[str]): List of names for the inputs.
            output_names (list[str]): List of names for the outputs.
            scope (darch.core.Scope, optional): Scope where the module will be
                registered.
        """

        def __init__(self,
                     name,
                     name_to_hyperp,
                     compile_fn,
                     input_names,
                     output_names,
                     scope=None):
            co.Module.__init__(self, scope, name)

            self._register(input_names, output_names, name_to_hyperp)
            self._compile_fn = compile_fn

        def _compile(self):
            input_name_to_val = self._get_input_values()
            hyperp_name_to_val = self._get_hyperp_values()
            self._fn = self._compile_fn(input_name_to_val, hyperp_name_to_val)

        def _forward(self):
            input_name_to_val = self._get_input_values()
            output_name_to_val = self._fn(input_name_to_val)
            self._set_output_values(output_name_to_val)

        def _update(self):
            pass


The code is compact and self-explanatory.
In this case, the we pass a compile_fn function that returns the forward_fn
function upon compilation.
To instantiate a module of this type we simply have to provide a compile function
that upon calling, returns a forward function.
For example, for implementing a convolutional module from scratch relying on this
module, we would do

For example, by looking at the Keras docstring for the Conv2D and taking a
subset of the options, we can write:

.. code:: python

    from keras.layers import Conv2D


    def conv2d(h_filters, h_kernel_size, h_strides, h_activation, h_use_bias):

        def compile_fn(di, dh):
            m = Conv2D(**dh)

            def forward_fn(di):
                return {"Out": m(di["In"])}

            return forward_fn

        return KerasModule(
            "Conv2D", {
                "filters": h_filters,
                "kernel_size": h_kernel_size,
                "strides": h_strides,
                "activation": h_activation,
                "use_bias": h_use_bias
            }, compile_fn, ["In"], ["Out"]).get_io()


A few points to pay attention to:

-   Input, output and hyperparameter names are specified when instantiating the
    KerasModule.

-   di and dh are dictionaries with inputs names mapping to input values and
    hyperparameter names mapping to hyperparameter values.

-   In the line :code:`Conv2D(**dh)`, simply used the dictionary unpacking functionality
    to call the Keras function that instantiates a Keras layer (as in the Keras
    API). We could have done the unpacking manually and perform additional computation.

-   Upon the instantiation of the Keras modules, we call get_io to get a pair
    (inputs, outputs), where both inputs and outputs are dictionaries, where
    inputs maps input names to input objects (i.e., an object from the class
    deep_architect.core.Input), and outputs maps output names to output objects
    (i.e., an object from the class deep_architect.core.Output).
    This is done because the search space constructs work directly on these dictionaries
    rather than on modules.
    Dealing directly with inputs and outputs makes the framework more easy to use
    because we can transparently work over subgraph structures without ever concerning
    ourselves about whether they are composed of multiple modules or not.

A minimal example to go from this wrapper code to an instantiated Keras
model would be.

.. code:: python

    from keras.layers import Input
    import deep_architect.hyperparameters as hp
    import deep_architect.core as co
    from deep_architect.searchers.common import random_specify
    from keras.models import Model

    # NOTE: this should probably be removed.
    # TODO: go over this case and see if there is stuff to simplify with getting
    # the values. maybe it should have namespace of our
    # TODO: I think that one good way of going about it to create a search space
    # factory from the inputs and outputs.
    # SearchSpaceFactory(search_space_fn); this is enough.
    # I think that all of the searchers will expect a search space factory because
    # it prevents you from shooting in the foot.
    D = hp.Discrete
    # specifying all the hyperparameters.
    x = Input((32, 32, 3), dtype='float32')
    h_filters = D([32, 64])
    h_kernel_size = D([1, 3, 5])
    h_strides = D([1])
    h_activation = D(['relu', 'sigmoid'])
    h_use_bias = D([0, 1])
    (inputs, outputs) = conv2d(h_filters, h_kernel_size, h_strides, h_activation,
                               h_use_bias)
    random_specify(outputs.values())
    co.forward({inputs["In"]: x})
    out = outputs["Out"].val
    model = Model(inputs=x, outputs=out)
    model.summary()

    import deep_architect.visualization as vi
    vi.draw_graph(outputs.values(), draw_module_hyperparameter_info=False)

As modules with single inputs and single outputs are so common, we defined
a few simplified functions that directly work with the Keras definition.
The goal of these functions is to reduce boilerplate and provide a more
concise workflow.
For example, the above function could be expressed in the same way as

.. code:: python

    import deep_architect.helpers.keras as hke


    def conv2d(h_filters, h_kernel_size, h_strides, h_activation, h_use_bias):
        return hke.siso_keras_module_from_keras_layer_fn(
            Conv2D, {
                "filters": h_filters,
                "kernel_size": h_kernel_size,
                "strides": h_strides,
                "activation": h_activation,
                "use_bias": h_use_bias
            })


    (inputs, outputs) = conv2d(h_filters, h_kernel_size, h_strides, h_activation,
                               h_use_bias)
    co.forward({inputs["In"]: x})
    out = outputs["Out"].val
    model = Model(inputs=x, outputs=out)
    model.summary()
    vi.draw_graph(outputs.values(), draw_module_hyperparameter_info=False)

We refer the reader to deep_architect.helpers.keras if the reader wishes to
inspect the implementation of this function and how does it fit with the
previous definition for a Keras module.
We promise that this functions to minimal additional code.
The main motivation to have these auxiliary functions in place is to
reduce boilerplate for some of the most common use cases.
As we have seen, it is possible to express everything that we need using
the initial KerasModule, with the other functions being for the purpose of
convenience for common specific cases.
It may be necessary to use KerasModule directly for implementing the
desired functionality in some cases, e.g., in the case of a module with multiple
outputs.

The co.forward calls the individual module forward and compile functions
as defined in KerasModule and passed as argument during the instantiation.
These are the main ideas for defining a module.
We invite the reader to inspect deep_architect.core.forward more carefully
for drilling down on how deep_architect.core.forward is defined is implemented
in terms of graph traversal.

This is sufficient to specialize the general module code in deep_architect.core
to support basic modules that come from Keras.

Let us now consider Pytorch.
The reader may think that Pytorch does not fit well in our framework due to
being a dynamic framework where the graph that is used for back propagation
is defined for each instance, i.e., defined by run, rather than static (as it is
the case of Keras) where the graph is defined upfront and used multiple times
for both training and inference.
Static versus dynamic is not really important for architecture search in
DeepArchitect.
There are multiple ways of getting around, e.g., searching over the
computational elements that are used in a dynamic element of the network.

Let us quickly walk through the DeepArchitect module specialization for
DeepArchitect.
We omit the docstring due to the similarity with the one for KerasModule.forward

.. code:: python


    class PyTorchModule(co.Module):

        def __init__(self,
                     name,
                     name_to_hyperp,
                     compile_fn,
                     input_names,
                     output_names,
                     scope=None):
            co.Module.__init__(self, scope, name)
            self._register(input_names, output_names, name_to_hyperp)
            self._compile_fn = compile_fn

        def _compile(self):
            input_name_to_val = self._get_input_values()
            hyperp_name_to_val = self._get_hyperp_values()
            self._fn, self.pyth_modules = self._compile_fn(input_name_to_val,
                                                           hyperp_name_to_val)
            for pyth_m in self.pyth_modules:
                assert isinstance(pyth_m, nn.Module)

        def _forward(self):
            input_name_to_val = self._get_input_values()
            output_name_to_val = self._fn(input_name_to_val)
            self._set_output_values(output_name_to_val)

        def _update(self):
            pass


We can see that the implementation for PyTorch is essentially the same as the
one for Keras.
The main difference is that the compile_fn function that returns both the
forward_fn function and the list of Pytorch modules (as in nn.Module) that have
been used in the computation.
Returning the list of modules is used to keep track of what Pytorch modules
are in use by the DeepArchitect module, which is necessary if we want to move them
to the GPU or CPU, or get their parameters.
As we see, the changes from Tensorflow to Pytorch are mainly a result by the
differences in our these two frameworks handle the declaration of computational
graphs.
Hopefully, this conveys to the reader the considerations that should be taken
when implemeneting support for a new framework in DeepArchitect.

.. code:: python


    def conv2d(h_filters, h_kernel_size, h_strides, h_activation, h_use_bias):

        def compile_fn(di, dh):
            m = Conv2D(**dh)

            def forward_fn(di):
                return {"Out": m(di["In"])}

            return forward_fn

        return KerasModule(
            "Conv2D", {
                "filters": h_filters,
                "kernel_size": h_kernel_size,
                "strides": h_strides,
                "activation": h_activation,
                "use_bias": h_use_bias
            }, compile_fn, ["In"], ["Out"]).get_io()


    def conv2d_pytorch(h_filters, h_kernel_size, h_strides, h_activation,
                       h_use_bias):
        return hke.siso_keras_module_from_keras_layer_fn(
            Conv2D, {
                "filters": h_filters,
                "kernel_size": h_kernel_size,
                "strides": h_strides,
                "activation": h_activation,
                "use_bias": h_use_bias
            })


DeepArchitect is not limited to deep learning frameworks---any domain that for
which we can define notions of compile and forward as they were discussed above
can be supported as above.
# TODO: perhaps move this to a different place.
Another aspect to keep in mind is that there is not a need for all the modules
of the computational graph to be in the same domains (e.g., a preprocessing
component followed by the actual graph propagation).
For the Tensorflow example, we have considered cases where we have mostly
Tensorflow operations flowing through the graph, but this is not necessarily
the case.
As long as the module gets inputs and hyperparameters values that work in the
context of its forward and compile functions, then everything works as expected.
This allows us to create search spaces with multiple different domains, e.g.,
for the Tensorflow case, some of the modules may produce variables and others
may produce Tensorflow operations.
DeepArchitect is a framework to search over computational graphs in arbitrary
domains.

We showcased support for both a static and a dynamic deep learning frameworks
here.
The notions of basic modules, substitution modules, independent hyperparameters, and
dependent hyperparameters are very general and can be used across a large range
of settings (e.g., scikit-learn or data augmentation pipelines).
We leave the consideration of these other non deep learning frameworks to the
reader.