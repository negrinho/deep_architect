
Supporting new frameworks
-------------------------

Supporting a framework in DeepArchitect requires the specialization of the module definition for that particular framework. For the frameworks that we currently support, the necessary changes across frameworks are minimal resulting mostly from framework idiosyncrasies, e.g., what information is needed to create a computational graph.

DeepArchitect is not limited to the frameworks that we currently support. Aside from module specialization, most of the other code in DeepArchitect is general and can be reused without changes across frameworks, e.g., searchers, logging, and visualization. We will walk the reader over the implementation of some of the helpers for the frameworks that are currently supported.

Specializing the module class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main module functions to look at are:

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

After all hyperparameters for a search space are specified, we can compile its modules. After all the independent hyperparameters have been assigned values, all substitution modules will have been disappeared and only basic modules will be in place. We can then execute the network. This can be seen in :py:meth:`deep_architect.core.Module.forward`. When :code:`forward` is called, :code:`_compile` is called once and then :code:`forward` is called right after. Compilation can be used to instantiate any state that is used by the module for the forward computation, e.g., creation of parameters. After compilation, :code:`forward` can be called multiple times to perform the computation repeatedly.

Let us look in detail at concrete examples for frameworks that are currently
supported in DeepArchitect.

Keras helpers
^^^^^^^^^^^^^

Let us look at the Keras helper defined in
`deep_architect/helpers/keras.py <https://github.com/negrinho/deep_architect/blob/master/deep_architect/helpers/keras.py>`_.

.. code:: python

    import deep_architect.core as co
    import tensorflow as tf
    import numpy as np


    class KerasModule(co.Module):
        """Class for taking Keras code and wrapping it in a DeepArchitect module.

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
            functionality in a DeepArchitect module. The instantiation of the Keras
            variables is taken care by the `compile_fn` function that takes a two
            dictionaries, one of inputs and another one of outputs, and
            returns another function that takes a dictionary of inputs and creates
            the computational graph. This functionality makes extensive use of closures.

            The keys of the dictionaries that are passed to the compile
            and forward function match the names of the inputs and hyperparameters
            respectively. The dictionary returned by the forward function has keys
            equal to the names of the outputs.

            This implementation is very similar to the implementation of the Tensorflow
            helper :class:`deep_architect.helpers.tensorflow_support.TensorflowModule`.

        Args:
            name (str): Name of the module
            name_to_hyperp (dict[str,deep_architect.core.Hyperparameter]): Dictionary of
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
            scope (deep_architect.core.Scope, optional): Scope where the module will be
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

The code is compact and self-explanatory. We pass a function called :code:`compile_fn` that returns a function called :code:`forward_fn` function upon compilation. To instantiate a module, we simply have to provide a compile function that returns a forward function when called. For example, for implementing a convolutional module from scratch relying on this module (check the Keras docstring for :code:`Conv2D`), we would do:

.. code:: python

    from keras.layers import Conv2D


    def conv2d(h_filters, h_kernel_size, h_strides, h_activation, h_use_bias):

        def compile_fn(di, dh):
            m = Conv2D(**dh)

            def forward_fn(di):
                return {"out": m(di["in"])}

            return forward_fn

        return KerasModule(
            "Conv2D", {
                "filters": h_filters,
                "kernel_size": h_kernel_size,
                "strides": h_strides,
                "activation": h_activation,
                "use_bias": h_use_bias
            }, compile_fn, ["in"], ["out"]).get_io()

A few points to pay attention to:

-   Input, output and hyperparameter names are specified when creating an instance of :code:`KerasModule`.

-   :code:`di` and :code:`dh` are dictionaries with inputs names mapping to input values and hyperparameter names mapping to hyperparameter values, respectively.

-   :code:`Conv2D(**dh)` uses dictionary unpacking to call the Keras function that instantiates a Keras layer (as in the Keras API). We could also have done the unpacking manually.

-   Upon the instantiation of the Keras module, we call :code:`get_io` to get a pair :code:`(inputs, outputs)`, where both :code:`inputs` and :code:`outputs` are dictionaries, where :code:`inputs` maps input names to input objects (i.e., an object from the class :py:class:`deep_architect.core.Input`), and :code:`outputs` maps output names to output objects (i.e., an object from the class :py:class:`deep_architect.core.Output`). Working directly with dictionaries of inputs and outputs is more convenient than working with modules, because we can transparently work with subgraph structures without concerning ourselves about whether they are composed of multiple modules or not.

A minimal example to go from this wrapper code to an instantiated Keras model is:

.. code:: python

    from keras.layers import Input
    import deep_architect.hyperparameters as hp
    import deep_architect.core as co
    from deep_architect.searchers.common import random_specify
    from keras.models import Model

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
    random_specify(outputs)
    co.forward({inputs["in"]: x})
    out = outputs["out"].val
    model = Model(inputs=x, outputs=out)
    model.summary()

    import deep_architect.visualization as vi
    vi.draw_graph(outputs, draw_module_hyperparameter_info=False)

As modules with a single input and a single output are common, we defined a few simplified functions that directly work with the Keras definition. The goal of these functions is to reduce boilerplate and provide a more concise workflow. For example, the above function could be expressed in the same way as:

.. code:: python

    import deep_architect.helpers.keras_support as hke


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
    co.forward({inputs["in"]: x})
    out = outputs["out"].val
    model = Model(inputs=x, outputs=out)
    model.summary()
    vi.draw_graph(outputs, draw_module_hyperparameter_info=False)

We refer the reader to `deep_architect.helpers.keras_support <https://github.com/negrinho/deep_architect/blob/master/deep_architect/helpers/keras.py>`__ if the reader wishes to inspect the implementation of this function and how does it fit with the previous definition for a Keras module. These functions require minimal additional code. These auxiliary functions are convenient to reduce boilerplate for some of the most common use cases. As we have seen, it is possible to express everything that we need using :code:`KerasModule`, with the other functions used for convenience for common specific cases.

Calls to :py:func:`deep_architect.core.forward` call the individual module forward and compile functions as defined in :code:`KerasModule` and passed as argument during the instantiation. We invite the reader to inspect :py:func:`deep_architect.core.forward` (found `here <https://github.com/negrinho/deep_architect/blob/master/deep_architect/core.py>`__) to understand how it is implemented using graph traversal. This is sufficient to specialize the general module code in :py:mod:`deep_architect.core` to support basic modules from Keras.

Pytorch helpers
^^^^^^^^^^^^^^^

The reader may think that Pytorch does not fit well in our framework due to being a dynamic framework where the graph used for back propagation is defined for each instance, i.e., defined by run, rather than static (as it is the case of Keras) where the graph is defined upfront and used across instances. Static versus dynamic is not an important distinction for architecture search in DeepArchitect. For example, we can search over computational elements that are used dynamically by the model, e.g., a recurrent cell.

Let us quickly walk through the DeepArchitect module specialization for PyTorch. We omit the docstring due to the similarity with the one for :code:`KerasModule.forward`.

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

We can see that the implementation for PyTorch is essentially the same as the one for Keras. The main difference is that the compile_fn function that returns both :code:`forward_fn` and the list of Pytorch modules (as in :code:`nn.Module`) that have been used in the computation. This list is used to keep track of which Pytorch modules are used by the DeepArchitect module. For example, this is necessary to move them to the GPU or CPU, or get their parameters. Changes from Tensorflow to Pytorch are mainly a result of the differences in how these two frameworks declare computational graphs.

.. code:: python

    import deep_architect.helpers.pytorch_support as hpy


    def conv2d(h_filters, h_kernel_size, h_strides, h_activation, h_use_bias):

        def compile_fn(di, dh):
            m = Conv2D(**dh)

            def forward_fn(di):
                return {"out": m(di["in"])}

            return forward_fn

        return PyTorchModule(
            "Conv2D", {
                "filters": h_filters,
                "kernel_size": h_kernel_size,
                "strides": h_strides,
                "activation": h_activation,
                "use_bias": h_use_bias
            }, compile_fn, ["in"], ["out"]).get_io()


    def conv2d_pytorch(h_filters, h_kernel_size, h_strides, h_activation,
                       h_use_bias):
        return hpy.siso_pytorch_module_from_pytorch_layer_fn(
            Conv2D, {
                "filters": h_filters,
                "kernel_size": h_kernel_size,
                "strides": h_strides,
                "activation": h_activation,
                "use_bias": h_use_bias
            })

Concluding remarks
^^^^^^^^^^^^^^^^^^

DeepArchitect is not limited to deep learning frameworks---any domain that for which we can define notions of compile and forward (or potentially, other operations) as they were discussed above can be supported. Another aspect to keep in mind is that there is not a need for all the modules of the computational graph to be in the same domain (e.g., a preprocessing component followed by the actual graph propagation).

We showcased support for both static (Keras) and dynamic deep learning (PyTorch) frameworks. The notions of basic modules, substitution modules, independent hyperparameters, and dependent hyperparameters are general and can be used across a large range of settings (e.g., scikit-learn or data augmentation pipelines). We leave the consideration of other frameworks (deep learning or otherwise) to the reader.
