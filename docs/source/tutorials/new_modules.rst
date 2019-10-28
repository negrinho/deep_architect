
Implementing new modules
------------------------


In this tutorial, we will cover the implementation of new modules in DeepArchitect. We will use Keras to discuss the implementation of new modules. These aspects are similar across frameworks. See the corresponding examples and tutorials for discussions of how to support other frameworks.

Starting with the framework helper module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The starting point for implementing a new module is the the helper for the framework that we are using (in this case, Keras):

.. code:: python

    from __future__ import absolute_import
    import deep_architect.core as co


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

With this helper we can instantiate modules by passing values for the name of the module, the names of the inputs and outputs, the hyperparameters, and the compile function. Compile captures most of the functionality for the specific module. Calling the compile function passed as argument returns a function (the forward function). _compile is called only once. It may be informative to revisit the definition of a general module in core.py.

Instances of this class are sufficient for most use cases that we have encountered, but there may exist special cases where inheriting from this class and implementing _compile and _forward directly may be necessary.

Working with inputs and outputs instead of modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When writing down search spaces, we work
mostly with inputs and outputs, so the following auxiliary function is useful,
albeit a bit redundant.

.. code:: python

    def keras_module(name,
                     compile_fn,
                     name_to_hyperp,
                     input_names,
                     output_names,
                     scope=None):
        return KerasModule(name, name_to_hyperp, compile_fn, input_names,
                           output_names, scope).get_io()

See below for a typical implementation of a module using these auxiliary functions:

.. code:: python

    from keras.layers import Conv2D, BatchNormalization


    def conv_relu_batch_norm(h_filters, h_kernel_size, h_strides):

        def compile_fn(di, dh):
            m_conv = Conv2D(
                dh["filters"], dh["kernel_size"], dh["strides"], padding='same')
            m_bn = BatchNormalization()

            def forward_fn(di):
                return {"out": m_bn(m_conv(di["in"]))}

            return forward_fn

        return keras_module('ConvReLUBatchNorm', compile_fn, {
            "filters": h_filters,
            "kernel_size": h_kernel_size,
            'strides': h_strides
        }, ["in"], ["out"])

The forward function is defined via a closure. When compile is called, we have specific values for the module inputs (which in this example, are Keras tensor nodes). We can interact with these objects during compilation (e.g., look up dimensions for the input tensors). The compile function is called with a dictionary of inputs (whose keys are input names and whose values are input values) and a dictionary of outputs (whose keys are hyperparameter names and whose values are hyperparameter values). The forward function is called with a dictionary of input values. Values for the hyperparameters are accessible (due to being in the closure), but they are often not needed in the forward function.

Simplifications for single-input single-output modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

While the above definition is a bit verbose, we expect it to be clear. We introduce a additional functions to make the creation of modules less verbose. For example, we are often dealing with single-input single-output modules, so we define the following function:

.. code:: python

    def siso_keras_module(name, compile_fn, name_to_hyperp, scope=None):
        return KerasModule(name, name_to_hyperp, compile_fn, ['in'], ['out'],
                           scope).get_io()

This saves us writing the names of the inputs and outputs for the single-input single-output case. As the reader becomes familiar with DeepArchitect, the reader will notice that we use in/out names for single-input/single-output modules and in0, in1, .../out0, out1, ... for modules that often have multiple inputs/outputs. These names are arbitrary and can be chosen differently.

Using this function, the above example would be similar except that we would not need to name the input and output explicitly.

.. code:: python

    def conv_relu_batch_norm(h_filters, h_kernel_size, h_strides):

        def compile_fn(di, dh):
            m_conv = Conv2D(
                dh["filters"], dh["kernel_size"], dh["strides"], padding='same')
            m_bn = BatchNormalization()

            def forward_fn(di):
                return {"out": m_bn(m_conv(di["in"]))}

            return forward_fn

        return siso_keras_module('ConvReLUBatchNorm', compile_fn, {
            "filters": h_filters,
            "kernel_size": h_kernel_size,
            'strides': h_strides
        })

Easily creating modules from framework functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another useful auxiliary function creates a module from a function (e.g., most functions in keras.layers).

.. code:: python

    def siso_keras_module_from_keras_layer_fn(layer_fn,
                                              name_to_hyperp,
                                              scope=None,
                                              name=None):

        def compile_fn(di, dh):
            m = layer_fn(**dh)

            def forward_fn(di):
                return {"out": m(di["in"])}

            return forward_fn

        if name is None:
            name = layer_fn.__name__

        return siso_keras_module(name, compile_fn, name_to_hyperp, scope)

This function is convenient for functions that return a single-input single-output Keras module. For example, to get a convolutional module, we do

.. code:: python

    def conv2d(h_filters, h_kernel_size):
        return siso_keras_module_from_keras_layer_fn(Conv2D, {
            "filters": h_filters,
            "kernel_size": h_kernel_size
        })

If additionally, we would like to set some attributes to fixed values and have other ones defined through hyperparameters, we can do

.. code:: python

    def conv2d(h_filters, h_kernel_size):
        fn = lambda filters, kernel_size: Conv2D(
            filters, kernel_size, padding='same')
        return siso_keras_module_from_keras_layer_fn(
            fn, {
                "filters": h_filters,
                "kernel_size": h_kernel_size
            }, name="Conv2D")

Implementing new substitution modules
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

So far, we covered how can we easily implement new modules in Keras. These aspects transfer mostly without changes across frameworks. Examples correspond to modules that implement computation. We will now look at modules whose purpose is not to implement computation, but to perform a structural transformation based on the value of its hyperparameters. We call these modules substitution modules.

Substitution modules are framework independent. Only basic modules need to be reimplemented when porting search spaces from one framework to a different one. Substitution modules and auxiliary functions work across frameworks. As a result, a large amount of code is reusable between frameworks. Basic modules are often simple to implement, as most of the complexity of the search space implementation is contained in auxiliary functions and substitution modules.

First, consider the definition of a substitution module.

.. code:: python

    class SubstitutionModule(co.Module):
        """Substitution modules are replaced by other modules when the all the
        hyperparameters that the module depends on are specified.

        Substitution modules implement a form of delayed evaluation.
        The main component of a substitution module is the substitution function.
        When called, this function returns a dictionary of inputs and a dictionary
        of outputs. These outputs and inputs are used in the place the substitution
        module is in. The substitution module effectively disappears from the
        network after the substitution operation is done.
        Substitution modules are used to implement many other modules,
        e.g., :func:`mimo_or`, :func:`siso_optional`, and :func:`siso_repeat`.

        Args:
            name (str): Name used to derive an unique name for the module.
            name_to_hyperp (dict[str, deep_architect.core.Hyperparameter]): Dictionary of
                name to hyperparameters that are needed for the substitution function.
                The names of the hyperparameters should be in correspondence to the
                name of the arguments of the substitution function.
            substitution_fn ((...) -> (dict[str, deep_architect.core.Input], dict[str, deep_architect.core.Output]):
                Function that is called with the values of hyperparameters and
                returns the inputs and the outputs of the
                network fragment to put in the place the substitution module
                currently is.
            input_names (list[str]): List of the input names of the substitution module.
            output_name (list[str]): List of the output names of the substitution module.
            scope ((deep_architect.core.Scope, optional)) Scope in which the module will be
                registered. If none is given, uses the default scope.
            allow_input_subset (bool): If true, allows the substitution function to
                return a strict subset of the names of the inputs existing before the
                substitution. Otherwise, the dictionary of inputs returned by the
                substitution function must contain exactly the same input names.
            allow_output_subset (bool): If true, allows the substitution function to
                return a strict subset of the names of the outputs existing before the
                substitution. Otherwise, the dictionary of outputs returned by the
                substitution function must contain exactly the same output names.
        """

        def __init__(self,
                     name,
                     name_to_hyperp,
                     substitution_fn,
                     input_names,
                     output_names,
                     scope=None,
                     allow_input_subset=False,
                     allow_output_subset=False):
            co.Module.__init__(self, scope, name)
            self.allow_input_subset = allow_input_subset
            self.allow_output_subset = allow_output_subset

            self._register(input_names, output_names, name_to_hyperp)
            self._substitution_fn = substitution_fn
            self._is_done = False
            self._update()

        def _update(self):
            """Implements the substitution operation.

            When all the hyperparameters that the module depends on are specified,
            the substitution operation is triggered, and the substitution operation
            is done.
            """
            if (not self._is_done) and all(
                    h.has_value_assigned() for h in self.hyperps.values()):
                dh = {name: h.get_value() for name, h in self.hyperps.items()}
                new_inputs, new_outputs = self._substitution_fn(dh)

                # test for checking that the inputs and outputs returned by the
                # substitution function are valid.
                if self.allow_input_subset:
                    assert len(self.inputs) <= len(new_inputs) and all(
                        name in self.inputs for name in new_inputs)
                else:
                    assert len(self.inputs) == len(new_inputs) and all(
                        name in self.inputs for name in new_inputs)

                if self.allow_output_subset:
                    assert len(self.outputs) <= len(new_outputs) and all(
                        name in self.outputs for name in new_outputs)
                else:
                    assert len(self.outputs) == len(new_outputs) and all(
                        name in self.outputs for name in new_outputs)

                # performing the substitution.
                for name, old_ix in self.inputs.items():
                    old_ix = self.inputs[name]
                    if name in new_inputs:
                        new_ix = new_inputs[name]
                        if old_ix.is_connected():
                            old_ix.reroute_connected_output(new_ix)
                        self.inputs[name] = new_ix
                    else:
                        if old_ix.is_connected():
                            old_ix.disconnect()

                for name, old_ox in self.outputs.items():
                    old_ox = self.outputs[name]
                    if name in new_outputs:
                        new_ox = new_outputs[name]
                        if old_ox.is_connected():
                            old_ox.reroute_all_connected_inputs(new_ox)
                        self.outputs[name] = new_ox
                    else:
                        if old_ox.is_connected():
                            old_ox.disconnect_all()

                self._is_done = True

A substitution module has hyperparameters and a substitution function that returns a graph fragment to replace the substitution module. Update is called each time one of the hyperparameters of the substitution module is assigned a value. The substitution is performed when all hyperparameter have been assigned values.

Substitution modules disappear from the graph when the substitution is performed. The substitution function may itself return a graph fragment containing substitution modules. When there are only basic modules left and all the hyperparameters have been assigned values, the search space is fully specified and we can call the compile and forward functions for each of the basic modules in in it.

Substitution modules delay the choice of a structural property of the search space until some hyperparameters are assigned values. These are very helpful to encode complex and expressive search spaces. We have defined a few useful substitution modules in `deep_architect/modules.py <https://github.com/negrinho/deep_architect/blob/master/deep_architect/modules.py>`_. Similar to the basic module definition that we looked above, it is more convenient to deal with the dictionaries of inputs and the dictionaries of outputs than with the modules, so we define this function

.. code:: python

    def substitution_module(name,
                            name_to_hyperp,
                            substitution_fn,
                            input_names,
                            output_names,
                            scope,
                            allow_input_subset=False,
                            allow_output_subset=False,
                            unpack_kwargs=True):
        """Same as the substitution module, but directly works with the dictionaries of
        inputs and outputs.

        A dictionary with inputs and a dictionary with outputs is the preferred way
        of dealing with modules when creating search spaces. Using inputs and outputs
        directly instead of modules allows us to return graphs in the
        substitution function. In this case, returning a graph resulting of the
        connection of multiple modules is entirely transparent to the substitution
        function.

        See also: :class:`deep_architect.modules.SubstitutionModule`.

        Args:
            name (str): Name used to derive an unique name for the module.
            name_to_hyperp (dict[str, deep_architect.core.Hyperparameter]): Dictionary of
                name to hyperparameters that are needed for the substitution function.
                The names of the hyperparameters should be in correspondence to the
                name of the arguments of the substitution function.
            substitution_fn ((...) -> (dict[str, deep_architect.core.Input], dict[str, deep_architect.core.Output]):
                Function that is called with the values of hyperparameters and
                values of inputs and returns the inputs and the outputs of the
                network fragment to put in the place the substitution module
                currently is.
            input_names (list[str]): List of the input names of the substitution module.
            output_name (list[str]): List of the output names of the substitution module.
            scope (deep_architect.core.Scope): Scope in which the module will be registered.

        Returns:
            (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output]):
                Tuple with dictionaries with the inputs and outputs of the module.
        """
        return SubstitutionModule(
            name,
            name_to_hyperp,
            substitution_fn,
            input_names,
            output_names,
            scope,
            allow_input_subset=allow_input_subset,
            allow_output_subset=allow_output_subset,
            unpack_kwargs=unpack_kwargs).get_io()

We will now look at two specific examples of substitution modules. The or substitution module. is one of the simplest, but also most useful substitution modules:

.. code:: python

    def mimo_or(fn_lst, h_or, input_names, output_names, scope=None, name=None):
        """Implements an or substitution operation.

        The hyperparameter takes values that are valid indices for the list of
        possible substitution functions. The set of keys of the dictionaries of
        inputs and outputs returned by the substitution functions have to be
        the same as the set of input names and output names, respectively. The
        substitution function chosen is used to replace the current substitution
        module, with connections changed appropriately.

        .. note::
            The current implementation also works if ``fn_lst`` is an indexable
            object (e.g., a dictionary), and the ``h_or`` takes values that
            are valid indices for the indexable (e.g., valid keys for the dictionary).

        Args:
            fn_lst (list[() -> (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output])]):
                List of possible substitution functions.
            h_or (deep_architect.core.Hyperparameter): Hyperparameter that chooses which
                function in the list is called to do the substitution.
            input_names (list[str]): List of inputs names of the module.
            output_names (list[str]): List of the output names of the module.
            scope (deep_architect.core.Scope, optional): Scope in which the module will be
                registered. If none is given, uses the default scope.
            name (str, optional): Name used to derive an unique name for the
                module. If none is given, uses the class name to derive
                the name.

        Returns:
            (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output]):
                Tuple with dictionaries with the inputs and outputs of the
                substitution module.
        """

        def substitution_fn(idx):
            return fn_lst[idx]()

        return substitution_module(
            _get_name(name, "Or"), {'idx': h_or}, substitution_fn, input_names,
            output_names, scope)

The implementation is extremely short. This module has a single hyperparameter that chooses which function in the list (or dictionary) to call. Each of the functions in the list returns a dictionary of inputs and a dictionary of outputs when called.

.. code:: python

    def dnn_cell(h_num_hidden, h_nonlin_name, h_swap, h_opt_drop, h_opt_bn,
                 h_drop_keep_prob):
        return mo.siso_sequential([
            affine_simplified(h_num_hidden),
            nonlinearity(h_nonlin_name),
            mo.siso_permutation([
                lambda: mo.siso_optional(lambda: dropout(h_drop_keep_prob),
                                         h_opt_drop),
                lambda: mo.siso_optional(batch_normalization, h_opt_bn),
            ], h_swap)
        ])

Optional is a special case of a substitution module. If the hyperparameter is such that the function is to be used, then the function (in the example above, a lambda function) is called. Otherwise, an identity module that passes the input unchanged to the output is used.

.. code:: python

    def siso_optional(fn, h_opt, scope=None, name=None):
        """Substitution module that determines to include or not the search
        space returned by `fn`.

        The hyperparameter takes boolean values (or equivalent integer zero and one
        values). If the hyperparameter takes the value ``False``, the input is simply
        put in the output. If the hyperparameter takes the value ``True``, the search
        space is instantiated by calling `fn`, and the substitution module is
        replaced by it.

        Args:
            fn (() -> (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output])):
                Function returning a graph fragment corresponding to a sub-search space.
            h_opt (deep_architect.core.Hyperparameter): Hyperparameter for whether to
                include the sub-search space or not.
            scope (deep_architect.core.Scope, optional): Scope in which the module will be
                registered. If none is given, uses the default scope.
            name (str, optional): Name used to derive an unique name for the
                module. If none is given, uses the class name to derive the name.

        Returns:
            (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output]):
                Tuple with dictionaries with the inputs and outputs of the
                substitution module.
        """

        def substitution_fn(opt):
            return fn() if opt else identity()

        return substitution_module(
            _get_name(name, "SISOOptional"), {'opt': h_opt}, substitution_fn,
            ['in'], ['out'], scope)

An example of a complex substitution module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Let us now look at a more complex use of a custom substitution module.

.. code:: python

    def motif(submotif_fn, num_nodes):
        assert num_nodes >= 1

        def substitution_fn(dh):
            print dh
            node_id_to_node_ids_used = {i: [i - 1] for i in range(1, num_nodes)}
            for name, v in dh.items():
                if v:
                    d = ut.json_string_to_json_object(name)
                    i = d["node_id"]
                    node_ids_used = node_id_to_node_ids_used[i]
                    j = d["in_node_id"]
                    node_ids_used.append(j)
            for i in range(1, num_nodes):
                node_id_to_node_ids_used[i] = sorted(node_id_to_node_ids_used[i])

            (inputs, outputs) = mo.identity()
            node_id_to_outputs = [outputs]
            in_inputs = inputs
            for i in range(1, num_nodes):
                node_ids_used = node_id_to_node_ids_used[i]
                num_edges = len(node_ids_used)

                outputs_lst = []
                for j in node_ids_used:
                    inputs, outputs = submotif_fn()
                    j_outputs = node_id_to_outputs[j]
                    inputs["in"].connect(j_outputs["out"])
                    outputs_lst.append(outputs)

                # if necessary, concatenate the results going into a node
                if num_edges > 1:
                    c_inputs, c_outputs = combine_with_concat(num_edges)
                    for idx, outputs in enumerate(outputs_lst):
                        c_inputs["in%d" % idx].connect(outputs["out"])
                else:
                    c_outputs = outputs_lst[0]
                node_id_to_outputs.append(c_outputs)

            out_outputs = node_id_to_outputs[-1]
            return in_inputs, out_outputs

        name_to_hyperp = {
            ut.json_object_to_json_string({
                "node_id": i,
                "in_node_id": j
            }): D([0, 1]) for i in range(1, num_nodes) for j in range(i - 1)
        }
        return mo.substitution_module(
            "Motif", name_to_hyperp, substitution_fn, ["in"], ["out"], scope=None)

This substitution module implements the notion of a motif inspired by this `paper <https://arxiv.org/abs/1711.00436>`_. This substitution module delays the creation of the motif structure until hyperparameters determining the connections of the motif are assigned values. The notion of a motif defined in the paper is recursive. The motif function takes a submotif function.

Concluding remarks
^^^^^^^^^^^^^^^^^^

This concludes our discussion about how to implement new modules in a specific framework that the reader is working with. We point the reader to the new_frameworks tutorial for learning about how to support a new framework by specializing the module class and to the search space constructs tutorials for a more in-depth coverage of how search spaces can be created by interconnecting modules.
