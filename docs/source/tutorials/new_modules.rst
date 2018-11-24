
Implementing new modules
------------------------


In this tutorial, we will cover how to implement new modules in DeepArchitect.
We will focus specifically on how to implement modules within a specific
framework. For a discussion of how to support new frameworks, see tutorial
new_frameworks.

We will use the Keras support to discuss the relevant aspects that come into
play when implementing new modules, but these aspects are almost the same across
frameworks, so we believe that the reader will be able to get the gist of it
for other frameworks.

Again, the main starting point of implenting a new module is the implementation of
the helper for the particular framework that we are dealing with, in this
case, Keras:

.. code:: python

    from __future__ import absolute_import
    import deep_architect.core as co


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


With this helper, creating new functions is a matter of instantiating modules
by passing the appropriate values for the name of the module, the names of the
inputs and outputs, the hyperparameters, and the compile function.
The compile function is perhaps the place that captures most of the speficic
functionality for the module in question that we want to implement.
Calling the compile function passed as argument returns a function, called that
we call the forward function. The _compile function is called only once.
It may be informative to revisit the definition of a general module in core.py.
Some aspects to note in the above definition are the


Instances of this class are sufficient for most use cases that we have encountered,
but there may exist special cases where inheriting from this class and implementing
the _compile and _forward functions directly may be necessary.
Another aspect to keep in mind is that in writing down search spaces, we work
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


A typical implementation of a module using these auxiliary functions is like this

.. code:: python

    from keras.layers import Conv2D, BatchNormalization


    def conv_relu_batch_norm(h_filters, h_kernel_size, h_strides):

        def compile_fn(di, dh):
            m_conv = Conv2D(
                dh["filters"], dh["kernel_size"], dh["strides"], padding='same')
            m_bn = BatchNormalization()

            def forward_fn(di):
                return {"Out": m_bn(m_conv(di["In"]))}

            return forward_fn

        return keras_module('ConvReLUBatchNorm', compile_fn, {
            "filters": h_filters,
            "kernel_size": h_kernel_size,
            'strides': h_strides
        }, ["In"], ["Out"])


We see that the implementation is straighforward. The forward function is defined
via a function closure. At the time that the compile function is called, we do
have specific values for the inputs of the module, which in this case are Keras
tensor nodes. If we were dealing with Tensorflow, these would Tensorflow op
nodes. This means that it is possible to interact with these objects when
the compile function is called, lookup information on them (e.g., dimensions
of the tensors), and implement compile time conditions based on them.
The compile function is called with a dictionary of inputs (whose keys are input
names and whose values are input values) and a dictionary of outputs
(whose keys are hyperparameter names and whose values are hyperparameter values).
The forward function is simply called with a dictionary of input values.
Values for the hyperparameters are accessible (due to being in the closure),
but they are often not needed inside the forward function.

While the above definition is a bit verbose, we expect it to be very straightforward
in what it is doing and how it is interacting with the Keras module helper
that we just presented above.
To make the creation of modules less verbose, we introduce a few additional functions.
For example, it is typical that we are often dealing with single input and single
output modules, so we have defined the following function.

.. code:: python


    def siso_keras_module(name, compile_fn, name_to_hyperp, scope=None):
        return KerasModule(name, name_to_hyperp, compile_fn, ['In'], ['Out'],
                           scope).get_io()


This essentially saves us writing the names of the inputs and outputs for the
single input and single output case. As the reader becomes familiar with
DeepArchitect, the reader will notice that we use In/Out names for single
input modules and In0, In1, ... and Out0, Out1, ... for modules that often
have multiple inputs and/or outputs. These names are arbitrary and can be chosen
differently.

Using this function, the above example would be written entirely similarly,
except that we do not need that we do not need to name the input and output
explicitly, as they will just take the default names of In and Out.

.. code:: python


    def conv_relu_batch_norm(h_filters, h_kernel_size, h_strides):

        def compile_fn(di, dh):
            m_conv = Conv(
                dh["filters"], dh["kernel_size"], dh["strides"], padding='same')
            m_bn = BatchNormalization()

            def forward_fn(di):
                return {"Out": m_bn(m_conv(di["In"]))}

            return forward_fn

        return keras_module('ConvReLUBatchNorm', compile_fn, {
            "filters": h_filters,
            "kernel_size": h_kernel_size,
            'strides': h_strides
        })


Another auxiliary function that can be quite useful is to create a module
directly from a function (e.g., most of the functions defined in keras.layers)
that returns a Keras module.

.. code:: python


    def siso_keras_module_from_keras_layer_fn(layer_fn,
                                              name_to_hyperp,
                                              scope=None,
                                              name=None):

        def compile_fn(di, dh):
            m = layer_fn(**dh)

            def forward_fn(di):
                return {"Out": m(di["In"])}

            return forward_fn

        if name is None:
            name = layer_fn.__name__

        return siso_keras_module(name, compile_fn, name_to_hyperp, scope)


This function is convenient from extremely simple and short cases for
functions that return directly a single input single output Keras module.
For example, for getting a convolutional module, we can do

.. code:: python


    def conv2d(h_filters, h_kernel_size):
        return siso_keras_module_from_keras_layer_fn(Conv2D, {
            "filters": h_filters,
            "kernel_size": h_kernel_size
        })


If additionaly, we would like to set some attributes to fixed values and have
other ones defined through hyperparameters, we can do as such

.. code:: python


    def conv2d(h_filters, h_kernel_size):
        fn = lambda filters, kernel_size: Conv2D(
            filters, kernel_size, padding='same')
        return siso_keras_module_from_keras_layer_fn(
            fn, {
                "filters": h_filters,
                "kernel_size": h_kernel_size
            }, name="Conv2D")


So far, we covered how can we easily implement new modules in a framework
that we are working with. These examples were all focused on Keras, but these
aspects that we covered so far trasfer mostly without changes across frameworks.
All the aspects that we have seen so far correspond to examples of modules that
actually implement computation. We will now look at examples of modules whose
purpose is not to implement computation, but to perform a structural transformation
based on the value of its hyperparameters. We call these modules substitution
modules. One of the big advantages of substitution modules is that they are
independent of the framework that we are working with. This means that
upon porting one search space from one framework to a different one, the only
modules that need to be ported are the basic modules. Any auxiliary functions that
simply put modules together and substitution modules work automatically across frameworks.
This means that a large amount of code is reusable when moving from one framework
to another one. The basic modules are often very simply to implement, being the
auxiliary functions and the substitution modules that often contain most of the
complexity of the search space.

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
                    h.has_value_assigned() for h in itervalues(self.hyperps)):
                dh = {name: h.get_value() for name, h in iteritems(self.hyperps)}
                new_inputs, new_outputs = self._substitution_fn(**dh)

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
                for name, old_ix in iteritems(self.inputs):
                    old_ix = self.inputs[name]
                    if name in new_inputs:
                        new_ix = new_inputs[name]
                        if old_ix.is_connected():
                            old_ix.reroute_connected_output(new_ix)
                        self.inputs[name] = new_ix
                    else:
                        if old_ix.is_connected():
                            old_ix.disconnect()

                for name, old_ox in iteritems(self.outputs):
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


The reader will not get all the details by looking at this, but the main idea is
that the substitution module has some hyperparameters associated to it and a
substitution function that returns a graph fragment that is used in the same
place of where the substitution module was before the substitution.
The main method in the case of the substitution module is update, which
is called each time one of the hyperparameters that is associated to the
substitution module is assigned until finally all hyperparameters have a value
assigned. The substitution is then performed.
Substitution modules disappear from the graph when the subsitution is performed,
being replaced by some graph fragment that is returned by the substitution function.
The substitution function may itself return a graph fragment with substitution
modules, which means that the process of substitution will proceed recursively
until there are only basic modules. At that point, the search space is fully
specified and we can call the compile and forward functions for each of the
basic modules involved in it.
The way to think about substitution modules is that they delay the choice of
some structural property of the search space until some hyperparameters are
assigned a value.
Substitution modules are very useful and allows us to write down more complex
and expressive search spaces. We have defined a few relatively useful
substitution modules in deep_architect/modules.
Similar to the basic module definition that we looked above, it is more convenient
to deal with the dictionaries of inputs and the dictionaries of outputs than
directly with the modules, so we define this function

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


We will now look at two specific examples of substitution modules. First a
very simple one that the reader will use widely and another one how often
it is useful when implementing more complex search spaces from the literature.
One of the simplest but also most useful substitution modules is the or
substiution module (we often just use the version with a single input and a single
output).

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


We see how short the implementation is. This module has a single hyperparameter
that determines the choice between which function in the function list (or dictionary)
to call. Each of the functions in the function list returns a dictionary of
inputs and a dictionary of outputs when called.
An example search space using subsitution modules, among others, can be found in
deep_architect/misc/.

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


Optional is a special case of a substitution module. If the hyperparameter is
such that the function is to be used, then the function
(in the example above, a lambda function) is called. Otherwise, an identity
modules that passes the input unchanged to the output is used.
Another aspect that is clear from the example above is that substitution modules
are modules, so they can be used in any place that a module can be used.
This makes the language to write search spaces very compositional.
# careful here.

Let us now look at a more complex use of a custom substitution module.

.. code:: python


    def motif(submotif_fn, num_nodes):
        assert num_nodes >= 1

        def substitution_fn(**dh):
            print dh
            node_id_to_node_ids_used = {i: [i - 1] for i in range(1, num_nodes)}
            for name, v in iteritems(dh):
                if v:
                    d = ut.json_string_to_json_object(name)
                    i = d["node_id"]
                    node_ids_used = node_id_to_node_ids_used[i]
                    j = d["in_node_id"]
                    node_ids_used.append(j)
            for i in range(1, num_nodes):
                node_id_to_node_ids_used[i] = sorted(node_id_to_node_ids_used[i])
            print node_id_to_node_ids_used

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
                    inputs["In"].connect(j_outputs["Out"])
                    outputs_lst.append(outputs)

                # if necessary, concatenate the results going into a node
                if num_edges > 1:
                    c_inputs, c_outputs = combine_with_concat(num_edges)
                    for idx, outputs in enumerate(outputs_lst):
                        c_inputs["In%d" % idx].connect(outputs["Out"])
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
            "Motif", name_to_hyperp, substitution_fn, ["In"], ["Out"], scope=None)
This substitution module implements the notion of a motif defined in the
paper (TODO: point to hierarhical paper).
The main goal of this substitution module is to delay the creation of the
motif structure until the values for values for the hyperparameters of the
connections in the motif are determined. The notion of the motif defined in the
paper is recursive. We see that the motif function takes a submotif function
that allows us to place submotifs in each of the edges that are included in the
top-level motif. If the reader wishes to read in more detail about these
search spaces in the literature, we point the reader to the tutorial
search spaces in the literature (TODO).

This concludes our discussion about how to implement new modules in a specific
framework that the reader is working with. We point the reader to the
new_frameworks tutorial for learning about how to support a new framework
by specializing the module class and to the search space constructs tutorials
for a more in-depth coverage of how search spaces can be created by
interconnecting modules.
