from collections import OrderedDict


def sorted_values_by_key(d):
    return [d[k] for k in sorted(d)]


class OrderedSet:

    def __init__(self):
        self.d = OrderedDict()

    def add(self, x):
        if x not in self.d:
            self.d[x] = None

    def update(self, xs):
        for x in xs:
            if x not in self.d:
                self.d[x] = None

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        for x in self.d.keys():
            yield x

    def __contains__(self, item):
        return item in self.d


class Scope:
    """A scope is used to help assign unique readable names to addressable objects.

    A scope keeps references to modules, hyperparameters, inputs, and outputs.
    """

    def __init__(self):
        self.name_to_elem = OrderedDict()
        self.elem_to_name = OrderedDict()

    def register(self, name, elem):
        """Registers an addressable object with the desired name.

        The name cannot exist in the scope, otherwise asserts ``False``.

        Args:
            name (str): Unique name.
            elem (deep_architect.core.Addressable): Addressable object to register.
        """
        assert name not in self.name_to_elem
        assert isinstance(elem, Addressable)
        self.name_to_elem[name] = elem
        self.elem_to_name[elem] = name

    def get_unused_name(self, prefix):
        """Creates a unique name by adding a numbered suffix to the prefix.

        Args:
            prefix (str): Prefix of the desired name.

        Returns:
            str: Unique name in the current scope.
        """
        i = 0
        while True:
            name = prefix + str(i)
            if name not in self.name_to_elem:
                break
            i += 1
        return name

    def get_name(self, elem):
        """Get the name of the addressable object registered in the scope.

        The object must exist in the scope.

        Args:
            elem (deep_architect.core.Addressable): Addressable object
                registered in the scope.

        Returns:
            str: Name with which the object was registered in the scope.
        """
        return self.elem_to_name[elem]

    def get_elem(self, name):
        """Get the object that is registered in the scope with the desired name.

        The name must exist in the scope.

        Args:
            name (str): Name of the addressable object registered in the scope.

        Returns:
            str: Addressable object with the corresponding name.
        """
        return self.name_to_elem[name]

    @staticmethod
    def reset_default_scope():
        """Replaces the current default scope with a new empty scope."""
        Scope.default_scope = Scope()


# TODO: is this called once for each time core is imported? check.
Scope.default_scope = Scope()


class Addressable:
    """Base class for classes whose objects have to be registered in a scope.

    Provides functionality to register objects in a scope.

    Args:
        scope (deep_architect.core.Scope): Scope object where the addressable
            object will be registered.
        name (str): Unique name used to register the addressable object.
    """

    def __init__(self, scope, name):
        scope.register(name, self)
        self.scope = scope

    def __repr__(self):
        return self.get_name()

    def get_name(self):
        """Get the name with which the object was registered in the scope.

        Returns:
            str: Unique name used to register the object.
        """
        return self.scope.get_name(self)

    def _get_base_name(self):
        """Get the class name.

        Useful to create unique names for an addressable object.

        Returns:
            str: Class name.
        """
        return self.__class__.__name__


class Hyperparameter(Addressable):
    """Base hyperparameter class.

    Specific hyperparameter types are created by inheriting from this class.
    Hyperparameters keep references to the modules that are dependent on them.

    .. note::
        Hyperparameters with easily serializable values are preferred due to the
        interaction with the search logging and multi-GPU functionalities.
        Typical valid serializable types are integers, floats, strings. Lists
        and dictionaries of serializable types are also valid.

    Args:
        scope (deep_architect.core.Scope, optional): Scope in which the hyperparameter
            will be registered. If none is given, uses the default scope.
        name (str, optional): Name used to derive an unique name for the
            hyperparameter. If none is given, uses the class name to derive
            the name.
    """

    def __init__(self, scope=None, name=None):
        scope = scope if scope is not None else Scope.default_scope
        name = scope.get_unused_name('.'.join(
            ['H', (name if name is not None else self._get_base_name()) + '-']))
        super().__init__(scope, name)

        self.assign_done = False
        self.modules = OrderedSet()
        self.dependent_hyperps = OrderedSet()

        self.val = None

    def has_value_assigned(self):
        """Checks if the hyperparameter has been assigned a value.

        Returns:
            bool: ``True`` if the hyperparameter has been assigned a value.
        """
        return self.assign_done

    def assign_value(self, val):
        """Assigns a value to the hyperparameter.

        The hyperparameter value must be valid for the hyperparameter in question.
        The hyperparameter becomes set if the call is successful.

        Args:
            val (object): Value to assign to the hyperparameter.
        """
        assert not self.assign_done
        self._check_value(val)
        self.assign_done = True
        self.val = val

        # calls update on the dependent modules to signal that this hyperparameter
        # has been set, and trigger any relevant local changes.
        for m in self.modules:
            m._update()

        # calls updates on the dependent hyperparameters.
        for h in self.dependent_hyperps:
            if not h.assign_done:
                h._update()

    def get_value(self):
        """Get the value assigned to the hyperparameter.

        The hyperparameter must have already been assigned a value, otherwise
        asserts ``False``.

        Returns:
            object: Value assigned to the hyperparameter.
        """
        assert self.assign_done
        return self.val

    def _register_module(self, module):
        """Registers a module as being dependent of this hyperparameter.

        Args:
            module (deep_architect.core.Module): Module dependent of this hyperparameter.
        """
        self.modules.add(module)

    def _register_dependent_hyperparameter(self, hyperp):
        """Registers an hyperparameter as being dependent on this hyperparameter.

        Args:
            module (deep_architect.core.Hyperparameter): Hyperparameter dependent of this hyperparameter.
        """
        # NOTE: for now, it is odd to register the same hyperparameter multiple times.
        assert hyperp not in self.dependent_hyperps
        assert isinstance(hyperp, DependentHyperparameter)
        self.dependent_hyperps.add(hyperp)

    def _check_value(self, val):
        """Checks if the value is valid for the hyperparameter.

        When ``set_val`` is called, this function is called to verify the
        validity of ``val``. This function is useful for error checking.
        """
        # raise NotImplementedError
        pass


class DependentHyperparameter(Hyperparameter):
    """Hyperparameter that depends on other hyperparameters.

    The value of a dependent hyperparameter is set by a calling a function
    using the values of the dependent hyperparameters as arguments.
    This hyperparameter is convenient when we want to express search spaces where
    the values of some hyperparameters are computed as a function of the
    values of some other hyperparameters, rather than set independently.

    Args:
        fn (dict[str, object] -> object): Function used to compute the value of the
            hyperparameter based on the values of the dependent hyperparameters.
        hyperps (dict[str, deep_architect.core.Hyperparameter]): Dictionary mapping
            names to hyperparameters. The names used in the dictionary should
            correspond to the names of the arguments of ``fn``.
        scope (deep_architect.core.Scope, optional): The scope in which to register the
            hyperparameter in.
        name (str, optional): Name from which the name of the hyperparameter
            in the scope is derived.
    """

    def __init__(self, fn, hyperps, scope=None, name=None):
        super().__init__(scope, name)
        # NOTE: this assert may or may not be necessary.
        # assert isinstance(hyperps, OrderedDict)
        self._hyperps = OrderedDict([(k, hyperps[k]) for k in sorted(hyperps)])
        self._fn = fn

        # registering the dependencies.
        for h in self._hyperps.values():
            h._register_dependent_hyperparameter(self)

        self._update()

    def _update(self):
        """Checks if the hyperparameter is ready to be set, and sets it if that
        is the case.
        """
        # assert not self.has_value_assigned()
        if all(h.has_value_assigned() for h in self._hyperps.values()):
            dh = {name: h.get_value() for name, h in self._hyperps.items()}
            v = self._fn(dh)
            self.assign_value(v)

    def _check_value(self, val):
        pass


class Input(Addressable):
    """Manages input connections.

    Inputs may be connected to a single output. Inputs and outputs are associated
    to a single module.

    See also: :class:`deep_architect.core.Output` and :class:`deep_architect.core.Module`.

    Args:
        module (deep_architect.core.Module): Module with which the input object
            is associated to.
        scope (deep_architect.core.Scope): Scope object where the input is
            going to be registered in.
        name (str): Unique name with which to register the input object.
    """

    def __init__(self, module, scope, name):
        name = '.'.join([module.get_name(), 'I', name])
        super().__init__(scope, name)

        self.module = module
        self.from_output = None

    def is_connected(self):
        """Checks if the input is connected.

        Returns:
            bool: ``True`` if the input is connected.
        """
        return self.from_output is not None

    def get_connected_output(self):
        """Get the output to which the input is connected to.

        Returns:
            deep_architect.core.Output: Output to which the input is connected to.
        """
        return self.from_output

    def get_module(self):
        """Get the module with which the input is associated with.

        Returns:
            deep_architect.core.Module: Module with which the input is associated with.
        """
        return self.module

    def connect(self, from_output):
        """Connect an output to this input.

        Changes the state of both the input and the output. Asserts ``False`` if
        the input is already connected.

        Args:
            from_output (deep_architect.core.Output): Output to connect to this input.
        """
        assert isinstance(from_output, Output)
        assert self.from_output is None
        self.from_output = from_output
        from_output.to_inputs.append(self)

    def disconnect(self):
        """Disconnects the input from the output it is connected to.

        Changes the state of both the input and the output. Asserts ``False`` if
        the input is not connected.
        """
        assert self.from_output is not None
        self.from_output.to_inputs.remove(self)
        self.from_output = None

    def reroute_connected_output(self, to_input):
        """Disconnects the input from the output it is connected to and connects
        the output to a new input, leaving this input in a disconnected state.

        Changes the state of both this input, the other input, and the output
        to which this input is connected to.

        .. note::
            Rerouting operations are widely used in
            :class:`deep_architect.modules.SubstitutionModule`. See also:
            :meth:`deep_architect.core.Output.reroute_all_connected_inputs`.

        Args:
            to_input (deep_architect.core.Input): Input to which the output
                is going to be connected to.
        """
        assert isinstance(to_input, Input)
        old_ox = self.from_output
        self.disconnect()
        old_ox.connect(to_input)


class Output(Addressable):
    """Manages output connections.

    Outputs may be connected to multiple inputs. Inputs and outputs are associated
    to a single module.

    See also: :class:`deep_architect.core.Input` and :class:`deep_architect.core.Module`.

    Args:
        module (deep_architect.core.Module): Module with which the output object
            is associated to.
        scope (deep_architect.core.Scope): Scope object where the output is
            going to be registered in.
        name (str): Unique name with which to register the output object.
    """

    def __init__(self, module, scope, name):
        name = '.'.join([module.get_name(), 'O', name])
        super().__init__(scope, name)

        self.module = module
        self.to_inputs = []

    def is_connected(self):
        """Checks if the output is connected.

        Returns:
            bool: ``True`` if the output is connected.
        """
        return len(self.to_inputs) > 0

    def get_connected_inputs(self):
        """Get the list of inputs to which is the output is connected to.

        Returns:
            list[deep_architect.core.Input]: List of the inputs to which the
                output is connect to.
        """
        return self.to_inputs

    def get_module(self):
        """Get the module object with which the output is associated with.

        Returns:
            deep_architect.core.Module: Module object with which the output is
                associated with.
        """
        return self.module

    def connect(self, to_input):
        """Connect an additional input to this output.

        Changes the state of both the input and the output.

        Args:
            to_input (deep_architect.core.Input): Input to connect to this output.
        """
        to_input.connect(self)

    def disconnect_all(self):
        """Disconnects all the inputs connected to this output.

        Changes the state of the output and all the inputs connected to it.
        """
        to_inputs = list(self.to_inputs)
        for ix in to_inputs:
            ix.disconnect()

    def reroute_all_connected_inputs(self, from_output):
        """Reroutes all the inputs to which the output is connected to a
        different output.

        .. note::
            Rerouting operations are widely used in
            :class:`deep_architect.modules.SubstitutionModule`. See also:
            :meth:`deep_architect.core.Input.reroute_connected_output`.

        Args:
            from_output (deep_architect.core.Output): Output to which the
                connected inputs are going to be rerouted to.
        """
        to_inputs = list(self.to_inputs)
        for ix in to_inputs:
            ix.disconnect()
            ix.connect(from_output)


class Module(Addressable):
    """Modules inputs and outputs, and depend on hyperparameters.

    Modules are some of the main components used to define search spaces.
    The inputs, outputs, and hyperparameters have names local to the module.
    These names are different than the ones used in the scope in which
    these objects are registered in.

    Search spaces based on modules are very general. They can be used
    across deep learning frameworks, and even for purposes that do not involve
    deep learning, e.g., searching over scikit-learn pipelines. The main
    operations to understand are compile and forward.

    Args:
        input_names (list[str]): List of inputs names of the module.
        output_names (list[str]): List of the output names of the module.
        name_to_hyperp (dict[str, deep_architect.core.Hyperparameter]):
            Dictionary of names of hyperparameters to hyperparameters.
        scope (deep_architect.core.Scope, optional): Scope object where the
            module is going to be registered in.
        name (str, optional): Unique name with which to register the module.
    """

    def __init__(self,
                 input_names,
                 output_names,
                 name_to_hyperp,
                 scope=None,
                 name=None):
        scope = scope if scope is not None else Scope.default_scope
        name = scope.get_unused_name('.'.join(
            ['M', (name if name is not None else self._get_base_name()) + '-']))
        super().__init__(scope, name)

        self.inputs = OrderedDict()
        self.outputs = OrderedDict()
        self.hyperps = OrderedDict()
        self._register(input_names, output_names, name_to_hyperp)

    def _register_input(self, name):
        """Creates a new input with the chosen local name.

        Args:
            name (str): Local name given to the input.
        """
        if name in self.inputs:
            raise ValueError("Input %s already exists in module %s" %
                             (name, self.get_name()))
        else:
            self.inputs[name] = Input(self, self.scope, name)

    def _register_output(self, name):
        """Creates a new output with the chosen local name.

        Args:
            name (str): Local name given to the output.
        """
        if name in self.outputs:
            raise ValueError("Output %s already exists in module %s" %
                             (name, self.get_name()))
        else:
            self.outputs[name] = Output(self, self.scope, name)

    def _register_hyperparameter(self, name, h):
        """Registers an hyperparameter that the module depends on.

        Args:
            name (str): Local name to give to the hyperparameter.
            h (deep_architect.core.Hyperparameter): Hyperparameter that the
                module depends on.
        """
        if name in self.outputs:
            raise ValueError("Hyperparameter %s already exists in module %s" %
                             (name, self.get_name()))
        else:
            if not isinstance(h, Hyperparameter):
                wrapped_h = Hyperparameter(self.scope, "Singleton")
                wrapped_h.assign_value(h)
                h = wrapped_h
            self.hyperps[name] = h
        h._register_module(self)

    def _register(self, input_names, output_names, name_to_hyperp):
        """Registers inputs, outputs, and hyperparameters locally for the module.

        This function is convenient to avoid code repetition when registering
        multiple inputs, outputs, and hyperparameters.

        Args:
            input_names (list[str]): List of inputs names of the module.
            output_names (list[str]): List of the output names of the module.
            name_to_hyperp (dict[str, deep_architect.core.Hyperparameter]):
                Dictionary of names of hyperparameters to hyperparameters.
        """
        for name in input_names:
            self._register_input(name)
        for name in output_names:
            self._register_output(name)
        for name in sorted(name_to_hyperp):
            self._register_hyperparameter(name, name_to_hyperp[name])

    def _get_input_values(self):
        """Get the values associated to the inputs of the module.

        This function is used to implement forward. See also:
        :meth:`_set_output_values` and :func:`forward`.

        Returns:
            dict[str, object]: Dictionary of local input names to their corresponding values.
        """
        return {name: ix.val for name, ix in self.inputs.items()}

    def _get_hyperp_values(self):
        """Get the values of the hyperparameters.

        Returns:
            dict[str, object]:
                Dictionary of local hyperparameter names to their corresponding values.
        """
        return {name: h.get_value() for name, h in self.hyperps.items()}

    def _set_output_values(self, output_name_to_val):
        """Set the values of the outputs of the module.

        This function is used to implement forward. See also:
        :meth:`_get_input_values` and :func:`forward`.

        Args:
            output_name_to_val (dict[str, object]): Dictionary of local output
                names to the corresponding values to assign to those outputs.
        """
        for name, val in output_name_to_val.items():
            self.outputs[name].val = val

    def get_io(self):
        """
        Returns:
            (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output]):
                Pair with dictionaries mapping
                the local input and output names to their corresponding
                input and output objects.
        """
        return self.inputs, self.outputs

    def get_hyperps(self):
        """
        Returns:
            dict[str, deep_architect.core.Hyperparameter]:
                Dictionary of local hyperparameter names to the corresponding
                hyperparameter objects.
        """
        return self.hyperps

    def _update(self):
        """Called when an hyperparameter that the module depends on is set."""
        pass


class SubstitutionModule(Module):
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
        input_names (list[str]): List of the input names of the substitution module.
        output_name (list[str]): List of the output names of the substitution module.
        name_to_hyperp (dict[str, deep_architect.core.Hyperparameter]): Dictionary of
            name to hyperparameters that are needed for the substitution function.
            The names of the hyperparameters should be in correspondence to the
            name of the arguments of the substitution function.
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
                 input_names,
                 output_names,
                 name_to_hyperp,
                 allow_input_subset=False,
                 allow_output_subset=False,
                 scope=None,
                 name=None):
        super().__init__(input_names, output_names, name_to_hyperp, scope, name)

        self.allow_input_subset = allow_input_subset
        self.allow_output_subset = allow_output_subset
        self.substitution_done = False
        self._update()

    def _update(self):
        """Implements the substitution operation.

        When all the hyperparameters that the module depends on are specified,
        the substitution operation is triggered, and the substitution operation
        is done.
        """
        if (not self.substitution_done) and all(
                h.has_value_assigned() for h in self.hyperps.values()):
            new_inputs, new_outputs = self.substitute()

            # test for checking that the inputs and outputs returned by the
            # substitution function are valid.
            if self.allow_input_subset:
                assert len(new_inputs) <= len(self.inputs) and all(
                    name in self.inputs for name in new_inputs)
            else:
                assert len(self.inputs) == len(new_inputs) and all(
                    name in self.inputs for name in new_inputs)

            if self.allow_output_subset:
                assert len(new_outputs) <= len(self.outputs) and all(
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

            self.substitution_done = True

    def substitute(self):
        """This function needs to be implemented when concrete instances are
        created through subclassing.
        """
        raise NotImplementedError


def extract_unique_modules(input_or_output_lst):
    """Get the modules associated to the inputs and outputs in the list.

    Each module appears appear only once in the resulting list of modules.

    Args:
        input_or_output_lst (list[deep_architect.core.Input or deep_architect.core.Output]):
            List of inputs or outputs from which to extract the associated modules.

    Returns:
        list[deep_architect.core.Module]:
            Unique modules to which the inputs and outputs in the list belong to.
    """
    ms = OrderedSet()
    for x in input_or_output_lst:
        assert isinstance(x, (Input, Output))
        ms.add(x.get_module())
    return list(ms)


# assumes that the inputs provided are sufficient to evaluate all the network.
# TODO: add the more general functionality that allows us to compute the sequence
# of forward operations for a subgraph of the full computational graph.
def determine_module_eval_seq(inputs):
    """Computes the module forward evaluation sequence necessary to evaluate
    the computational graph starting from the provided inputs.

    The computational graph is a directed acyclic graph. This function sorts
    the modules topologically based on their dependencies. It is assumed that
    the inputs in the dictionary provided are sufficient to compute forward for all
    modules in the graph. See also: :func:`forward`.

    Args:
        inputs (dict[str, deep_architect.core.Input]): dictionary of inputs sufficient
            to compute the forward computation of the whole graph through propagation.

    Returns:
        list[deep_architect.core.Module]:
            List of modules ordered in a way that allows to call forward on the
            modules in that order.
    """
    module_seq = []
    module_memo = set()
    input_memo = set(inputs.values())
    ms = extract_unique_modules(list(inputs.values()))
    for m in ms:
        if m not in module_memo and all(
                ix in input_memo for ix in m.inputs.values()):
            module_seq.append(m)
            module_memo.add(m)

            for ox in m.outputs.values():
                ix_lst = ox.get_connected_inputs()
                input_memo.update(ix_lst)
                m_lst = [ix.get_module() for ix in ix_lst]
                ms.extend(m_lst)
    return module_seq


def determine_input_output_cleanup_seq(inputs):
    """Determines the order in which the outputs can be cleaned.

    This sequence is aligned with the module evaluation sequence. Positionally,
    after each module evaluation, the values stored in val for both inputs and
    outputs can be deleted. This is useful to remove intermediate results to
    save memory.

    .. note::
        This function should be used only for fully-specified search spaces.

    Args:
        inputs (dict[str, deep_architect.core.Input]): Dictionary of named
            inputs which by being traversed forward will reach all the
            modules in the search space.

    Returns:
        (list[list[deep_architect.core.Input]], list[list[deep_architect.core.Output]]):
            List of lists with the inputs and outputs in the order they should be
            cleaned up after they are no longer needed.
    """
    module_eval_seq = determine_module_eval_seq(inputs)

    input_cleanup_seq = []
    for m in module_eval_seq:
        lst = list(m.inputs.values())
        input_cleanup_seq.append(lst)

    # number of inputs dependent on each input.
    output_counters = {}
    for m in module_eval_seq:
        for ox in m.outputs.values():
            output_counters[ox] = len(ox.get_connected_inputs())

    output_cleanup_seq = []
    for m in module_eval_seq:
        lst = []
        for ix in m.inputs.values():
            if ix.is_connected():
                ox = ix.get_connected_output()
                output_counters[ox] -= 1
                if output_counters[ox] == 0:
                    lst.append(ox)
        output_cleanup_seq.append(lst)

    return input_cleanup_seq, output_cleanup_seq


def traverse_backward(outputs, fn):
    """Backward traversal function through the graph.

    Traverses the graph going from outputs to inputs. The provided function is
    applied once to each module reached this way. This function is used to
    implement other functionality that requires traversing the graph. ``fn``
    typically has side effects, e.g., see :func:`is_specified` and
    :func:`get_unassigned_hyperparameters`. See also: :func:`traverse_forward`.

    Args:
        outputs (dict[str, deep_architect.core.Output]): Dictionary of named of
            outputs to start the traversal at.
        fn ((deep_architect.core.Module) -> (bool)): Function to apply to each
            module. Returns ``True`` if the traversal is to be stopped.
    """
    memo = set()
    output_lst = sorted_values_by_key(outputs)
    ms = extract_unique_modules(output_lst)
    for m in ms:
        is_over = fn(m)
        if is_over:
            break
        else:
            for ix in m.inputs.values():
                if ix.is_connected():
                    m_prev = ix.get_connected_output().get_module()
                    if m_prev not in memo:
                        memo.add(m_prev)
                        ms.append(m_prev)


def traverse_forward(inputs, fn):
    """Forward traversal function through the graph.

    Traverses the graph going from inputs to outputs. The provided function is
    applied once to each module reached this way. This function is used to
    implement other functionality that requires traversing the graph. ``fn``
    typically has side effects, e.g., see :func:`get_unconnected_outputs`.
    See also: :func:`traverse_backward`.

    Args:
        inputs (dict[str, deep_architect.core.Input]): Dictionary of named inputs
            to start the traversal at.
        fn ((deep_architect.core.Module) -> (bool)): Function to apply to each
            module. Returns ``True`` if the traversal is to be stopped.
    """
    memo = set()
    input_lst = sorted_values_by_key(inputs)
    ms = extract_unique_modules(input_lst)
    for m in ms:
        is_over = fn(m)
        if is_over:
            break
        else:
            for ox in m.outputs.values():
                if ox.is_connected():
                    for ix in ox.get_connected_inputs():
                        m_next = ix.get_module()
                        if m_next not in memo:
                            memo.add(m_next)
                            ms.append(m_next)


def get_modules_with_cond(outputs, cond_fn):
    ms = []

    def fn(m):
        if cond_fn(m):
            ms.append(m)

    traverse_backward(outputs, fn)
    return ms


def is_specified(outputs):
    """Checks if all the hyperparameters reachable by traversing backward from
    the outputs have been set.

    Args:
        outputs (dict[str, deep_architect.core.Output]): Dictionary of named
            outputs to start the traversal at.

    Returns:
        bool: ``True`` if all the hyperparameters have been set. ``False`` otherwise.
    """
    is_spec = [True]

    def fn(module):
        for h in module.hyperps.values():
            if not h.has_value_assigned():
                is_spec[0] = False
                return True
        return False

    traverse_backward(outputs, fn)
    return is_spec[0]


def get_unconnected_inputs(outputs):
    """Get the inputs that are reachable going backward from the provided outputs,
    but are not connected to any outputs.

    Often, these inputs have to be provided with a value when calling
    :func:`forward`.

    Args:
        outputs (list[deep_architect.core.Output]): Dictionary of named outputs
            to start the backward traversal at.

    Returns:
        list[deep_architect.core.Input]:
            Unconnected inputs reachable by traversing the graph backward starting
            from the provided outputs.
    """
    ix_lst = []

    def fn(x):
        for ix in x.inputs.values():
            if not ix.is_connected():
                ix_lst.append(ix)
        return False

    traverse_backward(outputs, fn)
    return ix_lst


def get_unconnected_outputs(inputs):
    """Get the outputs that are reachable going forward from the provided inputs,
    but are not connected to outputs.

    Often, the final result of a forward pass through the network will be at
    these outputs.

    Args:
        inputs (dict[str, deep_architect.core.Input]): Dictionary of named
            inputs to start the forward traversal at.

    Returns:
        list[deep_architect.core.Output]:
            Unconnected outputs reachable by traversing the graph forward starting
            from the provided inputs.
    """
    ox_lst = []

    def fn(x):
        for ox in x.outputs.values():
            if not ox.is_connected():
                ox_lst.append(ox)
        return False

    traverse_forward(inputs, fn)
    return ox_lst


def get_all_hyperparameters(outputs):
    """Going backward from the outputs provided, gets all hyperparameters.

    Hyperparameters that can be reached by traversing dependency links between
    hyperparameters are also included. Setting an hyperparameter may lead to the
    creation of additional hyperparameters, which will be most likely not set.
    Such behavior happens when dealing with,
    for example, hyperparameters associated with substitutition
    modules such as :func:`deep_architect.modules.siso_optional`,
    :func:`deep_architect.modules.siso_or`, and :func:`deep_architect.modules.siso_repeat`.

    Args:
        outputs (dict[str, deep_architect.core.Output]): Dictionary of named
            outputs to start the traversal at.

    Returns:
        OrderedSet[deep_architect.core.Hyperparameter]:
            Ordered set of hyperparameters that are currently present in the
            graph.
    """
    visited_hs = OrderedSet()

    def _add_reachable_hs(h_dep):
        assert isinstance(h_dep, DependentHyperparameter)
        local_memo = set([h_dep])
        h_dep_lst = [h_dep]
        idx = 0
        while idx < len(h_dep_lst):
            for h in h_dep_lst[idx]._hyperps.values():
                # cycle detection.
                # assert h not in local_memo

                if h not in visited_hs:
                    if isinstance(h, DependentHyperparameter):
                        h_dep_lst.append(h)
                local_memo.add(h)
                visited_hs.add(h)
            idx += 1

    # this function is applied on each of the modules in the graph.
    def fn(module):
        for h in module.hyperps.values():
            if h not in visited_hs:
                visited_hs.add(h)
                if isinstance(h, DependentHyperparameter):
                    _add_reachable_hs(h)
        return False

    traverse_backward(outputs, fn)
    return visited_hs


def get_unassigned_independent_hyperparameters(outputs):
    """Going backward from the outputs provided, gets all the independent
    hyperparameters that are not set yet.

    Setting an hyperparameter may lead to the creation of additional hyperparameters,
    which will be most likely not set. Such behavior happens when dealing with,
    for example, hyperparameters associated with substitutition
    modules such as :func:`deep_architect.modules.siso_optional`,
    :func:`deep_architect.modules.siso_or`, and :func:`deep_architect.modules.siso_repeat`.

    Args:
        outputs (dict[str, deep_architect.core.Output]): Dictionary of named
            outputs to start the traversal at.

    Returns:
        OrderedSet[deep_architect.core.Hyperparameter]:
            Ordered set of hyperparameters that are currently present in the
            graph and not have been assigned a value yet.
    """
    assert not is_specified(outputs)
    unassigned_indep_hs = OrderedSet()
    for h in get_all_hyperparameters(outputs):
        if not isinstance(
                h, DependentHyperparameter) and not h.has_value_assigned():
            unassigned_indep_hs.add(h)
    return unassigned_indep_hs


# TODO: perhaps change to not have to work until everything is specified.
# this can be done through a flag.


def unassigned_independent_hyperparameter_iterator(outputs):
    """Returns an iterator over the hyperparameters that are not specified in
    the current search space.

    This iterator is used by the searchers to go over the unspecified
    hyperparameters.

    .. note::
        It is assumed that all the hyperparameters that are touched by the
        iterator will be specified (most likely, right away). Otherwise, the
        iterator will never terminate.

    Args:
        outputs (dict[str, deep_architect.core.Output]): Dictionary of named
            outputs which by being traversed back will reach all the
            modules in the search space, and correspondingly all the current
            unspecified hyperparameters of the search space.

    Yields:
        (deep_architect.core.Hyperparameter):
            Next unspecified hyperparameter of the search space.
    """
    while not is_specified(outputs):
        hs = get_unassigned_independent_hyperparameters(outputs)
        for h in hs:
            if not h.has_value_assigned():
                yield h


def jsonify(inputs, outputs):
    """Returns a JSON representation of the fully-specified search space.

    This function is useful to create a representation of model that does not
    rely on the graph representation involving :class:`deep_architect.core.Module`,
    :class:`deep_architect.core.Input`, and :class:`deep_architect.core.Output`.

    Args:
        inputs (dict[str, deep_architect.core.Input]): Dictionary of named
            inputs which by being traversed forward will reach all the
            modules in the search space.
        outputs (dict[str, deep_architect.core.Output]): Dictionary of named
            outputs which by being traversed back will reach all the
            modules in the search space.

    Returns:
        (dict): JSON representation of the fully specified model.
    """
    modules = {}

    def add_module(m):
        module_name = m.get_name()
        input_names = {name: ix.get_name() for name, ix in m.inputs.items()}
        output_names = {name: ox.get_name() for name, ox in m.outputs.items()}
        hyperp_name_to_val = m._get_hyperp_values()
        in_connections = {}
        out_connections = {}
        in_modules = set()
        out_modules = set()
        for ix in m.inputs.values():
            if ix.is_connected():
                ox = ix.get_connected_output()
                ix_name = ix.get_name()
                in_connections[ix_name] = ox.get_name()
                in_module_name = ox.get_module().get_name()
                in_modules.add(in_module_name)

        for ox in m.outputs.values():
            if ox.is_connected():
                ox_name = ox.get_name()
                lst = []
                for ix in ox.get_connected_inputs():
                    lst.append(ix.get_name())
                    out_module_name = ix.get_module().get_name()
                    out_modules.add(out_module_name)
                out_connections[ox_name] = lst

        module_name = m.get_name()
        start_idx = module_name.index('.') + 1
        end_idx = len(module_name) - module_name[::-1].index('-') - 1
        module_type = module_name[start_idx:end_idx]
        modules[m.get_name()] = {
            "module_name": module_name,
            "module_type": module_type,
            "hyperp_name_to_val": hyperp_name_to_val,
            "input_names": input_names,
            "output_names": output_names,
            "in_connections": in_connections,
            "out_connections": out_connections,
            "in_modules": list(in_modules),
            "out_modules": list(out_modules),
        }

    traverse_backward(outputs, add_module)

    ms = determine_module_eval_seq(inputs)
    module_eval_seq = [m.get_name() for m in ms]

    ixs_lst, oxs_lst = determine_input_output_cleanup_seq(inputs)
    input_cleanup_seq = [[ix.get_name() for ix in ixs] for ixs in ixs_lst]
    output_cleanup_seq = [[ox.get_name() for ox in oxs] for oxs in oxs_lst]

    unconnected_inputs = {name: ix.get_name() for name, ix in inputs.items()}
    unconnected_outputs = {name: ox.get_name() for name, ox in outputs.items()}

    graph = {
        "modules": modules,
        "unconnected_inputs": unconnected_inputs,
        "unconnected_outputs": unconnected_outputs,
        "module_eval_seq": module_eval_seq,
        "input_cleanup_seq": input_cleanup_seq,
        "output_cleanup_seq": output_cleanup_seq,
    }
    return graph
