from collections import OrderedDict
from six import iterkeys, itervalues, iteritems

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
        for x in iterkeys(self.d):
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

        The name has to be unique, otherwise asserts to False.

        Args:
            name (str): Unique name.
            elem (Addressable): Addressable object to register.
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
            str: Unique name in the current scope object.
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
            elem (Addressable): Addressable object registerd in the scope.

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
        """Replaces the old scope with a new empty scope."""
        Scope.default_scope = Scope()

Scope.default_scope = Scope()

class Addressable:
    """Base class for classes whose objects have to be registered in a scope.

    Provides functionality to register objects in a scope object.

    Args:
        scope (Scope): Scope object where the addressable object will be registered.
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
    Hyperparameters are associated to modules.

    .. note::
        Hyperparameters with easily serializable values are preferred due to the
        interaction with search logging and the multi-GPU functionality.
        Typical types are integers, floats, strings and lists of these types.

    Args:
        scope (Scope, optional): Scope object in which the hyperparameter will be
            registered. If none is given, uses the default scope object.
        name (str, optional): Name used to derived an unique name for the
            hyperparameter. If none is given, uses the class name to derive
            the name.
    """
    def __init__(self, scope=None, name=None):
        scope = scope if scope is not None else Scope.default_scope
        name = scope.get_unused_name('.'.join(
            ['H', (name if name is not None else self._get_base_name()) + '-']))
        Addressable.__init__(self, scope, name)

        self.set_done = False
        self.modules = OrderedSet()

        self.val = None

    def is_set(self):
        """Checks if the hyperparameter has been assigned a value.

        Returns:
            bool: ``True`` if the hyperparameter has been assigned a value.
        """
        return self.set_done

    def set_val(self, val):
        """Assigns a value to the hyperparameter.

        The hyperparameter value must be valid for the hyperparameter in question.
        The hyperaparameter becomes set after if the set operation is successful.

        Args:
            val (object): Value to assign to the hyperparameter.
        """
        assert not self.set_done
        self._check_val(val)
        self.set_done = True
        self.val = val

        # calls update on the dependent modules to signal that this hyperparameter
        # has been set.
        for m in self.modules:
            m._update()

    def get_val(self):
        """Get the value assigned to the hyperparameter.

        The hyperparameter must already be assigned a value, otherwise it
        asserts to ``False``.

        Returns:
            object: Value assigned to the hyperparameter.
        """
        assert self.set_done
        return self.val

    def _register_module(self, module):
        """Registers a module as being dependent of this hyperparameter.

        Args:
            module (Module): Module dependent of this hyperparameter.
        """
        self.modules.add(module)

    def _check_val(self, val):
        """Checks if the value is valid for the hyperparameter.

        When ``set_val`` is called, this function is called to verify the
        validity of ``val``. This function is useful for error checking.
        """
        raise NotImplementedError

class Input(Addressable):
    """Manages input connections.

    Inputs may be connected to a single output. Inputs and outputs are associated
    to a single module.

    See also: :class:`Output` and :class`Module`.

    Args:
        module (Module): Module with which the input object is associated to.
        scope (Scope): Scope object where the input is going to be registered in.
        name (str): Unique name with which to register the input object.
    """
    def __init__(self, module, scope, name):
        name = '.'.join([module.get_name(), 'I', name])
        Addressable.__init__(self, scope, name)

        self.module = module
        self.from_output = None

    def is_connected(self):
        """Checks if the input is connected.

        Returns:
            bool: ``True`` if the input is connected.
        """
        return self.from_output is not None

    def get_connected_output(self):
        """Get the output to which is the input is connected.

        Returns:
            Output: Output object to which the input is connected.
        """
        return self.from_output

    def get_module(self):
        """Get the module object with which the input is associated with.

        Returns:
            Module: Module object with which the input is associated with.
        """
        return self.module

    def connect(self, from_output):
        """Connect an output to this input.

        Changes the state of both the input and the output. Assert ``False`` if
        the input is already connected.

        Args:
            from_output (Output): Output to connect to this input.
        """
        assert isinstance(from_output, Output)
        assert self.from_output is None
        self.from_output = from_output
        from_output.to_inputs.append(self)

    def disconnect(self):
        """Disconnects the input from the output it is connected to.

        Changes the state of both the input and the output. Assert ``False`` if
        the input is not connected.
        """
        assert self.from_output is not None
        self.from_output.to_inputs.remove(self)
        self.from_output = None

    def reroute_connected_output(self, to_input):
        """Disconnects the input from the output it is connected to and connects
        the output to a new input, leaving this input in a disconnected state.

        Changes the state of both this input, the other input, and the output
        to which this input is connected.

        Args:
            to_input (Input): Input to which the output that is connected to
                this input is going to be connected to.
        """
        assert isinstance(to_input, Input)
        old_ox = self.from_output
        self.disconnect()
        old_ox.connect(to_input)

### TODO: more documentation needs to be added from here.
class Output(Addressable):
    """Manages output connections.

    Outputs may be connected to multiple inputs. Inputs and outputs are associated
    to a single module.

    See also: :class:`Input` and :class`Module`.

    Args:
        module (Module): Module with which the output object is associated to.
        scope (Scope): Scope object where the output is going to be registered in.
        name (str): Unique name with which to register the output object.
    """
    def __init__(self, module, scope, name):
        name = '.'.join([module.get_name(), 'O', name])
        Addressable.__init__(self, scope, name)

        self.module = module
        self.to_inputs = []

    def is_connected(self):
        return len(self.to_inputs) > 0

    def get_connected_inputs(self):
        return self.to_inputs

    def get_module(self):
        return self.module

    def connect(self, to_input):
        """Connects an input to this output.

        :param to_input: Input to which this output goes to.
        :type to_input: Input
        """
        to_input.connect(self)

    def disconnect_all(self):
        for ix in self.to_inputs:
            ix.disconnect()

    # NOTE: the inputs that self connects to are rerouted to a different output.
    def reroute_all_connected_inputs(self, from_output):
        """Reroutes all the current inputs to which this output outputs to a different output.

        :param from_output: New output for all the currently connected inputs.
        :type from_output: Output
        """
        for ix in self.to_inputs:
            ix.disconnect()
            ix.connect(from_output)

class Module(Addressable):
    """A module has inputs, outputs, and hyperparameters.
    """
    def __init__(self, scope=None, name=None):
        """
        :type scope: Scope or None
        :type name: str or None
        """
        scope = scope if scope is not None else Scope.default_scope
        name = scope.get_unused_name('.'.join(
            ['M', (name if name is not None else self._get_base_name()) + '-']))
        Addressable.__init__(self, scope, name)

        self.inputs = OrderedDict()
        self.outputs = OrderedDict()
        self.hyperps = OrderedDict()
        self._is_compiled = False

    def _register_input(self, name):
        """Creates a new input with the chosen local name.

        :type name: str
        """
        assert name not in self.inputs
        self.inputs[name] = Input(self, self.scope, name)

    def _register_output(self, name):
        """Creates a new output with the chosen local name.

        :type name: str
        """
        assert name not in self.outputs
        self.outputs[name] = Output(self, self.scope, name)

    def _register_hyperparameter(self, h, name):
        """Registers an hyperparameter that the module depends on it.
        :type name: str
        :type h: Hyperparameter
        """
        assert isinstance(h, Hyperparameter) and name not in self.hyperps
        self.hyperps[name] = h
        h._register_module(self)

    def _register(self, input_names, output_names, name_to_hyperp):
        """
        :type input_names: __iter__
        :type name_to_hyperp: dict[str, Hyperparameter]
        """
        for name in input_names:
            self._register_input(name)
        for name in output_names:
            self._register_output(name)
        for name, h in iteritems(name_to_hyperp):
            self._register_hyperparameter(h, name)

    def _get_input_values(self):
        """
        :rtype: dict[str,Any]
        """
        return {name: ix.val for name, ix in iteritems(self.inputs)}

    def _get_hyperp_values(self):
        """
        :rtype: dict[str,Any]
        """
        return {name: h.get_val() for name, h in iteritems(self.hyperps)}

    def _set_output_values(self, output_name_to_val):
        """
        :type output_name_to_val: dict[str,Any]
        """
        for name, val in iteritems(output_name_to_val):
            self.outputs[name].val = val

    def get_io(self):
        """
        :rtype: (dict[str,Input], dict[str,Output])
        """
        return self.inputs, self.outputs

    def get_hyperps(self):
        """
        :rtype: dict[str,Hyperparameter]
        """
        return self.hyperps

    def _update(self):
        """Called when an hyperparameter that the module depends on is set."""
        raise NotImplementedError

    def _compile(self):
        raise NotImplementedError

    def _forward(self):
        raise NotImplementedError

    def forward(self):
        """Computation done by the module after being specified.

        The first time forward is called, _compile is called to instantiate
        any parameters. These functions change the state.
        """
        if not self._is_compiled:
            self._compile()
            self._is_compiled = True
        self._forward()


def extract_unique_modules(input_or_output_lst):
    """
    :param input_or_output_lst: List of inputs or outputs, from which to extract modules.
    :type input_or_output_lst: collections.Iterable of Input or collections.Iterable of Output
    :return: Unique modules in the input given.
    :rtype: list of Module
    """
    ms = OrderedSet()
    for x in input_or_output_lst:
        assert isinstance(x, (Input, Output))
        ms.add(x.get_module())
    return list(ms)

# assumes that the inputs provided are sufficient to evaluate all the network.
def determine_module_eval_seq(input_lst):
    """Computes the module forward evaluation sequence necessary to evaluate
    the computation graph starting from the provided inputs.

    In dynamic frameworks where forward is called multiple times, it is best
    to precompute the module evaluation sequence.

    :param input_lst: List of inputs.
    :type input_lst: collections.Iterable of Input
    :rtype: list of Module
    :return: Sequence in which to evaluate the modules.

    .. seealso:: :func:`forward`
    """
    module_seq = []
    module_memo = set()
    input_memo = set(input_lst)
    ms = extract_unique_modules(input_lst)
    for m in ms:
        if m not in module_memo and all(ix in input_memo for ix in itervalues(m.inputs)):
            module_seq.append(m)
            module_memo.add(m)

            for ox in itervalues(m.outputs):
                ix_lst = ox.get_connected_inputs()
                input_memo.update(ix_lst)
                m_lst = [ix.get_module() for ix in ix_lst]
                ms.extend(m_lst)
    return module_seq

def traverse_backward(output_lst, fn):
    """Traverses the graph going backward, from outputs to inputs. The
    provided function is applied once to each module reached this way.

    :param output_lst: Outputs to start from.
    :type output_lst: collections.Iterable of Output
    :param fn: Function to apply to modules.
    :type fn: (Module) -> bool

    .. seealso:: :func:`traverse_forward`
    """
    memo = set()
    ms = extract_unique_modules(output_lst)
    for m in ms:
        is_over = fn(m)
        if is_over:
            break
        else:
            for ix in itervalues(m.inputs):
                if ix.is_connected():
                    m_prev = ix.get_connected_output().get_module()
                    if m_prev not in memo:
                        memo.add(m_prev)
                        ms.append(m_prev)

def traverse_forward(input_lst, fn):
    """Traverses the graph going forward, from inputs to outputs. The
    provided function is applied once to each module reached this way.

    :param input_lst: Inputs to start from.
    :type input_lst: collections.Iterable of Input
    :param fn: Function to apply to modules.
    :type fn: (Module) -> bool

    .. seealso:: :func:`traverse_backward`
    """
    memo = set()
    ms = extract_unique_modules(input_lst)
    for m in ms:
        is_over = fn(m)
        if is_over:
            break
        else:
            for ox in itervalues(m.outputs):
                if ox.is_connected():
                    for ix in ox.get_connected_inputs():
                        m_next = ix.get_module()
                        if m_next not in memo:
                            memo.add(m_next)
                            ms.append(m_next)

def is_specified(output_lst):
    """Checks if all the hyperparameters reachable by traversing backward from
    the outputs have been set.

    :type output_lst: collections.Iterable of Output
    :param output_lst: List of outputs to start from.
    :return: True if all hyperparameters have been specified, False otherwise.
    :rtype: bool
    """
    is_spec = [True]

    def fn(module):
        for h in itervalues(module.hyperps):
            if not h.is_set():
                is_spec[0] = False
                return True
        return False
    traverse_backward(output_lst, fn)
    return is_spec[0]

def get_unset_hyperparameters(output_lst):
    """Gets all the hyperparameters that are not set yet.

    Setting an hyperparameter may lead to the creation of more hyperparameters,
    e.g., optional or repeat.

    :param output_lst: List of outputs to start from.
    :type output_lst: collections.Iterable of Output
    :return: Hyperparameters which have not been set.
    :rtype: OrderedSet of Hyperparameter
    """
    assert not is_specified(output_lst)
    hs = OrderedSet()

    def fn(module):
        for h in itervalues(module.hyperps):
            if not h.is_set():
                hs.add(h)
        return False
    traverse_backward(output_lst, fn)
    return hs

def forward(input_to_val, _module_seq=None):
    """Forward pass starting from the inputs.

    The forward computation of each module is called in the turn. For efficiency,
    in dynamic frameworks, the module evaluation sequence is best computed once
    and reused in each forward call.

    :param input_to_val: Dictionary of inputs to corresponding values.
    :type input_to_val: dict[Input,Any]
    :param _module_seq: collections.Iterable of Module
    """
    if _module_seq is None:
        _module_seq = determine_module_eval_seq(input_to_val.keys())

    for ix, val in iteritems(input_to_val):
        ix.val = val

    for m in _module_seq:
        m.forward()
        for ox in itervalues(m.outputs):
            for ix in ox.get_connected_inputs():
                ix.val = ox.val

def get_unconnected_inputs(output_lst):
    """
    :type output_lst: collections.Iterable of Output
    :return: Inputs which are not connected.
    :rtype: list of Input
    """
    ix_lst = []

    def fn(x):
        for ix in itervalues(x.inputs):
            if not ix.is_connected():
                ix_lst.append(ix)
        return False
    traverse_backward(output_lst, fn)
    return ix_lst

def get_unconnected_outputs(input_lst):
    """
    :type input_lst: collections.Iterable of Input
    :return: Outputs which are not connected.
    :rtype: list of Output
    """
    ox_lst = []

    def fn(x):
        for ox in itervalues(x.outputs):
            if not ox.is_connected():
                ox_lst.append(ox)
        return False
    traverse_forward(input_lst, fn)
    return ox_lst
