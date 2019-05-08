import deep_architect.core as co
from six import itervalues, iteritems, itertools
from six.moves import range


class Identity(co.Module):
    """Module passes the input to the output without changes.

    Args:
        scope (deep_architect.core.Scope, optional): Scope in which the module will be
            registered. If none is given, uses the default scope.
        name (str, optional): Name used to derive an unique name for the
            module. If none is given, uses the class name to derive
            the name.
    """

    def __init__(self, scope=None, name=None):
        co.Module.__init__(self, scope, name)
        self._register_input("In")
        self._register_output("Out")

    def _compile(self):
        pass

    def _forward(self):
        self.outputs['Out'].val = self.inputs['In'].val


class HyperparameterAggregator(co.Module):

    def __init__(self, name_to_hyperp, scope=None, name=None):
        co.Module.__init__(self, scope, name)
        self._register(["In"], ["Out"], name_to_hyperp)

    def _compile(self):
        pass

    def _forward(self):
        self.outputs['Out'].val = self.inputs['In'].val


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


def identity(scope=None, name=None):
    """Same as the Identity module, but directly works with dictionaries of
    inputs and outputs of the module.

    See :class:`Identity`.

    Returns:
        (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output]):
            Tuple with dictionaries with the inputs and outputs of the module.
    """
    return Identity(scope=scope, name=name).get_io()


def hyperparameter_aggregator(name_to_hyperp, scope=None, name=None):
    return HyperparameterAggregator(name_to_hyperp, scope, name).get_io()


def get_hyperparameter_aggregators(output_lst):
    co.get_modules_with_cond(
        output_lst, lambda m: isinstance(m, HyperparameterAggregator))


def substitution_module(name,
                        name_to_hyperp,
                        substitution_fn,
                        input_names,
                        output_names,
                        scope,
                        allow_input_subset=False,
                        allow_output_subset=False):
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
        allow_input_subset (bool): If true, allows the substitution function to
            return a strict subset of the names of the inputs existing before the
            substitution. Otherwise, the dictionary of inputs returned by the
            substitution function must contain exactly the same input names.
        allow_output_subset (bool): If true, allows the substitution function to
            return a strict subset of the names of the outputs existing before the
            substitution. Otherwise, the dictionary of outputs returned by the
            substitution function must contain exactly the same output names.

    Returns:
        (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output]):
            Tuple with dictionaries with the inputs and outputs of the module.
    """
    return SubstitutionModule(name,
                              name_to_hyperp,
                              substitution_fn,
                              input_names,
                              output_names,
                              scope,
                              allow_input_subset=allow_input_subset,
                              allow_output_subset=allow_output_subset).get_io()


def _get_name(name, default_name):
    # the default name is chosen if name is None
    return name if name is not None else default_name


# TODO: perhaps make the most general behavior with fn_lst being a general
# indexable object more explicit.
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

    return substitution_module(_get_name(name,
                                         "Or"), {'idx': h_or}, substitution_fn,
                               input_names, output_names, scope)


# TODO: perhaps change slightly the semantics of the repeat parameter.
def mimo_nested_repeat(fn_first,
                       fn_iter,
                       h_num_repeats,
                       input_names,
                       output_names,
                       scope=None,
                       name=None):
    """Nested repetition substitution module.

    The first function function returns a dictionary of inputs and a dictionary
    of outputs, and it is always called once. The second function takes the previous
    dictionaries of inputs and outputs and returns new dictionaries of inputs
    and outputs. The names of the inputs and outputs returned by the functions have
    to match the names of the inputs and outputs of the substitution module.
    The resulting network fragment is used in the place of the substitution
    module.

    Args:
        fn_first (() -> (dict[str, deep_architect.core.Input], dict[str, deep_architect.core.Output])):
            Function that returns the first network fragment, represented as
            dictionary of inputs and outputs.
        fn_iter ((dict[str, deep_architect.core.Input], dict[str, deep_architect.core.Output]) -> (dict[str, deep_architect.core.Input], dict[str, deep_architect.core.Output])):
            Function that takes the previous dictionaries of inputs and outputs
            and it is applied to generate the new dictionaries of inputs
            and outputs. This function is applied one time less that the
            value of the hyperparameter for the number of repeats.
        h_num_repeats (deep_architect.core.Hyperparameter): Hyperparameter for how
            many times should the iterative construct be repeated.
        input_names (list[str]): List of the input names of the substitution module.
        output_name (list[str]): List of the output names of the substitution module.
        scope (deep_architect.core.Scope, optional): Scope in which the module will be
            registered. If none is give`n, uses the default scope.
        name (str, optional): Name used to derive an unique name for the
            module. If none is given, uses the class name to derive
            the name.

    Returns:
        (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output]):
            Tuple with dictionaries with the inputs and outputs of the
            substitution module.
    """

    def substitution_fn(num_reps):
        assert num_reps > 0
        inputs, outputs = fn_first()
        for _ in range(1, num_reps):
            inputs, outputs = fn_iter(inputs, outputs)
        return inputs, outputs

    return substitution_module(_get_name(name, "NestedRepeat"),
                               {'num_reps': h_num_repeats}, substitution_fn,
                               input_names, output_names, scope)


def siso_nested_repeat(fn_first, fn_iter, h_num_repeats, scope=None, name=None):
    """Nested repetition substitution module.

    Similar to :func:`mimo_nested_repeat`, the only difference being that in this
    case the function returns an or substitution module that has a single input
    and a single output.

    The first function function returns a dictionary of inputs and a dictionary
    of outputs, and it is always called. The second function takes the previous
    dictionaries of inputs and outputs and returns new dictionaries of inputs
    and outputs. The resulting network fragment is used in the place of the
    current substitution module.

    Args:
        fn_first (() -> (dict[str, deep_architect.core.Input], dict[str, deep_architect.core.Output])):
            Function that returns the first network fragment, represented as
            dictionary of inputs and outputs.
        fn_iter ((dict[str, deep_architect.core.Input], dict[str, deep_architect.core.Output]) -> (dict[str, deep_architect.core.Input], dict[str, deep_architect.core.Output])):
            Function that takes the previous dictionaries of inputs and outputs
            and it is applied to generate the new dictionaries of inputs
            and outputs. This function is applied one time less that the
            value of the number of repeats hyperparameter.
        h_num_repeats (deep_architect.core.Hyperparameter): Hyperparameter for how
            many times to repeat the iterative construct.
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
    return mimo_nested_repeat(fn_first,
                              fn_iter,
                              h_num_repeats, ['In'], ['Out'],
                              scope=scope,
                              name=_get_name(name, "SISONestedRepeat"))


def siso_or(fn_lst, h_or, scope=None, name=None):
    """Implements an or substitution operation.

    The hyperparameter takes values that are valid indices for the list of
    possible substitution functions. The substitution function chosen is used to
    replace the current substitution module, with connections changed appropriately.

    See also :func:`mimo_or`.

    .. note::
        The current implementation also works if ``fn_lst`` is an indexable
        object (e.g., a dictionary), and the ``h_or`` takes values that
        are valid indices for the dictionary.

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
            module. If none is given, uses the class name to derive the name.

    Returns:
        (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output]):
            Tuple with dictionaries with the inputs and outputs of the
            substitution module.
    """
    return mimo_or(fn_lst,
                   h_or, ['In'], ['Out'],
                   scope=scope,
                   name=_get_name(name, "SISOOr"))


# NOTE: how to do repeat in the general mimo case.


def siso_repeat(fn, h_num_repeats, scope=None, name=None):
    """Calls the function multiple times and connects the resulting graph
    fragments sequentially.

    Args:
        fn (() -> (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output])):
            Function returning a graph fragment corresponding to a sub-search space.
        h_num_repeats (deep_architect.core.Hyperparameter): Hyperparameter for the number
            of times to repeat the search space returned by the function.
        scope (deep_architect.core.Scope, optional): Scope in which the module will be
            registered. If none is given, uses the default scope.
        name (str, optional): Name used to derive an unique name for the
            module. If none is given, uses the class name to derive the name.

    Returns:
        (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output]):
            Tuple with dictionaries with the inputs and outputs of the
            substitution module.
    """

    def substitution_fn(num_reps):
        assert num_reps > 0
        # instantiating all the graph fragments.
        inputs_lst = []
        outputs_lst = []
        for _ in range(num_reps):
            inputs, outputs = fn()
            inputs_lst.append(inputs)
            outputs_lst.append(outputs)

        # creating the sequential connection of the graph fragments.
        for i in range(1, num_reps):
            prev_outputs = outputs_lst[i - 1]
            next_inputs = inputs_lst[i]
            next_inputs['In'].connect(prev_outputs['Out'])
        return inputs_lst[0], outputs_lst[-1]

    return substitution_module(_get_name(name, "SISORepeat"),
                               {'num_reps': h_num_repeats}, substitution_fn,
                               ['In'], ['Out'], scope)


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

    return substitution_module(_get_name(name, "SISOOptional"), {'opt': h_opt},
                               substitution_fn, ['In'], ['Out'], scope)


# TODO: improve by not enumerating permutations
def siso_permutation(fn_lst, h_perm, scope=None, name=None):
    """Substitution module that permutes the sub-search spaces returned by the
    functions in the list and connects them sequentially.

    The hyperparameter takes positive integer values that index the possible
    permutations of the number of elements of the list provided, i.e., factorial
    in the length of the list possible values (zero indexed). The list is
    permuted according to the permutation chosen. The search spaces resulting
    from calling the functions in the list are connected sequentially.

    Args:
        fn_lst (list[() -> (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output])]):
            List of substitution functions.
        h_perm (deep_architect.core.Hyperparameter): Hyperparameter that chooses the
            permutation of the list to consider.
        scope (deep_architect.core.Scope, optional): Scope in which the module will be
            registered. If none is given, uses the default scope.
        name (str, optional): Name used to derive an unique name for the
            module. If none is given, uses the class name to derive the name.

    Returns:
        (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output]):
            Tuple with dictionaries with the inputs and outputs of the
            substitution module.
    """

    def substitution_fn(perm_idx):
        g = itertools.permutations(range(len(fn_lst)))
        for _ in range(perm_idx + 1):
            idxs = next(g)

        inputs_lst = []
        outputs_lst = []
        for i in idxs:
            inputs, outputs = fn_lst[i]()
            inputs_lst.append(inputs)
            outputs_lst.append(outputs)

        for i in range(1, len(fn_lst)):
            prev_outputs = outputs_lst[i - 1]
            next_inputs = inputs_lst[i]

            # NOTE: to extend this, think about the connection structure.
            next_inputs['In'].connect(prev_outputs['Out'])
        return inputs_lst[0], outputs_lst[-1]

    return substitution_module(_get_name(name, "SISOPermutation"),
                               {'perm_idx': h_perm}, substitution_fn, ['In'],
                               ['Out'], scope)


def siso_split_combine(fn, combine_fn, h_num_splits, scope=None, name=None):
    """Substitution module that create a number of parallel single-input
    single-output search spaces by calling the first function, and then combines
    all outputs with a multiple-input single-output search space returned by the
    second function.

    The second function returns a search space to combine the outputs of the
    branch search spaces. The hyperparameter captures how many branches to create.
    The resulting search space has a single input and a single output.

    .. note::
        It is assumed that the inputs and outputs of both the branch search
        spaces and the reduce search spaces are named in a specific way.

    Args:
        fn (() -> (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output])):
            Substitution function that return a single input and single output
            search space encoded by dictionaries of inputs and outputs.
        combine_fn ((int) -> (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output])):
            Returns a search space with a number of inputs equal to the number of
            of branches and combines them into a single output.
        h_num_splits (deep_architect.core.Hyperparameter): Hyperparameter for the
            number of parallel search spaces generated with the first function.
        scope (deep_architect.core.Scope, optional): Scope in which the module will be
            registered. If none is given, uses the default scope.
        name (str, optional): Name used to derive an unique name for the
            module. If none is given, uses the class name to derive the name.

    Returns:
        (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output]):
            Tuple with dictionaries with the inputs and outputs of the
            resulting search space graph.
    """

    def substitution_fn(num_splits):
        inputs_lst, outputs_lst = zip(*[fn() for _ in range(num_splits)])
        c_inputs, c_outputs = combine_fn(num_splits)

        i_inputs, i_outputs = identity()
        for i in range(num_splits):
            i_outputs['Out'].connect(inputs_lst[i]['In'])
            c_inputs['In' + str(i)].connect(outputs_lst[i]['Out'])
        return i_inputs, c_outputs

    return substitution_module(_get_name(name, "SISOSplitCombine"),
                               {'num_splits': h_num_splits}, substitution_fn,
                               ['In'], ['Out'], scope)


def preproc_apply_postproc(preproc_fn, apply_fn, postproc_fn):
    return siso_sequential([preproc_fn(), apply_fn(), postproc_fn()])


def dense_block(h_num_applies,
                h_end_in_combine,
                apply_fn,
                combine_fn,
                scope=None,
                name=None):

    def substitution_fn(num_applies, end_in_combine):
        assert num_applies > 0
        (i_inputs, i_outputs) = identity()
        prev_a_outputs = [i_outputs]
        prev_c_outputs = [i_outputs]
        for idx in range(num_applies):
            (a_inputs, a_outputs) = apply_fn()
            a_inputs['In'].connect(prev_c_outputs[-1]["Out"])
            prev_a_outputs.append(a_outputs)

            if idx < num_applies - 1 or end_in_combine:
                (c_inputs, c_outputs) = combine_fn(idx + 2)
                for i, iter_outputs in enumerate(prev_a_outputs):
                    c_inputs["In%d" % i].connect(iter_outputs["Out"])
                prev_c_outputs.append(c_outputs)

        if end_in_combine:
            o_outputs = prev_c_outputs[-1]
        else:
            o_outputs = prev_a_outputs[-1]
        return (i_inputs, o_outputs)

    return substitution_module(_get_name(name, "DenseBlock"), {
        "num_applies": h_num_applies,
        "end_in_combine": h_end_in_combine
    }, substitution_fn, ["In"], ["Out"], scope)


def siso_residual(main_fn, residual_fn, combine_fn):
    """Residual connection of two functions returning search spaces encoded
    as pairs of dictionaries of inputs and outputs.

    The third function returns a search space to continue the main and residual
    path search spaces. See also: :func:`siso_split_combine`. The main and
    residual functions return search space graphs with a single input and a
    single output . The combine function returns a search space with two
    inputs and a single output.

    .. note::
        Specific names are assumed for the inputs and outputs of the different
        search spaces.

    Args:
        main_fn (() -> (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output])):
            Function returning the dictionaries of the inputs and outputs for
            the search space of the main path of the configuration.
        residual_fn (() -> (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output])):
            Function returning the dictionaries of the inputs and outputs for
            the search space of the residual path of the configuration.
        combine_fn (() -> (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output])):
            Function returning the dictionaries of the inputs and outputs for
            the search space for combining the outputs of the main and residual
            search spaces.

    Returns:
        (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output]):
            Tuple with dictionaries with the inputs and outputs of the
            resulting search space graph.
    """
    (m_inputs, m_outputs) = main_fn()
    (r_inputs, r_outputs) = residual_fn()
    (c_inputs, c_outputs) = combine_fn()

    i_inputs, i_outputs = identity()
    i_outputs['Out'].connect(m_inputs['In'])
    i_outputs['Out'].connect(r_inputs['In'])

    m_outputs['Out'].connect(c_inputs['In0'])
    r_outputs['Out'].connect(c_inputs['In1'])

    return i_inputs, c_outputs


def siso_sequential(io_lst):
    """Connects in a serial connection a list of dictionaries of the inputs and
    outputs encoding single-input single-output search spaces.

    Args:
        io_lst (list[(dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output])]):
            List of single-input single-output dictionaries encoding

    Returns:
        (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output]):
            Tuple with dictionaries with the inputs and outputs of the
            resulting graph resulting from the sequential connection.
    """
    assert len(io_lst) > 0

    prev_outputs = io_lst[0][1]
    for next_inputs, next_outputs in io_lst[1:]:
        prev_outputs['Out'].connect(next_inputs['In'])
        prev_outputs = next_outputs
    return io_lst[0][0], io_lst[-1][1]


def buffer_io(inputs, outputs):
    buffered_inputs = {}
    for name, ix in iteritems(inputs):
        if isinstance(ix.get_module(), SubstitutionModule):
            b_inputs, b_outputs = identity()
            b_outputs['Out'].connect(ix)
            buffered_ix = b_inputs['In']
        else:
            buffered_ix = ix
        buffered_inputs[name] = buffered_ix

    buffered_outputs = {}
    for name, ox in iteritems(outputs):
        if isinstance(ox.get_module(), SubstitutionModule):
            b_inputs, b_outputs = identity()
            ox.connect(b_inputs['In'])
            buffered_ox = b_outputs['Out']
        else:
            buffered_ox = ox
        buffered_outputs[name] = buffered_ox

    return buffered_inputs, buffered_outputs


class SearchSpaceFactory:
    """Helper used to provide a nicer interface to create search spaces.

    The user should inherit from this class and implement :meth:`_get_search_space`.
    The function get_search_space should be given to the searcher upon creation
    of the searcher.

    Args:
        search_space_fn (() -> (dict[str,deep_architect.core.Input], dict[str,deep_architect.core.Output])):
            Returns the inputs and outputs of the search space, ready to be
            specified.
        reset_scope_upon_get (bool): Whether to clean the scope upon getting
            a new search space. Should be ``True`` in most cases.
    """

    def __init__(self, search_space_fn, reset_scope_upon_get=True):
        self.reset_scope_upon_get = reset_scope_upon_get
        self.search_space_fn = search_space_fn

    def get_search_space(self):
        """Returns the buffered search space."""
        if self.reset_scope_upon_get:
            co.Scope.reset_default_scope()

        (inputs, outputs) = buffer_io(*self.search_space_fn())
        return inputs, outputs