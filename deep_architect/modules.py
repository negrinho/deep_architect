import deep_architect.core as co
import itertools


def _get_name(name, default_name):
    # the default name is chosen if name is None
    return name if name is not None else default_name


def _get_io_if_module(x):
    return x.get_io() if isinstance(x, co.Module) else x


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
        super().__init__(["in"], ["out"], {}, scope, name)

    def compile(self):
        pass

    def forward(self):
        self.outputs['out'].val = self.inputs['in'].val


class HyperparameterAggregator(co.Module):

    def __init__(self, name_to_hyperp, scope=None, name=None):
        super().__init__(["in"], ["out"], name_to_hyperp, scope, name)

    def compile(self):
        pass

    def forward(self):
        self.outputs['out'].val = self.inputs['in'].val


def get_hyperparameter_aggregators(outputs):
    co.get_modules_with_cond(outputs,
                             lambda m: isinstance(m, HyperparameterAggregator))


class MIMOOr(co.SubstitutionModule):
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

    def __init__(self,
                 fn_lst,
                 h_choice,
                 input_names,
                 output_names,
                 scope=None,
                 name=None):
        super().__init__(input_names,
                         output_names, {"choice": h_choice},
                         scope=scope,
                         name=name)
        self.fn_lst = fn_lst

    def substitute(self):
        choice = self.hyperps["choice"].val
        x = self.fn_lst[choice]()
        return _get_io_if_module(x)


class MIMONestedRepeat(co.SubstitutionModule):
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

    def __init__(self,
                 fn_first,
                 fn_iter,
                 h_num_repeats,
                 input_names,
                 output_names,
                 scope=None,
                 name=None):
        super().__init__(input_names,
                         output_names, {"num_repeats": h_num_repeats},
                         scope=scope,
                         name=name)
        self.fn_first = fn_first
        self.fn_iter = fn_iter

    def substitute(self):
        num_repeats = self.hyperps["num_repeats"].val
        if num_repeats <= 0:
            raise ValueError(
                "Number of repeats must be greater than zero. Assigned value: %d"
                % num_repeats)

        x = self.fn_first()
        inputs, outputs = _get_io_if_module(x)
        for _ in range(1, num_repeats):
            x = self.fn_iter(inputs, outputs)
            inputs, outputs = _get_io_if_module(x)
        return inputs, outputs


class SISONestedRepeat(MIMONestedRepeat):
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

    def __init__(self,
                 fn_first,
                 fn_iter,
                 h_num_repeats,
                 input_names,
                 output_names,
                 scope=None,
                 name=None):
        super().__init__(fn_first,
                         fn_iter,
                         h_num_repeats, ["in"], ["out"],
                         scope=scope,
                         name=name)


class SISOOr(MIMOOr):
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

    def __init__(self,
                 fn_first,
                 fn_iter,
                 h_num_repeats,
                 input_names,
                 output_names,
                 scope=None,
                 name=None):
        super().__init__(fns, h_choice, ["in"], ["out"], scope=scope, name=name)



class SISORepeat(co.SubstitutionModule):
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

    def __init__(self, fn, h_num_repeats, scope=None, name=None):
        super().__init__(["in"], ["out"], {"num_repeats": h_num_repeats},
                         scope=scope,
                         name=name)
        self.fn = fn

    def substitute(self):
        num_repeats = self.hyperps["num_repeats"].val
        if num_repeats <= 0:
            raise ValueError(
                "Number of repeats must be greater than zero. Assigned value: %d"
                % num_repeats)

        return siso_sequential([self.fn() for _ in range(num_repeats)])


class SISOOptional(co.SubstitutionModule):
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

    def __init__(self, fn, h_opt, scope=None, name=None):
        super().__init__(["in"], ["out"], {"opt": h_opt},
                         scope=scope,
                         name=name)
        self.fn = fn

    def substitute(self):
        opt = self.hyperps["opt"].val
        x = self.fn() if opt else Identity()
        return _get_io_if_module(x)


class SISOPermutation(co.SubstitutionModule):
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

    def __init__(self, fn_lst, h_perm, scope=None, name=None):
        super().__init__(["in"], ["out"], {"perm_idx": h_perm},
                         scope=scope,
                         name=name)
        self.fn_lst = fn_lst

    def substitute(self):
        perm_idx = self.hyperps["perm_idx"].val
        g = itertools.permutations(range(len(self.fn_lst)))
        for _ in range(perm_idx + 1):
            indices = next(g)
        return siso_sequential([self.fn_lst[i]() for i in indices])


class SISOSplitCombine(co.SubstitutionModule):
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

    def __init__(self, fn, combine_fn, h_num_splits, scope=None, name=None):
        super().__init__(["in"], ["out"], {'num_splits': h_num_splits},
                         scope=scope,
                         name=name)
        self.fn = fn
        self.combine_fn = combine_fn

    def substitute(self):
        num_splits = self.hyperps['num_splits'].val
        inputs_lst, outputs_lst = list(
            zip(*[_get_io_if_module(self.fn()) for _ in range(num_splits)]))
        x = self.combine_fn(num_splits)
        c_inputs, c_outputs = _get_io_if_module(x)

        i_inputs, i_outputs = identity()
        for i in range(num_splits):
            i_outputs['out'].connect(inputs_lst[i]['in'])
            c_inputs['in' + str(i)].connect(outputs_lst[i]['out'])
        return i_inputs, c_outputs


def preproc_apply_postproc(preproc_fn, apply_fn, postproc_fn):
    return siso_sequential([preproc_fn(), apply_fn(), postproc_fn()])


class DenseBlock():

    def __init__(self,
                 h_num_applies,
                 h_end_in_combine,
                 apply_fn,
                 combine_fn,
                 scope=None,
                 name=None):
        super().__init__(["in"], ["out"], {
            "num_applies": h_num_applies,
            "end_in_combine": h_end_in_combine
        })
        self.apply_fn = apply_fn
        self.combine_fn = combine_fn

    def substitute(self):
        dh = self._get_hyperp_values()
        (i_inputs, i_outputs) = Identity().get_io()

        prev_a_outputs = [i_outputs]
        prev_c_outputs = [i_outputs]
        for idx in range(dh["num_applies"]):
            x = self.apply_fn()
            (a_inputs, a_outputs) = _get_io_if_module(x)
            a_inputs['in'].connect(prev_c_outputs[-1]["out"])
            prev_a_outputs.append(a_outputs)

            if idx < dh["num_applies"] - 1 or dh["end_in_combine"]:
                x = self.combine_fn(idx + 2)
                (c_inputs, c_outputs) = _get_io_if_module(x)
                for i, iter_outputs in enumerate(prev_a_outputs):
                    c_inputs["in%d" % i].connect(iter_outputs["out"])
                prev_c_outputs.append(c_outputs)

        if dh["end_in_combine"]:
            o_outputs = prev_c_outputs[-1]
        else:
            o_outputs = prev_a_outputs[-1]
        return (i_inputs, o_outputs)


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
    (m_inputs, m_outputs) = _get_io_if_module(main_fn())
    (r_inputs, r_outputs) = _get_io_if_module(residual_fn())
    (c_inputs, c_outputs) = _get_io_if_module(combine_fn())

    i_inputs, i_outputs = Identity().get_io()
    i_outputs['out'].connect(m_inputs['in'])
    i_outputs['out'].connect(r_inputs['in'])

    m_outputs['out'].connect(c_inputs['in0'])
    r_outputs['out'].connect(c_inputs['in1'])

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

    io_lst = [_get_io_if_module(x) for x in io_lst]
    prev_outputs = io_lst[0][1]
    for next_inputs, next_outputs in io_lst[1:]:
        prev_outputs['out'].connect(next_inputs['in'])
        prev_outputs = next_outputs
    return io_lst[0][0], io_lst[-1][1]


### NOTE: this needs to be fixed for the new case.
def buffer_io(inputs, outputs):
    buffered_inputs = {}
    for name, ix in inputs.items():
        if isinstance(ix.get_module(), co.SubstitutionModule):
            b_inputs, b_outputs = Identity().get_io()
            b_outputs['out'].connect(ix)
            buffered_ix = b_inputs['in']
        else:
            buffered_ix = ix
        buffered_inputs[name] = buffered_ix

    buffered_outputs = {}
    for name, ox in outputs.items():
        if isinstance(ox.get_module(), co.SubstitutionModule):
            b_inputs, b_outputs = Identity().get_io()
            ox.connect(b_inputs['in'])
            buffered_ox = b_outputs['out']
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
        reset_default_scope_upon_get (bool): Whether to clean the scope upon getting
            a new search space. Should be ``True`` in most cases.
    """

    def __init__(self, search_space_fn, reset_default_scope_upon_get=True):
        self.reset_default_scope_upon_get = reset_default_scope_upon_get
        self.search_space_fn = search_space_fn

    def get_search_space(self):
        """Returns the buffered search space."""
        if self.reset_default_scope_upon_get:
            co.Scope.reset_default_scope()

        (inputs, outputs) = buffer_io(*self.search_space_fn())
        return inputs, outputs
