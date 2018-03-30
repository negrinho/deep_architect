import darch.core as co
from six import itervalues, iteritems
from six.moves import xrange
import copy
import itertools

class Empty(co.Module):
    def __init__(self, scope=None, name=None):
        co.Module.__init__(self, scope, name)
        self._register_input("In")
        self._register_output("Out")

    def forward(self):
        self.outputs['Out'].val = self.inputs['In'].val

# NOTE: perhaps refactor to capture similarities between modules.
class SubstitutionModule(co.Module):
    def __init__(self, name, name_to_hyperp, sub_fn,
            input_names, output_names, scope=None):
        co.Module.__init__(self, scope, name)

        self._register(input_names, output_names, name_to_hyperp)
        self._sub_fn = sub_fn
        self._is_done = False
        self._update()

    def _update(self):
        if (not self._is_done) and all(h.is_set() for h in itervalues(self.hyperps)):
            argnames = self._sub_fn.__code__.co_varnames

            kwargs = {}
            for name, h in iteritems(self.hyperps):
                kwargs[name] = h.get_val()
            for name, ix in iteritems(self.inputs):
                if name in argnames:
                    kwargs[name] = ix.val

            new_inputs, new_outputs = self._sub_fn(**kwargs)
            assert frozenset(new_inputs.keys()) == frozenset(self.inputs.keys())
            assert frozenset(new_outputs.keys()) == frozenset(self.outputs.keys())

            self.old_inputs = copy.copy(self.inputs)
            self.old_outputs = copy.copy(self.outputs)

            for name, new_ix in iteritems(new_inputs):
                old_ix = self.inputs[name]
                if old_ix.is_connected():
                    old_ix.reroute_connected_output(new_ix)
                self.inputs[name] = new_ix

            for name, new_ox in iteritems(new_outputs):
                old_ox = self.outputs[name]
                if old_ox.is_connected():
                    old_ox.reroute_all_connected_inputs(new_ox)
                self.outputs[name] = new_ox

            self._is_done = True

def empty():
    return Empty().get_io()

def substitution_module(name, name_to_hyperp, sub_fn,
        input_names, output_names, scope):
    return SubstitutionModule(name,
        name_to_hyperp, sub_fn, input_names, output_names, scope).get_io()

def _get_name(name, default_name):
    return name if name is not None else default_name

def mimo_or(fn_lst, h_or, input_names, output_names, scope=None, name=None):
    def sub_fn(idx):
        return fn_lst[idx]()

    return substitution_module(_get_name(name, "Or"),
        {'idx' : h_or}, sub_fn, input_names, output_names, scope)

def mimo_nested_repeat(fn_first, fn_iter, h_num_repeats,
        input_names, output_names, scope=None, name=None):
    def sub_fn(num_reps):
        assert num_reps > 0
        inputs, outputs = fn_first()
        for _ in range(1, num_reps):
            inputs, outputs = fn_iter(inputs, outputs)
        return inputs, outputs

    return substitution_module(_get_name(name, "NestedRepeat"),
        {'num_reps' : h_num_repeats}, sub_fn, input_names, output_names, scope)

def siso_nested_repeat(fn_first, fn_iter, h_num_repeats, scope=None, name=None):
    return mimo_nested_repeat(fn_first, fn_iter, h_num_repeats, ['In'], ['Out'],
        scope=scope, name=_get_name(name, "SISONestedRepeat"))

def siso_or(fn_lst, h_or, scope=None, name=None):
    return mimo_or(fn_lst, h_or, ['In'], ['Out'],
        scope=scope, name=_get_name(name, "SISOOr"))

# NOTE: how to do repeat in the general mimo case.
def siso_repeat(fn, h_num_repeats, scope=None, name=None):
    def sub_fn(num_reps):
        assert num_reps > 0
        inputs_lst = []
        outputs_lst = []
        for _ in range(num_reps):
            inputs, outputs = fn()
            inputs_lst.append(inputs)
            outputs_lst.append(outputs)

        for i in range(1, num_reps):
            prev_outputs = outputs_lst[i - 1]
            next_inputs = inputs_lst[i]
            next_inputs['In'].connect(prev_outputs['Out'])
        return (inputs_lst[0], outputs_lst[-1])

    return substitution_module(_get_name(name, "SISORepeat"),
        {'num_reps' : h_num_repeats}, sub_fn, ['In'], ['Out'], scope)

def siso_optional(fn, h_opt, scope=None, name=None):
    def sub_fn(opt):
        return fn() if opt else empty()

    return substitution_module(_get_name(name, "SISOOptional"),
        {'opt' : h_opt}, sub_fn, ['In'], ['Out'], scope)

# TODO: improve by not enumerating permutations
def siso_permutation(fn_lst, h_perm, scope=None, name=None):
    def sub_fn(perm_idx):
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
        return (inputs_lst[0], outputs_lst[-1])

    return substitution_module(_get_name(name, "SISOPermutation"),
        {'perm_idx' : h_perm}, sub_fn, ['In'], ['Out'], scope)

def siso_split_combine(fn, combine_fn, h_num_splits, scope=None, name=None):
    def sub_fn(num_splits):
        inputs_lst, outputs_lst = zip(*[fn() for _ in xrange(num_splits)])
        c_inputs, c_outputs = combine_fn(num_splits)

        i_inputs, i_outputs = empty()
        for i in xrange(num_splits):
            i_outputs['Out'].connect(inputs_lst[i]['In'])
            c_inputs['In' + str(i)].connect(outputs_lst[i]['Out'])
        return (i_inputs, c_outputs)

    return substitution_module(_get_name(name, "SISOSplitCombine"),
        {'num_splits' : h_num_splits}, sub_fn, ['In'], ['Out'], scope)

def siso_residual(main_fn, residual_fn, combine_fn):
    (m_inputs, m_outputs) = main_fn()
    (r_inputs, r_outputs) = residual_fn()
    (c_inputs, c_outputs) = combine_fn()

    i_inputs, i_outputs = empty()
    i_outputs['Out'].connect(m_inputs['In'])
    i_outputs['Out'].connect(r_inputs['In'])

    m_outputs['Out'].connect(c_inputs['In0'])
    r_outputs['Out'].connect(c_inputs['In1'])

    return (i_inputs, c_outputs)

def siso_sequential(io_lst):
    assert len(io_lst) > 0

    prev_outputs = io_lst[0][1]
    for next_inputs, next_outputs in io_lst[1:]:
        prev_outputs['Out'].connect(next_inputs['In'])
        prev_outputs = next_outputs
    return io_lst[0][0], io_lst[-1][1]

class SearchSpaceFactory:
    """Helper used to provide a nicer interface to create new search spaces.

    The user should inherit from this class and implement the _get_search_space
    method. The function get_search_space should be given to the searcher
    upon creation.
    """
    def __init__(self, reset_scope_upon_get=True):
        self.reset_scope_upon_get = reset_scope_upon_get

    def get_search_space(self):
        if self.reset_scope_upon_get:
            co.Scope.reset_default_scope()

        (inputs, outputs, hs) = self._get_search_space()

        buffered_inputs = {}
        for name, ix in iteritems(inputs):
            b_inputs, b_outputs = empty()
            b_outputs['Out'].connect(ix)
            buffered_inputs[name] = b_inputs['In']

        buffered_outputs = {}
        for name, ox in iteritems(outputs):
            b_inputs, b_outputs = empty()
            ox.connect(b_inputs['In'])
            buffered_outputs[name] = b_outputs['Out']

        return buffered_inputs, buffered_outputs, hs

    def _get_search_space(self):
        raise NotImplementedError