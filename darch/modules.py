import darch.core as co
import darch.hyperparameters as hyp
import darch.utils as ut
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

# NOTE: this is kind of similar to TFModule and the other helper.
# perhaps only add the modifications to that functionality.
# NOTE: this part can be done at the module functionality.
# also unregister modules.
class SubstitutionModule(co.Module):
    def __init__(self, name, name_to_h, fn, input_names, output_names, scope=None):
        co.Module.__init__(self, scope, name)

        for name in input_names:
            self._register_input(name)
        for name in output_names:
            self._register_output(name)

        for name, h in iteritems(name_to_h):
            self._register_hyperparameter(h, name)
        
        self._fn = fn
        self._is_done = False
        self._update()

    def _update(self):
        if (not self._is_done) and all(h.is_set() for h in itervalues(self.hs)):
            argnames = self._fn.__code__.co_varnames
            
            kwargs = {}
            for name, h in iteritems(self.hs):
                kwargs[name] = h.get_val()
            for name, ix in iteritems(self.inputs):
                if name in argnames:        
                    kwargs[name] = ix.val

            new_inputs, new_outputs = self._fn(**kwargs)
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
            
            # NOTE: unregister removed from now.
            # could be done here in an OK manner.
            # self._unregister()
            self._is_done = True

def Or(fn_lst, h_or, input_names, output_names, scope=None, name=None):
    if name == None:
        name = "Or"

    def sub_fn(idx):
        return fn_lst[idx]()

    return SubstitutionModule(name, {'idx' : h_or}, sub_fn,
        input_names, output_names, scope)

# NOTE: useful for nesting constructions.
def NestedRepeat(fn_first, fn_iter, h_reps, input_names, output_names, scope=None, name=None):
    if name == None:
        name = "NestedRepeat"

    def sub_fn(num_reps):
        assert num_reps > 0

        inputs, outputs = fn_first()
        for _ in range(1, num_reps):
            inputs, outputs = fn_iter(inputs, outputs)
        return inputs, outputs
    
    return SubstitutionModule(name, {'num_reps' : h_reps}, sub_fn,
        input_names, output_names, scope)

def SISONestedRepeat(fn_first, fn_iter, h_reps, scope=None, name=None):
    if name == None:
        name = "SISONestedRepeat"

    return NestedRepeat(fn_first, fn_iter, h_reps, 
        ['In'], ['Out'], scope=scope, name=name)

def SISOOr(fn_lst, h_or, scope=None, name=None):
    if name == None:
        name = "SISOOr"

    return Or(fn_lst, h_or, 
        ['In'], ['Out'], scope=scope, name=name)

# NOTE: how to do repeat in the general case. it is possible,
# but requires connections of the inputs and outputs. how.
def SISORepeat(fn, h_reps, scope=None, name=None):
    if name == None:
        name = "SISORepeat"

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

            # NOTE: if extending this, it is worth to look in terms of 
            # the connection structure.
            next_inputs['In'].connect(prev_outputs['Out'])

        return (inputs_lst[0], outputs_lst[-1])
    
    return SubstitutionModule(name, {'num_reps' : h_reps}, sub_fn,
        ['In'], ['Out'], scope)

def SISOOptional(fn, h_opt, scope=None, name=None):
    if name == None:
        name = "SISOOptional"

    def sub_fn(Opt):
        if Opt:
            return fn()
        else:
            m = Empty()
            return (m.inputs, m.outputs)

    return SubstitutionModule(name, {'Opt' : h_opt}, sub_fn,
        ['In'], ['Out'], scope)

# NOTE: this assumes that there is a single one.
# TODO: add some simple hyperparameters.
# NOTE: this is only meant for permutations of a few elements.
# TODO: this can be done better without enumerating the 
# permutations
# TODO: also change hyperparameter.
def SISOPermutation(fn_lst, h_perm, scope=None, name=None):
    if name == None:
        name = "SISOPermutation"

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

            # NOTE: if extending this, it is worth to look in terms of 
            # the connection structure.
            next_inputs['In'].connect(prev_outputs['Out'])
        
        return (inputs_lst[0], outputs_lst[-1])

    return SubstitutionModule(name, {'perm_idx' : h_perm}, sub_fn,
        ['In'], ['Out'], scope)    

def SISOSplitCombine(fn, combine_fn, h_num_splits, scope=None, name=None):
    if name == None:
        name = "SISOSplitCombine"

    def sub_fn(num_splits):
        inputs_lst, outputs_lst = zip(*[fn() for _ in xrange(num_splits)])
        c_inputs, c_outputs = combine_fn(num_splits)        

        i_inputs, i_outputs = ut.module_to_io( Empty() )
        for i in xrange(num_splits):
            i_outputs['Out'].connect( inputs_lst[i]['In'] )
            c_inputs['In' + str(i)].connect( outputs_lst[i]['Out'] )

        return (i_inputs, c_outputs)

    return SubstitutionModule(name, {'num_splits' : h_num_splits}, sub_fn,
        ['In'], ['Out'], scope)   

def SISOResidual(main_fn, res_fn, combine_fn):
    (m_inputs, m_outputs) = main_fn()
    (r_inputs, r_outputs) = res_fn()
    (c_inputs, c_outputs) = combine_fn()

    i_inputs, i_outputs = ut.module_to_io( Empty() )
    i_outputs['Out'].connect( m_inputs['In'] )
    i_outputs['Out'].connect( r_inputs['In'] )

    m_outputs['Out'].connect( c_inputs['In0'] )
    r_outputs['Out'].connect( c_inputs['In1'] )

    return (i_inputs, c_outputs)
