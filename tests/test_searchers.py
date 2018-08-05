from six.moves import range
from six import iteritems
from deep_architect.core import Module, determine_module_eval_seq, forward
from deep_architect.hyperparameters import Discrete

def multiply_search_space_fn():
    class Multiply(Module):
        def __init__(self, compile_fn, *args, **kwargs):
            Module.__init__(self, *args, **kwargs)
            self._compile_fn = compile_fn
            self._fn = None
            self._register(['In'], ['Out'], {'multiplier': Discrete(['1', '2', '4', '8'])})

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

    def multiply_cfn(_, dh):
        multiplier = dh['multiplier']
        return lambda di: {'Out': multiplier * di['In']}

    module = Multiply(multiply_cfn)

    inp, out = module.get_io()
    return inp, out, module.get_hyperps()

class Tmp(object):
    def __init__(self, name_to_input, name_to_output):
        self.name_to_output = name_to_output
        self.name_to_input = name_to_input
        self._module_seq = None

    def __call__(self, val):
        return self.forward({'In': val})

    def forward(self, name_to_val):
        if self._module_seq is None:
            self._module_seq = determine_module_eval_seq(self.name_to_input.values())
        input_name_to_val = {ix: name_to_val[name]
                             for name, ix in iteritems(self.name_to_input)}
        forward(input_name_to_val, self._module_seq)
        output_name_to_val = {name: ox.val
                              for name, ox in iteritems(self.name_to_output)}
        return output_name_to_val

def test_random_searcher():
    from deep_architect.searchers import RandomSearcher

    for _ in range(5):
        searcher = RandomSearcher(multiply_search_space_fn)
        inputs, outputs, hyperps, _, _ = searcher.sample()
        multiplier = hyperps['multiplier'].get_value()

        val = Tmp(inputs, outputs)
        for i in range(5):
            assert i * multiplier == val(i)['Out']
