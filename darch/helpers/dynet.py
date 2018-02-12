import darch.core as co

# TODO: adapt for the new type of module.
class DyModule(co.Module):
    def __init__(self, name, name_to_h, compile_fn, 
            input_names, output_names, scope=None):
        co.Module.__init__(self, scope, name)

        if input_names is None:
            input_names = ["In"]
        for name in input_names:
            self._register_input(name)
        
        if output_names is None:
            output_names = ["Out"]        
        for name in output_names:
            self._register_output(name)

        for name, h in iteritems(name_to_h):
            self._register_hyperparameter(h, name)

        self._compile_fn = compile_fn

    # NOTE: while it may not depend on the inputs, it should 
    # always depend on the hyperparameters, otherwise it does not 
    # make sense.
    def _compile(self):
        argnames = self._compile_fn.__code__.co_varnames

        kwargs = {}
        for name, h in iteritems(self.hs):
            kwargs[name] = h.get_val()
        for name, ix in iteritems(self.inputs):
            if name in argnames:        
                kwargs[name] = ix.val

        self._fn = self._compile_fn(**kwargs)

    # NOTE: this is going to be interesting. how to do the compilation in 
    # this case?
    def _forward(self):
        kwargs = {name : ix.val for name, ix in iteritems(self.inputs) }
        name_to_val = self._fn(**kwargs)
        for name, val in iteritems(name_to_val):
            self.outputs[name].val = val