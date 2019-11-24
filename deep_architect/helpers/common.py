import deep_architect.core as co
import inspect


def get_positional_argnames(fn):
    sig = inspect.signature(fn)

    names = []
    for k, v in sig.parameters.items():
        if v.kind == v.POSITIONAL_OR_KEYWORD:
            names.append(k)
    return names


def get_name_to_hyperp(fn, args, kwargs):
    name_to_hyperp = {k: v for k, v in kwargs.items()}
    argnames = get_positional_argnames(fn)
    if len(argnames) < len(args):
        raise ValueError(
            "The number of positional arguments for %s cannot exceed those declared explicitly, i.e., *args cannot be used."
            % fn.__name__)
    for k, v in zip(argnames, args):
        if k in name_to_hyperp:
            raise ValueError(
                "Arguments should be passed positionally or as keywords, but not both. Arg name %s in %s."
                % (k, fn.__name__))
        name_to_hyperp[k] = v
    return name_to_hyperp


def compile_forward(inputs, outputs, name_to_input_val, module_eval_seq,
                    input_cleanup_seq, output_cleanup_seq):
    for name, ix in inputs.items():
        ix.val = name_to_input_val[name]

    for i, m in enumerate(module_eval_seq):
        m.compile()
        m.forward()

        for ox in m.outputs.values():
            for ix in ox.get_connected_inputs():
                ix.val = ox.val

        for ix in input_cleanup_seq[i]:
            del ix.val

        for ox in output_cleanup_seq[i]:
            del ox.val

    name_to_output_val = {name: ox.val for name, ox in outputs.items()}
    for ox in outputs.values():
        del ox.val
    return name_to_output_val


def forward(inputs, outputs, name_to_input_val, module_eval_seq,
            input_cleanup_seq, output_cleanup_seq):
    for name, ix in inputs.items():
        ix.val = name_to_input_val[name]

    for i, m in enumerate(module_eval_seq):
        m.forward()

        for ox in m.outputs.values():
            for ix in ox.get_connected_inputs():
                ix.val = ox.val

        for ix in input_cleanup_seq[i]:
            del ix.val

        for ox in output_cleanup_seq[i]:
            del ox.val

    name_to_output_val = {name: ox.val for name, ox in outputs.items()}
    for ox in outputs.values():
        del ox.val
    return name_to_output_val


def simplified_compile_forward(inputs, outputs, name_to_input_val):

    module_eval_seq = co.determine_module_eval_seq(inputs)
    x = co.determine_input_output_cleanup_seq(inputs)
    input_cleanup_seq, output_cleanup_seq = x
    output_name_to_val = compile_forward(inputs, outputs, name_to_input_val,
                                         module_eval_seq, input_cleanup_seq,
                                         output_cleanup_seq)
    return output_name_to_val


class Model:

    def __init__(self, inputs, outputs, init_input_name_to_val):
        self.outputs = outputs
        self.inputs = inputs
        self._module_eval_seq = co.determine_module_eval_seq(self.inputs)
        x = co.determine_input_output_cleanup_seq(self.inputs)
        self._input_cleanup_seq, self._output_cleanup_seq = x
        compile_forward(self.inputs, self.outputs, init_input_name_to_val,
                        self._module_eval_seq, self._input_cleanup_seq,
                        self._output_cleanup_seq)

    def forward(self, input_name_to_val):

        input_to_val = {
            ix: input_name_to_val[name] for (name, ix) in self.inputs.items()
        }
        output_name_to_val = hco.forward(self.inputs, self.outputs,
                                         input_name_to_val,
                                         self._module_eval_seq,
                                         self._input_cleanup_seq,
                                         self._output_cleanup_seq)
        return output_name_to_val


class SISOWrappedModule(co.Module):

    def __init__(self, fn, scope=None, name=None, *args, **kwargs):
        name = name if name is not None else fn.__name__
        name_to_hyperp = get_name_to_hyperp(fn, args, kwargs)
        super().__init__(["in"], ["out"], name_to_hyperp, scope, name)
        self.fn = fn

    def compile(self):
        dh = self._get_hyperp_values()
        self.m = self.fn(**dh)

    def forward(self):
        self.outputs["out"].val = self.m(self.inputs["in"].val)


class MIMOWrappedModule(co.Module):

    def __init__(self,
                 fn,
                 num_inputs,
                 num_outputs,
                 scope=None,
                 name=None,
                 *args,
                 **kwargs):
        name = name if name is not None else fn.__name__
        name_to_hyperp = get_name_to_hyperp(fn, args, kwargs)
        super().__init__(["in%d" % i for i in range(num_inputs)],
                         ["out%d" % i for i in range(num_outputs)],
                         name_to_hyperp, scope, name)
        self.fn = fn

    def compile(self):
        dh = self._get_hyperp_values()
        self.m = self.fn(**dh)

    def forward(self):
        args = [self.inputs["in%d" % i].val for i in range(len(self.inputs))]
        outs = self.m(*args)
        for i in range(len(self.outputs)):
            self.outputs['out%d' % i].val = outs[i]


class ListWrappedModule(co.Module):
    """Useful to wrap functions that take a list of arguments and return
    a single argument (e.g., often the case for combination functions such as
    concat and add).
    """

    def __init__(self, fn, num_inputs, scope=None, name=None, *args, **kwargs):
        name = name if name is not None else fn.__name__
        name_to_hyperp = get_name_to_hyperp(fn, args, kwargs)
        super().__init__(["in%d" for i in range(num_inputs)], ["out"],
                         name_to_hyperp, scope, name)
        self.fn = fn

    def compile(self):
        dh = self._get_hyperp_values()
        self.m = self.fn(**dh)

    def forward(self):
        lst = [self.inputs["in%i" % i].val for i in range(len(self.inputs))]
        self.outputs["out"].val = self.m(lst)


# TODO: change this is accordance after other types are introduced.
def get_siso_wrapped_module(fn, scope=None, name=None, **kwargs):

    def wrapped_fn(*args, **kwargs):
        return SISOWrappedModule(fn, scope, name, *args, **kwargs)

    return wrapped_fn


def get_siso_wrapped_module_io(fn, scope=None, name=None, **kwargs):

    def wrapped_fn(*args, **kwargs):
        return SISOWrappedModule(fn, scope, name, *args, **kwargs).get_io()

    return wrapped_fn
