import deep_architect.core as co


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
