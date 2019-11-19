import torch.nn as nn
import deep_architect.core as co
import deep_architect.helpers.common as hco


def get_pytorch_modules(outputs):
    all_modules = set()

    def fn(m):
        for x in vars(m).values():
            if isinstance(x, nn.Module):
                all_modules.add(x)

    co.traverse_backward(outputs, fn)

    return list(all_modules)


class PyTorchModel(nn.Module):
    """Encapsulates a network of modules of type :class:`deep_architect.helpers.pytorch_support.PyTorchModule`
    in a way that they can be used as :class:`torch.nn.Module`, e.g.,
    functionality to move the computation of the GPU or to get all the parameters
    involved in the computation are available.

    Using this class is the recommended way of wrapping a Pytorch architecture
    sampled from a search space. The topological order for evaluating for
    doing the forward computation of the architecture is computed by the
    container and cached for future calls to forward.

    Args:
        inputs (dict[str,deep_architect.core.Input]): Dictionary of names to inputs.
        outputs (dict[str,deep_architect.core.Output]): Dictionary of names to outputs.
    """

    def __init__(self, inputs, outputs, init_input_name_to_val):
        super().__init__()

        self.outputs = outputs
        self.inputs = inputs
        self._module_eval_seq = co.determine_module_eval_seq(self.inputs)
        x = co.determine_input_output_cleanup_seq(self.inputs)
        self._input_cleanup_seq, self._output_cleanup_seq = x
        hco.compile_forward(self.inputs, self.outputs, init_input_name_to_val,
                    self._module_eval_seq, self._input_cleanup_seq,
                    self._output_cleanup_seq)
        modules = get_pytorch_modules(self.outputs)
        for i, m in enumerate(modules):
            self.add_module(str(i), m)

    def __call__(self, input_name_to_val):
        return self.forward(input_name_to_val)

    def forward(self, input_name_to_val):
        """Forward computation of the module that is represented through the
        graph of DeepArchitect modules.
        """

        input_to_val = {
            ix: input_name_to_val[name] for (name, ix) in self.inputs.items()
        }
        output_name_to_val = hco.forward(self.inputs, self.outputs, input_name_to_val,
                    self._module_eval_seq, self._input_cleanup_seq,
                    self._output_cleanup_seq)
        return output_name_to_val
