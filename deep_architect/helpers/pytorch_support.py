import torch.nn as nn
import deep_architect.core as co
import deep_architect.modules as mo
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
        output_name_to_val = hco.forward(self.inputs, self.outputs,
                                         input_name_to_val,
                                         self._module_eval_seq,
                                         self._input_cleanup_seq,
                                         self._output_cleanup_seq)
        return output_name_to_val


# vars(nn) can be used to inspect the functions present that can be wrapped.
# TODO: some of these that are not single input single output.
# TODO: some of these return two inputs. still need to handle concats and sums.
fns = [
    nn.Identity, nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d,
    nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Threshold, nn.ReLU, nn.Hardtanh,
    nn.ReLU6, nn.Sigmoid, nn.Tanh, nn.Softmax, nn.Softmax2d, nn.LogSoftmax,
    nn.ELU, nn.SELU, nn.CELU, nn.GLU, nn.Hardshrink, nn.LeakyReLU,
    nn.LogSigmoid, nn.Softplus, nn.Softshrink, nn.MultiheadAttention, nn.PReLU,
    nn.Softsign, nn.Softmin, nn.Tanhshrink, nn.RReLU, nn.L1Loss, nn.NLLLoss,
    nn.KLDivLoss, nn.MSELoss, nn.BCELoss, nn.BCEWithLogitsLoss, nn.NLLLoss2d,
    nn.PoissonNLLLoss, nn.CosineEmbeddingLoss, nn.CTCLoss,
    nn.HingeEmbeddingLoss, nn.MarginRankingLoss, nn.MultiLabelMarginLoss,
    nn.MultiLabelSoftMarginLoss, nn.MultiMarginLoss, nn.SmoothL1Loss,
    nn.SoftMarginLoss, nn.CrossEntropyLoss, nn.AvgPool1d, nn.AvgPool2d,
    nn.AvgPool3d, nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.MaxUnpool1d,
    nn.MaxUnpool2d, nn.MaxUnpool3d, nn.FractionalMaxPool2d,
    nn.FractionalMaxPool3d, nn.LPPool1d, nn.LPPool2d, nn.LocalResponseNorm,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm1d,
    nn.InstanceNorm2d, nn.InstanceNorm3d, nn.LayerNorm, nn.GroupNorm,
    nn.SyncBatchNorm, nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout,
    nn.FeatureAlphaDropout, nn.ReflectionPad1d, nn.ReflectionPad2d,
    nn.ReplicationPad2d, nn.ReplicationPad1d, nn.ReplicationPad3d,
    nn.CrossMapLRN2d, nn.Embedding, nn.EmbeddingBag, nn.RNN, nn.LSTM, nn.GRU,
    nn.RNNCell, nn.LSTMCell, nn.GRUCell, nn.PixelShuffle, nn.Upsample,
    nn.UpsamplingNearest2d, nn.UpsamplingBilinear2d, nn.PairwiseDistance,
    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
    nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
    nn.TripletMarginLoss, nn.ZeroPad2d, nn.ConstantPad1d, nn.ConstantPad2d,
    nn.ConstantPad3d, nn.Bilinear, nn.CosineSimilarity, nn.Unfold, nn.Fold,
    nn.AdaptiveLogSoftmaxWithLoss, nn.TransformerEncoder, nn.TransformerDecoder,
    nn.TransformerEncoderLayer, nn.TransformerDecoderLayer, nn.Transformer,
    nn.Flatten
]

raw_fns = {f.__name__: f for f in fns}
m_fns = {f.__name__: hco.get_siso_wrapped_module(f) for f in fns}
io_fns = {f.__name__.lower(): hco.get_siso_wrapped_module_io(f) for f in fns}
