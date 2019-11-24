import keras
import keras.layers as kl
import deep_architect.helpers.common as hco

# print(vars(kl).keys())

fns = [
    kl.Layer, kl.Input, kl.InputLayer, kl.InputSpec, kl.Add, kl.Subtract,
    kl.Multiply, kl.Average, kl.Maximum, kl.Minimum, kl.Concatenate, kl.Dot,
    kl.Dense, kl.Activation, kl.Dropout, kl.Flatten, kl.Reshape, kl.Permute,
    kl.RepeatVector, kl.Lambda, kl.ActivityRegularization, kl.Masking,
    kl.SpatialDropout1D, kl.SpatialDropout2D, kl.SpatialDropout3D, kl.Conv1D,
    kl.Conv2D, kl.SeparableConv1D, kl.SeparableConv2D, kl.DepthwiseConv2D,
    kl.Conv2DTranspose, kl.Conv3D, kl.Conv3DTranspose, kl.Cropping1D,
    kl.Cropping2D, kl.Cropping3D, kl.UpSampling1D, kl.UpSampling2D,
    kl.UpSampling3D, kl.ZeroPadding1D, kl.ZeroPadding2D, kl.ZeroPadding3D,
    kl.Convolution1D, kl.Convolution2D, kl.Convolution3D, kl.Deconvolution2D,
    kl.Deconvolution3D, kl.MaxPooling1D, kl.MaxPooling2D, kl.MaxPooling3D,
    kl.AveragePooling1D, kl.AveragePooling2D, kl.AveragePooling3D,
    kl.GlobalMaxPooling1D, kl.GlobalMaxPooling2D, kl.GlobalMaxPooling3D,
    kl.GlobalAveragePooling2D, kl.GlobalAveragePooling1D,
    kl.GlobalAveragePooling3D, kl.MaxPool1D, kl.MaxPool2D, kl.MaxPool3D,
    kl.AvgPool1D, kl.AvgPool2D, kl.AvgPool3D, kl.GlobalMaxPool1D,
    kl.GlobalMaxPool2D, kl.GlobalMaxPool3D, kl.GlobalAvgPool1D,
    kl.GlobalAvgPool2D, kl.GlobalAvgPool3D, kl.LocallyConnected1D,
    kl.LocallyConnected2D, kl.RNN, kl.SimpleRNN, kl.GRU, kl.LSTM,
    kl.SimpleRNNCell, kl.GRUCell, kl.LSTMCell, kl.StackedRNNCells, kl.CuDNNGRU,
    kl.CuDNNLSTM, kl.normalization, kl.BatchNormalization, kl.Embedding,
    kl.GaussianNoise, kl.GaussianDropout, kl.AlphaDropout, kl.LeakyReLU,
    kl.PReLU, kl.ELU, kl.ThresholdedReLU, kl.Softmax, kl.ReLU, kl.Bidirectional,
    kl.TimeDistributed, kl.ConvLSTM2D, kl.ConvLSTM2DCell, kl.MaxoutDense,
    kl.Highway, kl.AtrousConvolution1D, kl.AtrousConvolution2D, kl.Recurrent,
    kl.ConvRecurrent2D
]

raw_fns = {f.__name__: f for f in fns}
m_fns = {f.__name__: hco.get_siso_wrapped_module(f) for f in fns}
io_fns = {f.__name__.lower(): hco.get_siso_wrapped_module_io(f) for f in fns}

# TODO: add some simple functionality to directly get a random model from the
# search space, potentially by passing simple specs to inputs and outputs.