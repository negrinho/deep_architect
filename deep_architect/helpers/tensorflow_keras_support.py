import tensorflow as tf
# import tensorflow.keras
import tensorflow.keras.layers as kl
import deep_architect.helpers.common as hco

# print(vars(kl).keys())

fns = [
    kl.DenseFeature, kl.DenseFeatures, kl.Layer, kl.Input, kl.InputLayer,
    kl.InputSpec, kl.ELU, kl.LeakyReLU, kl.PReLU, kl.ReLU, kl.Softmax,
    kl.ThresholdedReLU, kl.Conv1D, kl.Convolution1D, kl.Conv2D,
    kl.Convolution2D, kl.Conv2DTranspose, kl.Convolution2DTranspose, kl.Conv3D,
    kl.Convolution3D, kl.Conv3DTranspose, kl.Convolution3DTranspose,
    kl.Cropping1D, kl.Cropping2D, kl.Cropping3D, kl.DepthwiseConv2D,
    kl.SeparableConv1D, kl.SeparableConvolution1D, kl.SeparableConv2D,
    kl.SeparableConvolution2D, kl.UpSampling1D, kl.UpSampling2D,
    kl.UpSampling3D, kl.ZeroPadding1D, kl.ZeroPadding2D, kl.ZeroPadding3D,
    kl.ConvLSTM2D, kl.Activation, kl.ActivityRegularization, kl.Dense,
    kl.Dropout, kl.Flatten, kl.Lambda, kl.Masking, kl.Permute, kl.RepeatVector,
    kl.Reshape, kl.SpatialDropout1D, kl.SpatialDropout2D, kl.SpatialDropout3D,
    kl.CuDNNGRU, kl.CuDNNLSTM, kl.AdditiveAttention, kl.Attention, kl.Embedding,
    kl.LocallyConnected1D, kl.LocallyConnected2D, kl.Add, kl.Average,
    kl.Concatenate, kl.Dot, kl.Maximum, kl.Minimum, kl.Multiply, kl.Subtract,
    kl.AlphaDropout, kl.GaussianDropout, kl.GaussianNoise,
    kl.BatchNormalization, kl.LayerNormalization, kl.AveragePooling1D,
    kl.AvgPool1D, kl.AveragePooling2D, kl.AvgPool2D, kl.AveragePooling3D,
    kl.AvgPool3D, kl.GlobalAveragePooling1D, kl.GlobalAvgPool1D,
    kl.GlobalAveragePooling2D, kl.GlobalAvgPool2D, kl.GlobalAveragePooling3D,
    kl.GlobalAvgPool3D, kl.GlobalMaxPool1D, kl.GlobalMaxPooling1D,
    kl.GlobalMaxPool2D, kl.GlobalMaxPooling2D, kl.GlobalMaxPool3D,
    kl.GlobalMaxPooling3D, kl.MaxPool1D, kl.MaxPooling1D, kl.MaxPool2D,
    kl.MaxPooling2D, kl.MaxPool3D, kl.MaxPooling3D, kl.GRU, kl.GRUCell, kl.LSTM,
    kl.LSTMCell, kl.RNN, kl.SimpleRNN, kl.SimpleRNNCell, kl.StackedRNNCells,
    kl.Bidirectional, kl.TimeDistributed
]

raw_fns = {f.__name__: f for f in fns}
m_fns = {f.__name__: hco.get_siso_wrapped_module(f) for f in fns}
io_fns = {f.__name__.lower(): hco.get_siso_wrapped_module_io(f) for f in fns}

g = globals()
for name, fn in m_fns.items():
    g[name] = fn

for name, fn in io_fns.items():
    g[name] = fn
