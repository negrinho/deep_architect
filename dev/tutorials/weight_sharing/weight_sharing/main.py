###${MARKDOWN}
# Recently, with the success of papers such as
# [Efficient Neural Architecture Search via Parameter Sharing](https://arxiv.org/abs/1802.03268),
# weight sharing has emerged as a popular way to speed up evaluation of sampled
# architectures in architecture search. Weight sharing simply involves sharing
# the weights of common layers between different sampled architectures.
# While DeepArchitect does not currently support weight sharing natively,
# there is a simple way to implement weight sharing within the context of the
# framework.
#
# This tutorial demonstrates how to implement basic weight sharing with a dynamic deep
# learning framework such as Tensorflow eager or pytorch. This example specifically
# will use Tensorflow eager.

import tensorflow as tf
from deep_architect.helpers.tfeager import siso_tfeager_module


# The WeightSharer object is a simple wrapper around a dictionary that stores a
# mapping from a layer name to the shared layer used by the DeepArchitect modules.
class WeightSharer(object):

    def __init__(self):
        self.name_to_weight = {}

    def get(self, name, weight_fn):
        if name not in self.name_to_weight:
            self.name_to_weight[name] = weight_fn()
        return self.name_to_weight[name]


weight_sharer = WeightSharer()


# Now to share weights or any other object, simply use a common name for the object
# you wish to share, and pass in a function to initialize the object. The first
# time the function is called, the convolution layer will be created. Subsequent
# calls will simply retrieve the layer from the WeightSharer object.
def conv2D(filter_size, channels, name):

    def compile_fn(di, dh):
        conv_fn = lambda: tf.keras.layers.Conv2D(channels, filter_size)
        conv = weight_sharer.get(name, conv_fn)

        def fn(di, isTraining=True):
            return {'Out': conv(di['In'])}

        return fn

    return siso_tfeager_module('Conv2D', compile_fn, {})


conv_original = conv2D(3, 32, 'conv_layer')

# Now, calling the function again with the same 'name' argument will return
# a DeepArchitect module with the same internal convolutional layer
conv_shared = conv2D(3, 32, 'conv_layer')

# Implementing such a weight sharing scheme with another dynamic framework such
# as Pytorch is just as straightforward as above. To implement a weight sharing
# scheme with a graph based framework like Tensorflow, you must run the tensors
# that you wish to store and store the actual weights.