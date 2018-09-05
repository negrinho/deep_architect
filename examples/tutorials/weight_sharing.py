from collections import OrderedDict
import tensorflow as tf
from dev.architecture_search_benchmarks.search_spaces.common_eager import siso_tfem

"""
This is an example of how to implement weight sharing with a dynamic deep
learning framework such as Tensorflow eager or pytorch. This example specifically
will use Tensorflow eager.
"""
class WeightSharer(object):
    def __init__(self):
        self.name_to_weight = OrderedDict()
    
    def get(self, name, weight_fn):
        if name not in self.name_to_weight:
            self.name_to_weight[name] = weight_fn()
        return self.name_to_weight[name]

weight_sharer = WeightSharer()

"""
To share weights or any other object, simply use a common name for the object
you wish to share, and pass in a function to initialize the object. The first
time the function is called, the convolution layer will be created. Subsequent
calls will simply retrieve the layer from the WeightSharer object.
"""

def conv2D(filter_size, channels, name):
    def cfn(di, dh):
        conv_fn = lambda: tf.keras.layers.Conv2D(channels, filter_size)
        conv = weight_sharer.get(name, conv_fn)
        def fn(di, isTraining=True):
            return {'Out' : conv(di['In'])}
        return fn

    return siso_tfem('Conv2D', cfn, {})