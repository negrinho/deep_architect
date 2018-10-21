# def Module_fn(filter_ns, filter_ls, keep_ps, repeat_ns):
#     b = RepeatTied(
#     Concat([
#         Conv2D(filter_ns, filter_ls, [1], ["SAME"]),
#         MaybeSwap_fn( ReLU(), BatchNormalization() ),
#         Optional_fn( Dropout(keep_ps) )
#     ]), repeat_ns)
#     return b

# filter_nums = range(48, 129, 16)
# repeat_nums = [2 ** i for i in xrange(6)]
# mult_fn = lambda ls, alpha: list(alpha * np.array(ls))
# M = Concat([MH,
#         Conv2D(filter_nums, [3, 5, 7], [2], ["SAME"]),
#         Module_fn(filter_nums, [3, 5], [0.5, 0.9], repeat_nums),
#         Conv2D(filter_nums, [3, 5, 7], [2], ["SAME"]),
#         Module_fn(mult_fn(filter_nums, 2), [3, 5], [0.5, 0.9], repeat_nums),
#         Affine([num_classes], aff_initers) ])

import deep_architect.core as co
import deep_architect.modules as mo
import deep_architect.hyperparameters as hp
import deep_architect.utils as ut
from deep_architect.searchers.common import random_specify
import deep_architect.helpers.keras as hke
import deep_architect.visualization as vi

D = hp.Discrete

from keras.layers import Conv2D, BatchNormalization, Dropout, Activation, Dense
from keras.models import Model


def relu():
    return hke.siso_keras_module_from_keras_layer_fn(
        lambda: Activation('relu'), {}, name='ReLU')


def batch_normalization():
    return hke.siso_keras_module_from_keras_layer_fn(BatchNormalization, {})


def dropout(h_rate):
    return hke.siso_keras_module_from_keras_layer_fn(Dropout, {"rate": h_rate})


def dense(h_units):
    return hke.siso_keras_module_from_keras_layer_fn(Dense, {"units": h_units})


def conv2d(h_filters, h_kernel_size, stride):
    return hke.siso_keras_module_from_keras_layer_fn(
        lambda filters, kernel_size: Conv2D(
            filters, kernel_size, padding='same', strides=stride), {
                "filters": h_filters,
                "kernel_size": h_kernel_size
            })


def module(h_num_filters, h_kernel_size, h_swap, h_opt_drop, h_drop_rate,
           h_num_repeats):

    return mo.siso_repeat(
        lambda: mo.siso_sequential([
            conv2d(h_num_filters, h_kernel_size, D([1])),
            mo.siso_permutation([relu, batch_normalization], h_swap),
            mo.siso_optional(lambda: dropout(h_drop_rate), h_opt_drop)
        ]), h_num_repeats)


def model(num_classes):
    reduce_fn = lambda: conv2d(D(range(48, 129, 16)), D([3, 5, 7]), D([2]))
    conv_fn = lambda: module(
        D(range(48, 129, 16)), D([3, 5]), D([0, 1]), D([0, 1]), D([0.5, 0.1]),
        D([2**i for i in xrange(6)]))

    return mo.siso_sequential([
        reduce_fn(),
        conv_fn(),
        reduce_fn(),
        conv_fn(),
        dense(D([num_classes]))
    ])
