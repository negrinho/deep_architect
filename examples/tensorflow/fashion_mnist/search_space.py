from deep_architect.contrib.deep_learning_backend import tf_ops
from deep_architect.helpers import tensorflow as htf
import deep_architect.modules as mo
from deep_architect.hyperparameters import D
def resnet_block(stride):
    res_in, res_out = mo.empty()
    ins, outs = mo.siso_sequential([
        (res_in, res_out),
        tf_ops.conv2d(D([]))
    ])
