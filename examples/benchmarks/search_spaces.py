import deep_architect.contrib.misc.search_spaces.tensorflow.dnn as css_dnn
import deep_architect.contrib.misc.search_spaces.tensorflow.cnn2d as css_cnn2d
from deep_architect.contrib.misc.search_spaces.tensorflow.common import D
import deep_architect.modules as mo


def get_dnn_search_space_fn(num_classes):
    return mo.SearchSpaceFactory(
        lambda: css_dnn.dnn_net(num_classes)).get_search_space


def get_conv_search_space_fn(num_classes):

    def search_space_fn():
        h_num_spatial_reductions = D([2, 3, 4])
        h_pool_op = D(['max', 'avg'])
        return mo.siso_sequential([
            css_cnn2d.conv_net(h_num_spatial_reductions),
            css_cnn2d.spatial_squeeze(h_pool_op, D([num_classes]))
        ])

    return mo.SearchSpaceFactory(search_space_fn).get_search_space


name_to_search_space_fn = {
    'dnn': get_dnn_search_space_fn,
    'conv': get_conv_search_space_fn,
}
