import deep_architect.modules as mo
import deep_architect.contrib.misc.search_spaces.tensorflow.dnn as css_dnn


def get_dnn_search_space_fn(num_classes):
    return mo.SearchSpaceFactory(
        lambda: css_dnn.dnn_net(num_classes)).get_search_space


num_classes = 10
search_space_fn = mo.SearchSpaceFactory(
    get_dnn_search_space_fn(num_classes)).get_search_space
