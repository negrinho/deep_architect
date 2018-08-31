
import deep_architect.contrib.useful.search_spaces.tensorflow.dnn as css_dnn
import deep_architect.contrib.useful.search_spaces.tensorflow.cnn2d as css_cnn2d
from deep_architect.contrib.useful.search_spaces.tensorflow.common import D
import deep_architect.modules as mo

def get_hyperps():
    return {
        'learning_rate_init' : D([1e-2, 1e-3, 1e-4, 1e-5]),
        'optimizer_type' : D(['adam', 'sgd_mom'])}

class SSF0(mo.SearchSpaceFactory):
    def __init__(self, num_classes):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes

    def _get_search_space(self):
        inputs, outputs = css_dnn.dnn_net(self.num_classes)
        return inputs, outputs, get_hyperps()

class SSF1(mo.SearchSpaceFactory):
    def __init__(self, num_classes):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes

    def _get_search_space(self):
        h_num_spatial_reductions = D([2, 3, 4])
        h_pool_op = D(['max', 'avg'])
        inputs, outputs = mo.siso_sequential([
            css_cnn2d.conv_net(h_num_spatial_reductions),
            css_cnn2d.spatial_squeeze(h_pool_op, D([self.num_classes]))])
        return inputs, outputs, get_hyperps()

name_to_search_space_fn = {
    'dnn' : lambda num_classes: SSF0(num_classes).get_search_space,
    'conv' : lambda num_classes: SSF1(num_classes).get_search_space,
}