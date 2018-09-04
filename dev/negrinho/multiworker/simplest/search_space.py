
import deep_architect.modules as mo
import deep_architect.contrib.misc.search_spaces.tensorflow.dnn as css_dnn

class SSF(mo.SearchSpaceFactory):
    def __init__(self, num_classes):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes

    def _get_search_space(self):
        inputs, outputs = css_dnn.dnn_net(self.num_classes)
        return inputs, outputs, {}

num_classes = 10
search_space_factory = SSF(num_classes)
search_space_fn = search_space_factory.get_search_space
