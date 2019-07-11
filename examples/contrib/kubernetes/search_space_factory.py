from deep_architect.contrib.misc.search_spaces.tensorflow_eager.nasnet_space import SSF_NasnetA
from deep_architect.contrib.misc.search_spaces.tensorflow_eager.genetic_space import SSF_Genetic
from deep_architect.contrib.misc.search_spaces.tensorflow_eager.nasbench_space import SSF_Nasbench
from deep_architect.contrib.misc.search_spaces.tensorflow_eager.hierarchical_space import flat_search_space, hierarchical_search_space
from deep_architect.contrib.misc.search_spaces.tensorflow import dnn
import deep_architect.modules as mo

name_to_search_space_factory_fn = {
    'dnn':
    lambda num_classes: mo.SearchSpaceFactory(lambda: dnn.dnn_net(num_classes)),
    'nasnet':
    lambda num_classes: SSF_NasnetA(),
    'hierarchical':
    lambda num_classes: mo.SearchSpaceFactory(lambda: hierarchical_search_space(
        num_classes)),
    'flat':
    lambda num_classes: mo.SearchSpaceFactory(lambda: flat_search_space(
        num_classes)),
    'genetic':
    lambda num_classes: SSF_Genetic(),
    'nasbench':
    lambda num_classes: SSF_Nasbench()
}
