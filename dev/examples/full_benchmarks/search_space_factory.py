from deep_architect.contrib.misc.search_spaces.tensorflow_eager.nasnet_space import SSF_NasnetA
from deep_architect.contrib.misc.search_spaces.tensorflow_eager.genetic_space import SSF_Genetic
from deep_architect.contrib.misc.search_spaces.tensorflow_eager.nasbench_space import SSF_Nasbench
from deep_architect.contrib.misc.search_spaces.tensorflow.dnn import dnn_net
from deep_architect.contrib.misc.search_spaces.tensorflow_eager.hierarchical_space import (
    flat_search_space, hierarchical_search_space)
from deep_architect.modules import SearchSpaceFactory

name_to_search_space_factory_fn = {
    'zoph_sp1':
    lambda num_classes: SSF_NasnetA(),
    'genetic':
    lambda num_classes: SSF_Genetic(),
    'nasbench':
    lambda num_classes: SSF_Nasbench(),
    'flat':
    lambda num_classes: SearchSpaceFactory(flat_search_space),
    'hierarchical':
    lambda num_classes: SearchSpaceFactory(hierarchical_search_space),
    'dnn':
    lambda num_classes: SearchSpaceFactory(lambda: dnn_net(num_classes))
}
