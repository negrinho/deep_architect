from dev.regularized_evolution.search_space.evolution_search_space import SSFZoph17
# from deep_architect.contrib.enas.search_space.enas_search_space import SSFEnasnet
import dev.regularized_evolution.search_space.dnn as css_dnn
import deep_architect.modules as mo


class SSF0(mo.SearchSpaceFactory):

    def __init__(self, num_classes):
        mo.SearchSpaceFactory.__init__(
            self, lambda: css_dnn.dnn_net(num_classes))


name_to_search_space_factory_fn = {
    'zoph_sp1': lambda num_classes: SSFZoph17('sp1', num_classes),
    'zoph_sp2': lambda num_classes: SSFZoph17('sp2', num_classes),
    'zoph_sp3': lambda num_classes: SSFZoph17('sp3', num_classes),
    # 'enas': lambda num_classes: SSFEnasnet(num_classes, 12, 36),
    # 'enas_noshare': lambda num_classes: SSFEnasnet(num_classes, 12, 36, False),
    'dnn': SSF0
}
