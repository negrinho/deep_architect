from deep_architect.contrib.regularized_evolution.search_space.evolution_search_space import SSFZoph17
from .enas_search_space_eager import SSFEnasnetEager

name_to_search_space_factory_fn = {
    'zoph_sp1': lambda num_classes: SSFZoph17('sp1', num_classes),
    'zoph_sp2': lambda num_classes: SSFZoph17('sp2', num_classes),
    'zoph_sp3': lambda num_classes: SSFZoph17('sp3', num_classes),
    'enas_eager': lambda num_classes: SSFEnasnetEager(num_classes, 12, 36)
}
