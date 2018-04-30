from .evolution_search_space import SSFZoph17
from .nas_search_space import SSFNasnet

name_to_search_space_factory_fn = {
    'zoph_sp1': lambda num_classes: SSFZoph17('sp1', num_classes),
    'zoph_sp2': lambda num_classes: SSFZoph17('sp2', num_classes),
    'zoph_sp3': lambda num_classes: SSFZoph17('sp3', num_classes),
    'nasnet': lambda num_classes: SSFNasnet(num_classes)
}