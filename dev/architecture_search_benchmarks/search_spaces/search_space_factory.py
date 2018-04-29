import search_spaces.evolution_search_space as ss
import darch.modules as mo

class SSFZoph17(mo.SearchSpaceFactory):
    def __init__(self, search_space, num_classes):
        mo.SearchSpaceFactory.__init__(self)
        self.num_classes = num_classes

        if search_space == 'sp1':
            self.search_space_fn = ss.get_search_space_1
        elif search_space == 'sp2':
            self.search_space_fn = ss.get_search_space_2
        elif search_space == 'sp3':
            self.search_space_fn = ss.get_search_space_3

    def _get_search_space(self):
        inputs, outputs = self.search_space_fn(self.num_classes)
        return inputs, outputs, {}

name_to_search_space_factory_fn = {
    'zoph_sp1': lambda num_classes: SSFZoph17('sp1', num_classes),
    'zoph_sp2': lambda num_classes: SSFZoph17('sp2', num_classes),
    'zoph_sp3': lambda num_classes: SSFZoph17('sp3', num_classes)
}