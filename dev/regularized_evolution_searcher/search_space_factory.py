import evolution_search_space as ss
import darch.core as co

class SearchSpaceFactory:
    def __init__(self, search_space, num_classes):
        self.num_classes = num_classes
        if search_space == 'sp1':
            self.search_space_fn = ss.get_search_space_1
        elif search_space == 'sp2':
            self.search_space_fn = ss.get_search_space_2
        elif search_space == 'sp3':
            self.search_space_fn = ss.get_search_space_3

    def get_search_space(self):
        co.Scope.reset_default_scope()
        inputs, outputs = self.search_space_fn(self.num_classes)
        return inputs, outputs, {}