import deep_architect.core as co
from deep_architect.searchers.common import Searcher, random_specify_hyperparameter


class SpecificSearcher(Searcher):

    def __init__(self, search_space_fn):
        Searcher.__init__(self, search_space_fn)

    def sample(self):
        hparams = {
            'normal_op_0_0': 'avg3',
            'normal_op_0_1': 'max3',
            'normal_op_1_0': 'none',
            'normal_op_1_1': 'avg3',
            'normal_op_2_0': 'depth_sep5',
            'normal_op_2_1': 'depth_sep3',
            'normal_op_3_0': 'depth_sep3',
            'normal_op_3_1': 'none',
            'normal_op_4_0': 'avg3',
            'normal_op_4_1': 'depth_sep3',
            'reduction_op_0_0': 'avg3',
            'reduction_op_0_1': 'depth_sep3',
            'reduction_op_1_0': 'max3',
            'reduction_op_1_1': 'max3',
            'reduction_op_2_0': 'max3',
            'reduction_op_2_1': 'depth_sep3',
            'reduction_op_3_0': 'depth_sep7',
            'reduction_op_3_1': 'avg3',
            'reduction_op_4_0': 'depth_sep3',
            'reduction_op_4_1': '1x7_7x1',
            'normal_in_0_0': 0,
            'normal_in_0_1': 0,
            'normal_in_1_0': 0,
            'normal_in_1_1': 1,
            'normal_in_2_0': 2,
            'normal_in_2_1': 1,
            'normal_in_3_0': 2,
            'normal_in_3_1': 1,
            'normal_in_4_0': 4,
            'normal_in_4_1': 0,
            'reduction_in_0_0': 0,
            'reduction_in_0_1': 1,
            'reduction_in_1_0': 1,
            'reduction_in_1_1': 0,
            'reduction_in_2_0': 0,
            'reduction_in_2_1': 2,
            'reduction_in_3_0': 0,
            'reduction_in_3_1': 1,
            'reduction_in_4_0': 3,
            'reduction_in_4_1': 0,
        }
        ins, outs = self.search_space_fn()
        hyperp_value_lst = []
        for h in co.unassigned_independent_hyperparameter_iterator(
                outs.values()):
            if len(h.vs) == 1:
                v = random_specify_hyperparameter(h)
            else:
                name = h.get_name().split('.')[1].split('-')[0]
                if name in hparams:
                    v = hparams[name]
                    h.assign_value(v)
                else:
                    print(name + ' not found')
                    print(h.vs)
                    v = random_specify_hyperparameter(h)
            hyperp_value_lst.append(v)
        return ins, outs, hyperp_value_lst, {}

    def update(self, val, cfg_d):
        pass

    def get_searcher_state_token(self):
        pass

    def save_state(self, folder_name):
        pass

    def load(self, folder_name):
        pass
