from nasbench import api
import deep_architect.core as co
import deep_architect.contrib.misc.search_spaces.tensorflow_eager.nasbench_space as nb
INPUT = 'input'
OUTPUT = 'output'
node_op_names = {
    'conv1': 'conv1x1-bn-relu',
    'conv3': 'conv3x3-bn-relu',
    'max3': 'maxpool3x3'
}


class NasbenchEvaluator:

    def __init__(self, tfrecord_file, num_nodes_per_cell, epochs):
        self.nasbench = api.NASBench(tfrecord_file)
        self.num_nodes_per_cell = num_nodes_per_cell
        self.epochs = epochs
        self.ssf = nb.SSF_Nasbench()

    """
    Preconditions: inputs and outputs should be unspecified. vs should be
    the full hyperparameter value list
    """

    def eval(self, vs):
        _, outputs = self.ssf.get_search_space()
        node_ops = [None] * self.num_nodes_per_cell
        node_ops[0] = INPUT
        node_ops[-1] = OUTPUT
        matrix = [[0] * self.num_nodes_per_cell
                  for _ in range(self.num_nodes_per_cell)]
        for i, h in enumerate(
                co.unassigned_independent_hyperparameter_iterator(outputs)):
            h_name = h.get_name().split('-')[-2]
            if 'node' in h_name:
                node_num = int(h_name.split('_')[-1])
                node_ops[node_num + 1] = node_op_names[vs[i]]
            elif 'in' in h_name:
                h_name = h_name.split('_')
                matrix[int(h_name[1])][int(h_name[2])] = vs[i]

            h.assign_value(vs[i])
        model_spec = api.ModelSpec(matrix=matrix, ops=node_ops)
        data = self.nasbench.query(model_spec, epochs=self.epochs)
        return data
