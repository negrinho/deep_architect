
import darch.core as co
import darch.contrib.search_spaces.tensorflow.cnn2d as css_cnn2d
import darch.hyperparameters as hp
import darch.modules as mo
import darch.searchers as se
import darch.visualization as vi
import darch.helpers.tensorflow as htf
import tensorflow as tf

D = hp.Discrete

def get_search_space():
    co.Scope.reset_default_scope()
    h_num_spatial_reductions = D([1, 2, 4])
    inputs, outputs = mo.siso_sequential([
        mo.empty(),
        css_cnn2d.conv_net(h_num_spatial_reductions),
        mo.empty()
        ])
    return inputs, outputs, {}

def test_search_space():
    searcher = se.RandomSearcher(get_search_space)

    for idx in xrange(3):
        inputs, outputs, hyperps, _, _ = searcher.sample()
        vi.draw_graph(outputs.values(), True, True, graph_name='graph%d' % idx,
            print_to_screen=False, out_folderpath='./temp')

        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        co.forward({inputs['In'] : x})
        print htf.get_num_trainable_parameters()

if __name__ == '__main__':
    test_search_space()

# TODO: add more tests for the different search spaces.



