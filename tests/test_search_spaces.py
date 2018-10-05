import deep_architect.core as co
import deep_architect.contrib.misc.search_spaces.tensorflow.cnn2d as css_cnn2d
import deep_architect.hyperparameters as hp
import deep_architect.modules as mo
import deep_architect.searchers as se
import deep_architect.visualization as vi
import deep_architect.helpers.tensorflow as htf
import tensorflow as tf

D = hp.Discrete


def get_search_space():
    co.Scope.reset_default_scope()
    h_num_spatial_reductions = D([1, 2, 4])
    inputs, outputs = mo.siso_sequential([
        mo.identity(),
        css_cnn2d.conv_net(h_num_spatial_reductions),
        mo.identity()
    ])
    return inputs, outputs, {}


def test_search_space():
    searcher = se.RandomSearcher(get_search_space)

    for idx in xrange(3):
        inputs, outputs, _, _ = searcher.sample()
        vi.draw_graph(
            outputs.values(),
            True,
            True,
            graph_name='graph%d' % idx,
            print_to_screen=False,
            out_folderpath='./temp')

        x = tf.placeholder(tf.float32, [None, 32, 32, 3])
        co.forward({inputs['In']: x})
        print htf.get_num_trainable_parameters()


if __name__ == '__main__':
    test_search_space()

# TODO: add more tests for the different search spaces.
