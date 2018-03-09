
import sys
sys.path.append('../')

import data_representation as dr
import evaluators as ev
import search_space as ss
import darch.searchers as se
import search_spaces.dnn as ssd
import darch.modules as mo
import darch.hyperparameters as hp

D = hp.Discrete

(Xtrain, ytrain, Xval, yval, Xtest, ytest) = dr.load_mnist('..', True)
train_dataset = dr.InMemoryDataset(Xtrain, ytrain, False)
val_dataset = dr.InMemoryDataset(Xval, yval, False)
num_classes = 10
evaluator = ev.SimpleClassifierEvaluator(train_dataset, val_dataset, num_classes, 
    './out', log_output_to_terminal=True, learning_rate_init=1e-4, 
    max_num_training_epochs=10)

def get_search_space_fn(num_classes):
    def search_space_fn():
        # some parameters.
        num_filters_lst = [32, 48, 64, 96, 128]
        filter_width_lst = [1, 3, 5]
        num_hidden_lst = [256, 512, 1024]
        keep_ps = [0.25, 0.5, 0.75]

        conv2d_fn = lambda: ssd.conv2d_simplified(D(num_filters_lst), D(filter_width_lst), 1)
        max_pool2d_fn = lambda: ssd.max_pool2d(D([2]), D([2]))

        (inputs, outputs) = mo.siso_sequential([
            conv2d_fn(), ssd.relu(), max_pool2d_fn(),
            conv2d_fn(), ssd.relu(), max_pool2d_fn(), 
            ssd.affine_simplified(D(num_hidden_lst)), ssd.relu(),
            ssd.dropout(D(keep_ps)), ssd.affine_simplified(D([num_classes]))])
        return inputs, outputs, {}
    return search_space_fn

searcher = se.RandomSearcher(get_search_space_fn(num_classes))
for _ in xrange(16):
    (inputs, outputs, hs, vs, cfg_d) = searcher.sample()
    r = evaluator.eval(inputs, outputs, hs)
    print vs, r, cfg_d
    searcher.update(r, cfg_d)

### what is nice about this, is that there is a very high level way of building 
# the experiment.

# this loop is very compact.

# could do a cifar-10 test.num_classes
# simple interface to get data.

# basically, how do you profile search spaces.