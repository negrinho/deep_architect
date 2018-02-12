from __future__ import print_function
from six.moves import xrange

import tests.search_space as ss
import tests.evaluator as ev
import tests.dataset as ds

import darch.searchers as se
import darch.core as co
import darch.utils as ut

import numpy as np
from pprint import pprint
import reduced_toolbox.tb_logging as tb_lg

# checks that the hyperparameters are iterated in the same order in both calls.
def test_ordering():
    (inputs, outputs, hs) = ss.search_space_fn()
    v_hist = se.random_specify(outputs.values(), hs.values())
    feats  = se.extract_features(inputs, outputs, hs)
    print()
    (inputs, outputs, hs) = ss.search_space_fn()
    se.specify(outputs.values(), hs.values(), v_hist)
    new_feats  = se.extract_features(inputs, outputs, hs)

    for (x, y) in zip(feats, new_feats):
        pprint(x)
        pprint(y)
        print()
        assert tuple(x) == tuple(y)

def test_memory():
    num_samples = 128
    mem = tb_lg.MemoryTracker()

    # does the memory change by just creating different search spaces.
    print(mem.memory_total())
    for _ in xrange(num_samples):
        (inputs, outputs, hs) = ss.search_space_fn()
        v_hist = se.random_specify(outputs.values(), hs.values())
    print(mem.memory_total())

    # do the memory change by actually by compiling and forwarding multiple ones.
    (train_dataset, val_dataset, test_dataset, in_d, num_classes) = ds.load_data(
        'mnist')

    evaluator = ev.ClassifierEvaluator(train_dataset, val_dataset, in_d, num_classes,
        '.', 'mnist',
        output_to_terminal=True, test_dataset=test_dataset,
        max_minutes_per_model=0.05, batch_size_init=128)

    print(mem.memory_total())
    for _ in xrange(num_samples):
        (inputs, outputs, hs) = ss.search_space_fn()
        v_hist = se.random_specify(outputs.values(), hs.values())
        val = evaluator.eval(inputs, outputs, hs)
    print(mem.memory_total())

def test_visualization():
    num_samples = 8
    for _ in xrange(num_samples):
        (inputs, outputs, hs) = ss.search_space_fn()
        v_hist = se.random_specify(outputs.values(), hs.values())
        ut.draw_graph(outputs.values(), True, True)

def test_speed():
    num_samples = 32

    t = tb_lg.TimeTracker()
    t.time_since_last()

    t_inst = t_spec = t_feats = 0.0
    for _ in xrange(num_samples):
        (inputs, outputs, hs) = ss.search_space_fn()
        t_inst += t.time_since_last()
        v_hist = se.random_specify(outputs.values(), hs.values())
        t_spec += t.time_since_last()
        feats = se.extract_features(inputs, outputs, hs)
        t_feats += t.time_since_last()

    print 'Instantiate search space: %0.2e secs.' % (t_inst / num_samples)   
    print 'Specify: %0.2e secs.' % (t_spec / num_samples)
    print 'Extract features: %0.2e secs.' % (t_feats / num_samples)    
    pprint(feats)


if __name__ == '__main__':
    # test_ordering()
    # test_memory()
    # test_visualization()
    test_speed()


# TODO: the iterator should not realize different orderings if it is reset
# mid-iteration. this is not important if we only use the iterator on a
# single pass to instantiate the model.

# TODO: write tests to make sure that certains things are deterministic.
# that I can save things to disk and retrieve them and so on.

# TODO: profile for memory and speed. what are the bottlenecks of
# running on the CPU and GPU.

# TODO: make sure that I can start and restart and still works the same way than
# doing things from the beginning.
