import darch.contrib.useful.evaluators.tensorflow.ptb_evaluator as ev
import darch.searchers as se
import darch.contrib.useful.search_spaces.tensorflow.lstm_search_space as ss
import darch.contrib.useful.datasets.ptb_reader as reader

if __name__ == '__main__':
    train_data, valid_data, test_data, vocab = reader.ptb_raw_data('darch/data/simple-examples/data')

    batch_size = 128
    num_steps = 35
    train_x, train_y = reader.ptb_producer(train_data, batch_size, num_steps, name=None)
    val_x, val_y = reader.ptb_producer(valid_data, batch_size, num_steps, name=None)
    data = {'train': (train_x, train_y),
            'val': (val_x, val_y)}
    ss_fn = lambda:ss.ptb_search_space(batch_size, num_steps, vocab)
    searcher = se.RandomSearcher(ss_fn)
    print vocab
    for _ in xrange(10):
        (inputs, outputs, hs, vs, cfg_d) = searcher.sample()
        r = ev.evaluate_fn(inputs, outputs, hs, data, vocab, batch_size, num_steps)
        print vs, r, cfg_d
        searcher.update(r['val_acc'], cfg_d)
