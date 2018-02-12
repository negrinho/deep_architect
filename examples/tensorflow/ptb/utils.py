import numpy as np
from tensorflow.python.framework.errors_impl import NotFoundError
from reader import ptb_raw_data

# to get data do
# $ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
# $ tar xvf simple-examples.tgz
try:
    train_data, valid_data, test_data, vocabulary = ptb_raw_data('simple-examples/data')
except NotFoundError:
    train_data, valid_data, test_data, vocabulary = ptb_raw_data(
        'examples/tensorflow/modified_examples/ptb/simple-examples/data')
train_data = np.array(train_data)
valid_data = np.array(valid_data)
test_data = np.array(test_data)
train_it = valid_it = test_it = 0


#train_data = train_data[:10000]  # TODO just for testing

def get_data(name):
    if name == 'train':
        data = train_data
    elif name == 'valid':
        data = valid_data
    elif name == 'test':
        data = test_data
    else:
        raise ValueError('name must be one of ["train", "valid", "test"]')
    return data


def get_it(name):
    if name == 'train':
        return train_it
    elif name == 'valid':
        return valid_it
    elif name == 'test':
        return test_it


def update_it(name, newval):
    global train_it, valid_it, test_it
    if name == 'train':
        train_it = newval
    elif name == 'valid':
        valid_it = newval
    elif name == 'test':
        test_it = newval


def get_batch(name, batch_size, num_steps):
    data = get_data(name)
    it = get_it(name)

    if len(data) >= it + batch_size * num_steps + 1:
        x = data[it:it + batch_size * num_steps].reshape([batch_size, num_steps])
        y = data[it+1:it + batch_size * num_steps + 1].reshape([batch_size, num_steps])
        update_it(name, it + batch_size * num_steps)
        return x, y
    else:
        update_it(name, 0)
        return get_batch(name, batch_size, num_steps)


def get_size(name):
    return len(get_data(name))
