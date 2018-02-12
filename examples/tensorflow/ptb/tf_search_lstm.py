import numpy as np
import tensorflow as tf
from search_models import get_darch_models
from utils import vocabulary, get_batch, get_size


BATCH_SIZE = 1000
NUM_STEPS = 4  # the number of unrolled steps of LSTM
NUM_EPOCHS = 11
VALID_EPOCHS = 10


x = tf.placeholder(dtype=tf.int32, name='x', shape=(None, NUM_STEPS))
y = tf.placeholder(dtype=tf.int32, name='y', shape=(None, NUM_STEPS))
output_gold = tf.one_hot(y, depth=vocabulary, axis=-1)

models_output = get_darch_models(x)

for n_model, output in enumerate(models_output):
    loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=output_gold,
        logits=output
    )
    red_loss = tf.reduce_sum(loss)

    lr = 20
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(red_loss)

    best_valid = np.inf
    with tf.Session() as session:
        try:
            tf.global_variables_initializer().run()
            print(n_model, [m.values() for m in session.graph.get_operations()])

            for step in range(NUM_EPOCHS):
                batch_nr = get_size('train') // (BATCH_SIZE * NUM_STEPS)
                loss_total = 0
                for batch in range(batch_nr):
                    _x, _y = get_batch('train', BATCH_SIZE, NUM_STEPS)
                    feed_dict = {x: _x, y: _y}

                    _, loss_val = session.run([optimizer, red_loss], feed_dict=feed_dict)
                    loss_total += loss_val
                print(n_model, loss_total / (batch_nr * BATCH_SIZE * NUM_STEPS))

                if (step + 1) % VALID_EPOCHS == 0:
                    batch_nr = get_size('valid') // (BATCH_SIZE * NUM_STEPS)
                    loss_total = 0
                    for batch in range(batch_nr):
                        _x, _y = get_batch('valid', BATCH_SIZE, NUM_STEPS)
                        feed_dict = {x: _x, y: _y}

                        loss_val = session.run(red_loss, feed_dict=feed_dict)
                        loss_total += loss_val
                    print(n_model, 'valid loss:', loss_total / (batch_nr * BATCH_SIZE * NUM_STEPS))
                    if loss_total < best_valid:
                        best_valid = loss_total
                    else:
                        lr /= 4.0
                        optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(red_loss)

        except KeyboardInterrupt:
            print(n_model, 'Exiting from training early')

        batch_nr = get_size('test') // (BATCH_SIZE * NUM_STEPS)
        loss_total = 0
        for batch in range(batch_nr):
            _x, _y = get_batch('test', BATCH_SIZE, NUM_STEPS)
            feed_dict = {x: _x, y: _y}

            loss_val = session.run(red_loss, feed_dict=feed_dict)
            loss_total += loss_val
        print(n_model, 'test loss:', loss_total / (batch_nr * BATCH_SIZE * NUM_STEPS))
