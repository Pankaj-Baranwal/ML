import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import FSRNN
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

hm_epochs = 3
n_classes = 10
batch_size = 128

chunk_size = 28
n_chunks = 28

rnn_size = 128

x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def fsrnn_model(x):
    #Create one Slow and three Fast cells
    slow = tf.contrib.rnn.BasicLSTMCell(10)
    # num_units
    fast = [tf.contrib.rnn.BasicLSTMCell(10),
            tf.contrib.rnn.BasicLSTMCell(10)]

    #Create a single FS-RNN using the cells
    fs_lstm = FSRNN.FSRNNCell(fast, slow)

    #Get initial state and create tf op to run one timestep
    # Args = batch_size, df
    init_state = fs_lstm.zero_state(batch_size, tf.float32)

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    output, final_state = fs_lstm(x, init_state)

    return output

def train_neural_network(x):
    prediction = fsrnn_model(x)
    
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        # (-1, n_chunks, chunk_size) = 1 image at a time.
        print('Accuracy:',accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels}))

train_neural_network(x)