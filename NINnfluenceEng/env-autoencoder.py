# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import nn_engine_input


env_data_batch, ego_data_batch,target_data_batch = nn_engine_input.get_batch()

# Parameters
learning_rate = 0.05
training_epochs = env_data_batch.shape[0]
batch_size = 256
display_step = 1000
examples_to_show = 10

# Network Parameters
n_hidden_1 = 100 # 1st layer num features
n_hidden_2 = 50 # 2nd layer num features
n_input = 210 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [1, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    print('eprochs : ', training_epochs)
    for epoch in range(training_epochs):
        # Loop over all batches
            # Run optimization op (backprop) and cost op (to get loss value)
        feed_value = env_data_batch[epoch]
        feed_value = np.reshape(feed_value,newshape=(1,210))
        _, c = sess.run([optimizer, cost], feed_dict={X: feed_value})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")
    env_eval_data_batch, ego_eval_data_batch,target_eval_data_batch = nn_engine_input.get_evaluation_batch()
    feed_value = env_eval_data_batch[10]
    feed_value = np.reshape(feed_value,newshape=(1,210))
    # Applying encode and decode over test set
    encode_decode = sess.run(
        y_pred, feed_dict={X: feed_value})
    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    feed_value = np.reshape(feed_value, newshape = (5,6,7))
    encode_decode = np.reshape(encode_decode, newshape = (5,6,7))
    for i in range(7):
        a[0][i].imshow(feed_value[:,:,i])
        a[1][i].imshow(feed_value[:,:,i])
    f.show()
    plt.draw()
    plt.waitforbuttonpress()