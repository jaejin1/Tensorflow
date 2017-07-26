import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import pprint

pp = pprint.PrettyPrinter(indent= 4)
sess = tf.InteractiveSession()

h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]
"""
with tf.variable_scope('one_cell') as scope:
    hidden_size = 2
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
    print(cell.output_size, cell.state_size)

    x_data = np.array([[h]], dtype = np.float32)
    pp.pprint(x_data)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    sess.run(tf.global_variables_initializer())

    pp.pprint(outputs.eval())
"""
"""
with tf.variable_scope('two_sequances') as scope:
    hidden_size = 2
    cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)


    x_data = np.array([[h,e,l,l,o]], dtype = np.float32)
    print(x_data.shape)

    pp.pprint(x_data)
    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    sess.run(tf.global_variables_initializer())

    pp.pprint(outputs.eval())
"""
"""
with tf.variable_scope('3_batches') as scope:
    x_data = np.array([[h, e, l, l, o],
                       [e ,o ,l ,l ,l],
                       [l, l, e, e, l]], dtype=np.float32)
    pp.pprint(x_data)
    print(x_data.shape)
    hidden_size = 2
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)


    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    sess.run(tf.global_variables_initializer())

    pp.pprint(outputs.eval())
"""
with tf.variable_scope('initial_state') as scope:
    batch_size = 3
    sequence_length = 5
    input_dim = 3


    x_data = np.array([[h, e, l, l, o],
                       [e ,o ,l ,l ,l],
                       [l, l, e, e, l]], dtype=np.float32)
    x_data = np.arange(45, dtype=np.float32).reshape(batch_size, sequence_length, input_dim)
    pp.pprint(x_data)


    hidden_size = 2
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size)



    outputs, _states = tf.nn.dynamic_rnn(cell, x_data, dtype=tf.float32)

    sess.run(tf.global_variables_initializer())

    pp.pprint(outputs.eval())

