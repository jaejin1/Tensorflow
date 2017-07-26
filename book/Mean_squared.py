import tensorflow as tf
import numpy as np

def mean_squared_error(y,t):
    return 0.5 * np.sum((y-t)**2)

t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])


print("sss",mean_squared_error(y,t))



cost = tf.reduce_mean(tf.square(y-t))  # reduce_mean은 평균내주는거임 따라서 0.5 가 아닌 0.1을 곱하는것과 같다 10개이기 때문에



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(cost))

