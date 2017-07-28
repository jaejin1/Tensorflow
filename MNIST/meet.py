import tensorflow as tf
import matplotlib.pyplot as plt
import random

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

'''
tf.Variable()
tf.get_variable(name, shape=None, dtype=tf.float32, initializer=None,
       regularizer=None, trainable=True, collections=None)
tf.Variable처럼 직접 값을 전달하는 대신 initializer를 사용합니다. initializer는 모양(shape)을 가져와서 텐서를 제공하는 함수입니다. 여기 TensorFlow에서 사용 가능한 몇 개의 initializer가 있습니다.
'''

hypothesis = tf.nn.softmax(tf.matmul(X, W)+ b)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15                # 트레이닝 에폭이 15
batch_size = 100                    # 배치 사이즈가 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):   # 에폭 반복 15번
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)        # Mnist train 개수를 배치 사이즈로 나눔 ..

        for i in range(total_batch):                                    # 위의 나눈것을 반복함
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)     # x 와 y의 값을 가져온다
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y:batch_ys})         # cost , optimizer 돌림 .
            avg_cost += c / total_batch


            print('Epoch : ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))

    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
    print("Prediction: ", sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1]}))

    plt.imshow(mnist.test.images[r:r+1].reshape(28,28), cmap='Greys', interpolation='nearest')
    plt.show()


