import tensorflow as tf

x_data = [1,2,3]
y_data = [1,2,3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

print(X)
print(Y)

# W와 X가 행렬이 아니므로 그냥 곱을 사용함 tf.matmul 이 아닌..
hypothesis = W * X + b


cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 텐서플로우에 기본적으로 저장되어있는 경사하강법 사용함
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

train_op = optimizer.minimize(cost)

# 세션 생성후 초기화
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, sess.run(W), sess.run(b))


    print("\n=== Test ===")
    print("x: 5, Y:", sess.run(hypothesis, feed_dict={X : 5}))
    print("X: 2.5, Y: ", sess.run(hypothesis, feed_dict={X: 2.5}))