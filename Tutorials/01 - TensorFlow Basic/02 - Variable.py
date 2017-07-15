import tensorflow as tf

X = tf.placeholder(tf.float32, [None, 3])
print(X)

x_data = [[1,2,3], [4,5,6]]

W = tf.Variable(tf.random_normal([3,2]))
b = tf.Variable(tf.random_normal([2,1]))

# 랜덤으로 초기화

expr = tf.matmul(X, W) + b
a = expr - 1

sess = tf.Session()

sess.run(tf.global_variables_initializer())
# 위의 Variable의 값들을 초기화하기위해 한번 실행해야함.

print("=== x_data === ")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")

for step in range(100):
    _, result = sess.run([a,expr], feed_dict={X: x_data})
    result2 = sess.run(expr, feed_dict={X: x_data})


print(result)
print(result2)
sess.close()

