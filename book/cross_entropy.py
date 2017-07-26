import tensorflow as tf
import numpy as np

t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])

# 책에 있는 파이썬의 문법으로만 cross_entropy 구현
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

print(cross_entropy_error(y, t))

# reduction_indices=0 옵션에서 위처럼 0을 전달하면 열합계 , 1을 전달하면 행 합계, 아무것도 전달안하면 전체 합계이다

# log안에 1e-7을 더해주는 이유는 로그 0 이면 마이너스 무한대를 의미해서 nan값이 출력된다

# 텐서플로우의 수학함수로 cross_entropy 구현
cost = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y+1e-7),reduction_indices=0))


# 텐서플로우의 수학함수로 softmax된 값을 cross_entropy
test = tf.nn.softmax(y)
cost_crossentropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(test),reduction_indices=0))

# 텐서플로우의 softmax_cross_entropy 함수 이용
cost_softmax_crossentropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=t))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print(sess.run(cost))

    print("test", sess.run(test))
    print("cost2", sess.run(cost_crossentropy))
    print("cost3", sess.run(cost_softmax_crossentropy))
    # 결국 값은 똑같당.!!