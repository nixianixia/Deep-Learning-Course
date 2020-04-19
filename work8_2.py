import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

x = tf.constant([64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03])
y = tf.constant([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])

t1 = len(x) * tf.reduce_sum(x * y) - tf.reduce_sum(x) * tf.reduce_sum(y)
t2 = len(x) * tf.reduce_sum(tf.pow(x, 2)) - tf.reduce_sum(x) ** 2
w = t1 / t2
b = (tf.reduce_sum(y * w) * tf.reduce_sum(x)) / len(x)
print("w:{}\nb:{}".format(w, b))