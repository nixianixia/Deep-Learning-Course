import tensorflow as tf

a = tf.Variable([1,2,3],tf.float32)
print(a)
with tf.GradientTape() as tape:
	print(type(tape))

