import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

a = tf.constant(2)
b = tf.constant(3)
p = tf.stack([a, b], axis=0)