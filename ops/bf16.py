import tensorflow as tf
x = tf.constant([2., 0.5, 1.5], dtype=tf.bfloat16)
tf.math.rsqrt(x)
