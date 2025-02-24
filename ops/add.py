import tensorflow as tf

x = tf.constant([[1, 1, 1], [1, 1, 1]])
out = tf.reduce_sum(x, -1).numpy()
print(out)