import json
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

data = tf.constant([[1, 2, 3], [4, 5, 6]])
axis = tf.rank(data) - 1
out = tf.math.reduce_sum(data, axis=axis, name="Sum_1")
print(f"op type: {out.op.type}")
