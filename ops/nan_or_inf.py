import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_eager_execution()

epsilon = 5.96e-08

def tensor_name(name):
  return f"{name}:0" if ":" not in name else name

x = tf.placeholder(shape=[None], dtype=tf.bfloat16, name="x")
y = tf.placeholder(shape=[None], dtype=tf.bfloat16, name="y")

# nan
out1 = tf.divide(x * 0, y, name="out1")

# inf
out2 = tf.divide(x, y, name="out2")

# fix inf
repl_y = tf.where(
    tf.less(y, 0.0),
    tf.ones_like(y, dtype=y.dtype) * (-epsilon),
    tf.ones_like(y, dtype=y.dtype) * epsilon,
    name="repl_y"
)
safe_y = tf.where(tf.less(tf.abs(y), epsilon), repl_y, y, name="safe_y")
out3 = tf.divide(x * 0, safe_y, name="out3")
out4 = tf.divide(x, safe_y, name="out4")

fetches = [
  tensor_name(out1.op.name),
  tensor_name(out2.op.name),
  tensor_name(out3.op.name),
  tensor_name(out4.op.name),
]


# Run session
sess = tf.Session()
inputs = {
  "x:0": [100.0, 200.0, 300.0, 400.0, 500.0],
  "y:0": np.zeros(shape=[5])
}
res = sess.run(fetches, feed_dict=inputs)
print()

for f, r in zip(fetches, res):
  print(f"{f}: {r}")
