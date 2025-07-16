import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.disable_eager_execution()

def tensor_name(name):
  return f"{name}:0" if ":" not in name else name

epsilon = 5.96e-08

x = tf.placeholder(shape=[None], dtype=tf.bfloat16, name="x")
y = tf.placeholder(shape=[None], dtype=tf.bfloat16, name="y")

repl_y = tf.where(
    tf.less(y, 0.0),
    tf.ones_like(y, dtype=y.dtype) * (-epsilon),
    tf.ones_like(y, dtype=y.dtype) * epsilon,
    name="repl_y"
)
safe_y = tf.where(tf.less(tf.abs(y), epsilon), repl_y, y, name="safe_y")
out = tf.divide(x, safe_y, name="out")

graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
print(graph_def)
print()

inputs = {
  "x:0": np.random.rand(5),
  "y:0": [0.5, -1e-08, 0.0, 0.3, -1e-08]
}

fetches = [
  tensor_name(x.op.name), 
  tensor_name(y.op.name), 
  tensor_name(repl_y.op.name), 
  tensor_name(safe_y.op.name), 
  tensor_name(out.op.name)
]

sess = tf.Session()
res = sess.run(fetches, feed_dict=inputs)

print()
for f, r in zip(fetches, res):
  # t = tf.convert_to_tensor(r)
  print(f"{f}: {r}")

graph_file_path = "./where.pbtxt"
graph_writer = open(graph_file_path, "w")
graph_writer.write(str(graph_def))
graph_writer.close()