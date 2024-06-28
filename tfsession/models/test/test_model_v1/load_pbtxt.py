# Create a fake longseq_attn_softsearch/transpose_5
"""
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
import numpy as np

tf.disable_eager_execution()
c1 = tf.constant(np.random.rand(2, 10, 1), name="longseq_attn_softsearch/transpose_5", dtype=tf.float32)
c2 = tf.constant([4, 2, 2, 2], name="Const_2", dtype=tf.int32)

graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_writer = open("out.json", "w")
graph_writer.write(str(graph_def))
graph_writer.close()
"""

# Const_2
"""
[array([4,  2,  2,  2], dtype=int32)]
"""

# split/split_dim
"""
[1]
"""

import tensorflow.compat.v1 as tf
from google.protobuf import text_format
import numpy as np

# Let's read our pbtxt file into a Graph protobuf
f = open("test.pbtxt", "r")
graph_protobuf = text_format.Parse(f.read(), tf.GraphDef())

# Import the graph protobuf into our new graph.
graph = tf.Graph()
graph.as_default()
tf.import_graph_def(graph_def=graph_protobuf, name="")

inputs = {"fc_hlm_bucket_embedding:0": np.random.rand(2, 10)}
with tf.Session() as sess:
  out1, out2 = sess.run(["longseq_attn_softsearch/transpose_5:0", "split:0"], feed_dict=inputs)
  t1 = tf.convert_to_tensor(np.array(out1))
  t2 = tf.convert_to_tensor(np.array(out2))
  print(t1)
  print(t2)
  
  y = sess.run("longseq_attn_softsearch/mul_18:0", feed_dict=inputs)
  output = tf.convert_to_tensor(np.array(y))
  print(output)
  # print(sess.run(["split/split_dim:0"], feed_dict={"fc_hlm_bucket_embedding:0": x}))
