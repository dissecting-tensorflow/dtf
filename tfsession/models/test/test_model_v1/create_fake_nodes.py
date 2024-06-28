import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
import numpy as np

tf.disable_eager_execution()
c1 = tf.constant(np.random.rand(2, 4, 1), name="longseq_attn_softsearch/transpose_5", dtype=tf.float32)
c2 = tf.constant([4, 2, 2, 2], name="Const_2", dtype=tf.int32)

graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
graph_writer = open("fake_nodes.json", "w")
graph_writer.write(str(graph_def))
graph_writer.close()
