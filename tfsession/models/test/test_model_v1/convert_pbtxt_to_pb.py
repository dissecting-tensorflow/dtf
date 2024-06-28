import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
from google.protobuf import text_format
import numpy as np

tf.disable_eager_execution()

# Let's read our pbtxt file into a Graph protobuf
f = open("test.pbtxt", "r")
graph_protobuf = text_format.Parse(f.read(), tf.GraphDef())

# Import the graph protobuf into our new graph.
tf.import_graph_def(graph_def=graph_protobuf, name="")
graph = tf.get_default_graph()
graph_def = graph.as_graph_def()
# print(str(graph_def))
with tf.gfile.GFile("test.pb", 'wb') as f:
    data = graph_def.SerializeToString()
    f.write(data)
