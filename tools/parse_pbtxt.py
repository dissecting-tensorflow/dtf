import os
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
from google.protobuf import text_format
import numpy as np

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--graph", "-g", required=True, help="path to graph")
args = parser.parse_args()
graph_pbtxt_path = args.graph

# Let's read our pbtxt file into a Graph protobuf
f = open(graph_pbtxt_path, "r")
graph_protobuf = text_format.Parse(f.read(), tf.GraphDef())

tf.disable_eager_execution()

# Import the graph protobuf into our new graph.
tf.import_graph_def(graph_def=graph_protobuf, name="")

fetches = [
  "strided_slice_436/stack:0",
  "strided_slice_453/stack_1:0",
  "strided_slice_436/stack_2:0",
]

with tf.Session() as sess:
  outputs = sess.run(fetches, feed_dict=[])
  for i, name in enumerate(fetches):
    value = np.array(outputs[i])
    tensor = tf.convert_to_tensor(value)
    print("{}: {}".format(name, tensor))
    print(np.array(value))
    print("")

"""
Output:
strided_slice_436/stack:0: Tensor("Const:0", shape=(3,), dtype=int32)
[0 0 0]

strided_slice_453/stack_1:0: Tensor("Const_1:0", shape=(3,), dtype=int32)
[0 3 0]

strided_slice_436/stack_2:0: Tensor("Const_2:0", shape=(3,), dtype=int32)
[1 1 1]
"""
