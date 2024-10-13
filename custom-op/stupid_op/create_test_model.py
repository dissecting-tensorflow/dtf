"""
Python 2.7.16 or 3.7.3
TensorFlow 1.15.0
"""

import os
import sys
import json

cwd = os.getcwd()
if cwd in sys.path:
  sys.path.remove(cwd)
print(sys.path)

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
import numpy as np

solib = "/path/to/libstupid_op.so"
print("Loading {}".format(solib))
custom_op = tf.load_op_library(solib)
print(json.dumps(dir(custom_op), indent=2))
print()

tf.disable_eager_execution()

graph_file_path = "./faulty_model_v1.pb"

# create the input placeholder
# create the input placeholder
X = tf.placeholder(tf.int8, shape=[1], name="X")

A = tf.constant(np.random.rand(3, 4), name="matrixA", dtype=tf.float32)
# X = tf.placeholder(tf.float32, shape=[None, 5], name="X")
# B = tf.constant(np.random.rand(5, 5), name="matrixB", dtype=tf.float32)

# create MatMul node
# C = tf.matmul(A, B, name="matrixC")

# launch the graph in a session
with tf.Session() as sess:
  # # create the dictionary:
  # d = {
  #   "matrixA:0": np.random.rand(20480, 20480),
  #   "matrixB:0": np.random.rand(20480, 20480)
  # }

  # # feed it to placeholder a via the dict 
  # print(sess.run(C, feed_dict=d))

  graph = tf.get_default_graph()
  graph_def = graph.as_graph_def()

  # # Faulty MatMul node
  # matmul_node_def = tf.NodeDef()
  # matmul_node_def.name = "Output"
  # matmul_node_def.op = "MatMul"
  # matmul_node_def.input.append("matrixA")
  # matmul_node_def.input.append("matrixB")
  # matmul_node_def.attr["T"].CopyFrom(tf.AttrValue(type=dtypes.as_dtype(tf.float32).as_datatype_enum))
  # matmul_node_def.attr["transpose_a"].CopyFrom(tf.AttrValue(b=dtypes.as_dtype(tf.bool).as_datatype_enum))
  # matmul_node_def.attr["transpose_b"].CopyFrom(tf.AttrValue(b=dtypes.as_dtype(tf.bool).as_datatype_enum))
  # graph_def.node.append(matmul_node_def)

  # Faulty Stupid node
  stupid_node_def = tf.NodeDef()
  stupid_node_def.name = "Output"
  stupid_node_def.op = "Stupid"
  stupid_node_def.input.append("X")
  stupid_node_def.input.append("matrixA")
  stupid_node_def.attr["T0"].CopyFrom(tf.AttrValue(type=dtypes.as_dtype(tf.int8).as_datatype_enum))
  stupid_node_def.attr["T1"].CopyFrom(tf.AttrValue(type=dtypes.as_dtype(tf.float32).as_datatype_enum))
  graph_def.node.append(stupid_node_def)

  graph_pbtxt_path = graph_file_path + "txt"
  graph_writer = open(graph_pbtxt_path, "w")
  graph_writer.write(str(graph_def))
  graph_writer.close()
  print(graph_pbtxt_path)
  with tf.gfile.GFile(graph_file_path, 'wb') as f:
    data = graph_def.SerializeToString()
    f.write(data)
  print(graph_file_path)
