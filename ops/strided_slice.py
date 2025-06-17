"""
TensorFlow 1.15.0
https://www.tensorflow.org/api_docs/python/tf/strided_slice

tf.strided_slice(
    input_,
    begin,
    end,
    strides=None,
    begin_mask=0,
    end_mask=0,
    ellipsis_mask=0,
    new_axis_mask=0,
    shrink_axis_mask=0,
    var=None,
    name=None
)
"""

MODEL_NAME = "strided_slice_v1"

import os
import sys

import logging
import coloredlogs

# install a handler on the root logger
coloredlogs.install(
    level=logging.DEBUG,
    fmt="%(levelname)s %(message)s"
)

cwd = os.getcwd()
if cwd in sys.path:
  sys.path.remove(cwd)
print(sys.path)

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()

# create the input placeholder
X = tf.placeholder(tf.int32, shape=[4], name="X")
t1 = tf.constant(np.random.randint(100, size=[1, 100]), name="t1", dtype=tf.int32)

# compute output
n = tf.math.reduce_min(X)
output = t1[:, :n]
print(output)

# launch the graph in a session
graph_pbtxt_path = MODEL_NAME + ".pbtxt"
with tf.Session() as sess:
  graph = tf.get_default_graph()
  graph_def = graph.as_graph_def()
  graph_writer = open(graph_pbtxt_path, "w")
  graph_writer.write(str(graph_def))
  graph_writer.close()
  print(graph_pbtxt_path)
  graph_pb_path = MODEL_NAME + ".pb"
  with tf.gfile.GFile(graph_pb_path, 'wb') as f:
    data = graph_def.SerializeToString()
    f.write(data)
  print(graph_pb_path)

