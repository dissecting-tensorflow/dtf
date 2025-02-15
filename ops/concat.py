"""
Python 2.7.16
TensorFlow 1.15.0
"""

MODEL_NAME = "toy_model_v1"

import os
import sys

cwd = os.getcwd()
if cwd in sys.path:
  sys.path.remove(cwd)
print(sys.path)

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()

graph_file_path = "concat.pb"

t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]

# compute output
output = tf.concat([t1, t2], 0, name="Concat_1")

# launch the graph in a session
with tf.Session() as sess:
  # # create the dictionary:
  # d = {
  #   "X1:0": np.random.rand(3, 4),
  #   "X2:0": np.random.rand(3, 4)
  # }

  # # feed it to placeholder a via the dict 
  # print(sess.run(C, feed_dict=d))
  print(sess.run(output, feed_dict={}))

  graph = tf.get_default_graph()
  graph_def = graph.as_graph_def()
  graph_pbtxt_file = graph_file_path + "txt"
  graph_writer = open(graph_pbtxt_file, "w")
  graph_writer.write(str(graph_def))
  graph_writer.close()
  print(graph_pbtxt_file)
  with tf.gfile.GFile(graph_file_path, 'wb') as f:
    data = graph_def.SerializeToString()
    f.write(data)
  print(graph_file_path)
