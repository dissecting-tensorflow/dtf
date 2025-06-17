"""
Python 2.7.16
TensorFlow 1.15.0
"""

import tensorflow._api.v2.compat.v1 as tf
from tensorflow.python.framework import graph_util
import numpy as np

tf.disable_eager_execution()

graph_file_path = "./gpu/graph.pb"

# create the input placeholder
X = tf.placeholder(tf.float32, shape=[None, 5], name="X")
weight_initer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)

# create network parameters
W = tf.get_variable(name="Weight", dtype=tf.float32, shape=[5, 7], initializer=weight_initer)
bias_initer = tf.constant(0., shape=[7], dtype=tf.float32)
b = tf.get_variable(name="Bias", dtype=tf.float32, initializer=bias_initer)

# create MatMul node
x_w = tf.matmul(X, W, name="MatMul")

# create Add node
x_w_b = tf.add(x_w, b, name="Add")

# create Sigmoid node
h = tf.nn.sigmoid(x_w_b, name="Sigmoid") 

# Add an Op to initialize variables
init_op = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
  # initialize variables
  sess.run(init_op)

  # # create the dictionary:
  # d = {X: np.random.rand(100, 784)}

  # # feed it to placeholder a via the dict 
  # print(sess.run(h, feed_dict=d))

  graph = tf.get_default_graph()
  input_graph_def = graph.as_graph_def()
  output_node_names = ['Sigmoid']
  output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names)
  graph_writer = open(graph_file_path + ".json", "w")
  graph_writer.write(str(output_graph_def))
  graph_writer.close()
  with tf.gfile.GFile(graph_file_path, 'wb') as f:
      data = output_graph_def.SerializeToString()
      f.write(data)


# return pb_filepath