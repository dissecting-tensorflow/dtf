import numpy as np
import tensorflow.compat.v1 as tf

frozen_graph = "./gpu/graph.pb"

# Add an Op to initialize variables
init_op = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
  with tf.gfile.GFile(frozen_graph, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
  
  # initialize variables
  sess.run(init_op)

  # create the dictionary:
  d = {"X:0": np.random.rand(3, 5)}

  # # feed it to placeholder a via the dict 
  print(sess.run(fetches=["Sigmoid:0"], feed_dict=d))

