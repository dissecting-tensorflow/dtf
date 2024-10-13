import os
import numpy as np
import tensorflow.compat.v1 as tf

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", required=True, help="path to model")
args = parser.parse_args()
frozen_graph = args.model

print("Model {}".format(frozen_graph))

solib = os.environ.get("STUPID_SOLIB_PATH")
print("Loading {}".format(solib))
custom_op = tf.load_op_library(solib)
print("Loaded {} {}".format(solib, custom_op))
print()

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
  X = np.random.rand(3, 4)
  X[0][0] = 0.1
  print("X:\n", X)
  d = {"X:0": X}

  # # feed it to placeholder a via the dict 
  print(sess.run(fetches=["Output:0"], feed_dict=d))
