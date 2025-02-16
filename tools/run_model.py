"""
Python >= 3.7.3
pip install tensorflow==2.5.0

python tools/run_model.py -g tfsession/models/test/test_model_v1/sample.pb
"""

############################################################################################################
# Customize logging
############################################################################################################
import logging
import coloredlogs

# install a handler on the root logger
coloredlogs.install(
    level=logging.INFO,
    fmt="%(levelname)s %(message)s"
)

logging.debug("This is a debug message")
logging.warning("This is a warning message")
############################################################################################################

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--graph", "-g", required=True, help="model graph path")
args = parser.parse_args()
graph_path = args.graph

import os
import numpy as np
import tensorflow.compat.v1 as tf


def load_frozen_graph(frozen_graph_path):
  from tensorflow.python.lib.io import file_io
  graph_def = tf.GraphDef()
  file_content = file_io.read_file_to_string(frozen_graph_path, True)
  try:
    graph_def.ParseFromString(file_content)
    logging.debug("Loaded [%s] from [%s] as binary successfully.", type(graph_def), frozen_graph_path)
  except Exception as ex:
    raise ex

  return graph_def


def get_feed_nodes(graph_def):
  feeds = []
  for node in graph_def.node:
    if node.op == "Placeholder":
      feeds.append(node)
  return feeds


def generate_inference_input(graph_def, batch_size=3):
  feed_nodes = get_feed_nodes(graph_def)
  feeds = {}
  fetches = ["Sigmoid:0"]
  for feed in feed_nodes:
    name = f"{feed.name}:0"
    shape = [d.size for d in feed.attr["shape"].shape.dim]
    if shape[0] in [None, -1]:
      shape[0] = batch_size
    dtype = tf.as_dtype(feed.attr["dtype"].type)
    ndarr = None
    if "int" in str(dtype):
      mu, sigma = 10, 1
      ndarr = np.random.normal(mu, sigma, shape).astype(np.int)
    elif "float" in str(dtype):
      mu, sigma = 1.0, 0.1
      ndarr = np.random.normal(mu, sigma, size=shape)
    else:
      raise Exception(f"Unsupported dtype {dtype}")
    feeds[name] = ndarr
    print(f"{name}: {ndarr} {dtype}")

  return feeds, fetches



graph_def = load_frozen_graph(graph_path)
feeds, fetches = generate_inference_input(graph_def)
for k, _ in feeds.items():
    print(f"feed name: {k}")

print()
print(f"fetches: {fetches}")


# Import graph
with tf.gfile.GFile(graph_path, "rb") as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def, name='')
print("Successfully imported graph")

# launch the graph in a session
with tf.Session() as sess:
  # feed it to placeholder a via the dict
  outputs = sess.run(fetches=fetches, feed_dict=feeds)
  for i in range(len(fetches)):
    print("{}:\n{}".format(fetches[i], outputs[i]))
    print()

"""
python tools/run_model.py -g tfsession/models/test/test_model_v1/sample.pb
"""