import logging
import tensorflow as tf
from tensorflow.python.lib.io import file_io

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--graph", "-g", required=True, help="path to graph")
args = parser.parse_args()
frozen_graph_path = args.graph

def load_frozen_graph(frozen_graph_path):
  graph_def = tf.GraphDef()
  file_content = file_io.read_file_to_string(frozen_graph_path, True)
  try:
    graph_def.ParseFromString(file_content)
    logging.debug("Loaded [%s] from [%s] as binary successfully.", type(graph_def), frozen_graph_path)
  except Exception as ex:
    raise ex
  return graph_def

graph = load_frozen_graph(frozen_graph_path)
node_def = None
for node in graph.node:
  if node.op == "Placeholder":
    node_def = node
    break
shape = [d.size for d in node_def.attr["shape"].shape.dim]
print(f"shape: {shape}")
