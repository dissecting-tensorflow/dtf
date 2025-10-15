import os
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import graph_util
from google.protobuf import text_format
import numpy as np

tf.disable_eager_execution()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--graph", "-g", required=True, help="path to graph")
args = parser.parse_args()
graph_pbtxt_path = args.graph

# Let's read our pbtxt file into a Graph protobuf
f = open(graph_pbtxt_path, "r")
graph_def = text_format.Parse(f.read(), tf.GraphDef())
for node in graph_def.node:
  if node.name == "strided_slice_441/stack_1":
    node.input[:] = []
    print(node)
