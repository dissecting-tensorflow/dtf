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
sess = tf.Session()

# Import the graph protobuf into our new graph.
tf.import_graph_def(graph_def=graph_protobuf, name="")

gd = tf.get_default_graph().as_graph_def()

for node in gd.node:
  if node.op != "Const":
    continue
  tproto = node.attr["value"].tensor
  decoded = tf.io.decode_raw(tproto.tensor_content, out_type=tf.as_dtype(tproto.dtype), little_endian=True)
  out = decoded.eval(session=sess)
  print(f"{node.name}: {out}")

