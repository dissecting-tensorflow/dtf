import sys
import tensorflow.compat.v1 as tf
from tensorflow.python.lib.io import file_io

pb_file_path = "sample.pb"
file_content = file_io.read_file_to_string(pb_file_path, True)

graph_def = tf.GraphDef()
graph_def.ParseFromString(file_content)
target_node = None
for n in graph_def.node:
  if n.name == "I4":
    target_node = n
    break

tensor_proto = None
if target_node is not None:
  v = target_node.attr.get("value")
  # type(value)
  # <class 'tensorflow.core.framework.attr_value_pb2.AttrValue'>
  if v is not None:
    tensor_proto = v.tensor
    # type(tensor_proto)
    # <class 'tensorflow.core.framework.tensor_pb2.TensorProto'>

if tensor_proto is None:
  sys.exit(1)

# type(tensor)
# <class 'tensorflow.core.framework.tensor_pb2.TensorProto'>

from tensorflow.python.framework import tensor_util
a = tensor_util.MakeNdarray(tensor_proto)
print(a)


