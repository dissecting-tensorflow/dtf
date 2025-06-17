import tensorflow as tf
from tensorflow.core.framework import types_pb2
node_def = tf.compat.v1.NodeDef()
node_def.attr["dtype"].CopyFrom(tf.compat.v1.AttrValue(type=tf.float16.as_datatype_enum))
dtype_enum = node_def.attr["dtype"].type
dtype = tf.as_dtype(dtype_enum)
print(f"{types_pb2.DataType.Name(dtype_enum)}({dtype_enum}): {dtype.name}")

"""
Output:
DT_HALF(19): float16
"""
