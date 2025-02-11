import tensorflow as tf
node_def = tf.NodeDef()
node_def.attr["dtype"].CopyFrom(tf.AttrValue(type=tf.float16.as_datatype_enum))
dtype_enum = node_def.attr["dtype"].type
dtype = tf.as_dtype(dtype_enum)
print(f"{dtype_enum}: {dtype}")
