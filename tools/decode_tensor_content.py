"""
Give the following TensorProto
tproto = tensor {
  dtype: DT_INT32
  tensor_shape {
    dim {
      size: 3
    }
  }
  tensor_content: "\001\000\000\000\002\000\000\000\003\000\000\000"
}
"""

import tensorflow._api.v2.compat.v1 as tf
tf.disable_eager_execution()
sess = tf.Session()

c = tf.constant([1, 2, 3], dtype=tf.int32)
tproto = c.op.node_def.attr["value"].tensor

decoded = tf.io.decode_raw(tproto.tensor_content, out_type=tf.as_dtype(tproto.dtype), little_endian=True)
out = decoded.eval(session=sess)
print(out)
