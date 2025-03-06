"""
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
          dim {
            size: 1
          }
          dim {
            size: 6
          }
        }
        tensor_content: "\001\000\000\000\002\000\000\000\003\000\000\000\004\000\000\000\005\000\000\000\364\001\000\000"
      }
    }
  }
"""

import tensorflow._api.v2.compat.v1 as tf
tf.disable_eager_execution()
from tensorflow.core.framework.tensor_pb2 import TensorProto

c = tf.constant([[1., 2., 3.], [4., 5., 6.]], dtype=tf.float32)
node = c.op.node_def
print(node)
print()

tp = tf.make_tensor_proto(
  [[1.1, 2.2, 3.3], [4.4, 5.5, 6.6]],
  dtype=tf.float32, 
  shape=(2, 3)
)
node.attr["value"].CopyFrom(tf.AttrValue(tensor=tp))
print(node)