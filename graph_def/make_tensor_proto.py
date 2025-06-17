import logging
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util, tensor_shape


def make_tensor_proto(data, dtype):
    try:
        if isinstance(data, np.ndarray) and isinstance(dtype, tf.DType):
            tensor_proto = tensor_pb2.TensorProto(
                dtype=dtype.as_datatype_enum,
                tensor_shape=tensor_shape.as_shape(data.shape).as_proto(),
            )
            tensor_proto.tensor_content = data.astype(dtype.as_numpy_dtype()).tobytes()
        else:
            tensor_proto = tensor_util.make_tensor_proto(data, dtype=dtype)
        return tensor_proto
    except Exception as e:
        logging.warning(
            "Failed to make tensor_proto. Value is: {}, dtype is: {}, error info: {}".format(
                data, dtype, str(e)
            )
        )
        return None

a =  np.random.rand(2, 3)
dtype = dtypes.as_dtype(a.dtype)
tp = make_tensor_proto(a, dtype)
print()
print("Tensor Proto:")
print(tp)
