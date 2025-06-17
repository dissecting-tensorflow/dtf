from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto

dim0 = TensorShapeProto.Dim()
dim1 = TensorShapeProto.Dim()
dim0.size = -1
dim1.size = 3
ts = TensorShapeProto()
ts.dim.append(dim0)
ts.dim.append(dim1)
print(ts)
