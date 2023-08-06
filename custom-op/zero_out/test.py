import tensorflow as tf
zero_out_module = tf.load_op_library('bazel-bin/zero_out/libzero_out.so')
output_tensor = zero_out_module.zero_out([[1, 2], [3, 4]])
print(output_tensor.numpy())
