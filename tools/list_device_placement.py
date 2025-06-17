import tensorflow as tf

# Enable device placement logging
tf.debugging.set_log_device_placement(True)

# Example operation
op_name = "MatMul"
a = tf.constant([[1.0, 2.0]], name="a")
b = tf.constant([[3.0], [4.0]], name="b")
result = tf.matmul(a, b, name=op_name)  # Run MatMul op

print(f"Operation '{op_name}' executed. Check logs for device placement.")