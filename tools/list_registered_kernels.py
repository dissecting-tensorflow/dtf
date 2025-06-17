import tensorflow._api.v2.compat.v1 as tf
tf.debugging.set_log_device_placement(True)

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices[0], 'GPU')

# Disable soft device placement to prevent CPU fallback
tf.config.set_soft_device_placement(False)

# Force generating error "All kernels registered for op Einsum"
# Using an unregistered data type such as tf.string
with tf.device('/device:GPU:0'):
  m0 = tf.constant("A", dtype=tf.string)
  m1 = tf.constant("B", dtype=tf.string)
  e = tf.einsum('ij,jk->ik', m0, m1)

"""
tensorflow.python.framework.errors_impl.NotFoundError: Could not find device for node: {{node Einsum}} = Einsum[N=2, T=DT_STRING, equation="ij,jk->ik"]
All kernels registered for op Einsum:
  device='GPU'; T in [DT_HALF]
  device='GPU'; T in [DT_FLOAT]
  device='GPU'; T in [DT_DOUBLE]
  device='GPU'; T in [DT_COMPLEX64]
  device='GPU'; T in [DT_COMPLEX128]
  device='CPU'; T in [DT_INT64]
  device='CPU'; T in [DT_INT32]
  device='CPU'; T in [DT_HALF]
  device='CPU'; T in [DT_FLOAT]
  device='CPU'; T in [DT_DOUBLE]
  device='CPU'; T in [DT_COMPLEX64]
  device='CPU'; T in [DT_COMPLEX128]
  device='CPU'; T in [DT_BFLOAT16]
 [Op:Einsum]
"""
