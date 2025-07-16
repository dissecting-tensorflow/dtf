import tensorflow._api.v2.compat.v1 as tf
tf.disable_eager_execution()

import numpy as np

c = tf.constant(np.random.rand(16, 10, 16))
shape = tf.constant([-1, 16])
reshape = tf.reshape(c, shape)
with tf.Session() as sess:
  out = sess.run(reshape)
  print(out.shape)