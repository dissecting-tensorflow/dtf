import tensorflow._api.v2.compat.v1 as tf
tf.disable_eager_execution()

import numpy as np

c = tf.constant(np.random.rand(1, 51, 384))
# -1 tells TensorFlow to automatically calculate the size of that specific dimension based on 
# the total number of elements in the tensor and the other specified dimensions.
shape = tf.constant([1024, 384])
reshape = tf.reshape(c, shape)
with tf.Session() as sess:
  out = sess.run(reshape)
  print(out.shape)
