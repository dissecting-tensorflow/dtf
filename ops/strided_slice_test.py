import numpy as np
import tensorflow as tf

t = tf.constant(np.random.rand(1, 51, 384))

print()
                         #begin     #end       #strides
o1 = tf.strided_slice(t, [0, 0], [0, 1024], [1, 1], begin_mask=1, end_mask=1)
print(o1)
print(o1.shape[1])
print()

# x[:, 0:1024]