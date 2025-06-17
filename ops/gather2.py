import numpy as np
import tensorflow as tf

# keepdims = True
int_data = tf.constant(np.random.randint(low=0, high=1, size=(2, 2)))
sum = tf.math.reduce_sum(int_data, axis=1, keepdims=True)
data = tf.constant(np.random.randint(low=0, high=10, size=(2, 4)))
print("=" * 20)
print("tf.gather")
print("=" * 20)
print("params: ", end="")
print(data)
print("indices: ", end="")
print(sum)
print(f"axis: 0")
gather = tf.gather(data, indices=sum, axis=0)
print(f"output: ", end="")
print(gather)
