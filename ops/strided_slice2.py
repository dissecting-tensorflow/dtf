import tensorflow as tf

print()
print()
t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])
print(f"t.shape: {t.shape}")


print()
                         #begin     #end       #strides
o1 = tf.strided_slice(t, [1, 0, 0], [2, 1, 3], [1, 1, 1])  # [[[3, 3, 3]]]
print(o1)
print()

o2 = tf.strided_slice(t, [1, 0, 0], [2, 2, 3], [1, 1, 1])  # [[[3, 3, 3],
                                                           #   [4, 4, 4]]]
print(o2)
print()

o3 = tf.strided_slice(t, [1, -1, 0], [2, -3, 3], [1, -1, 1])  # [[[4, 4, 4],
                                                              #   [3, 3, 3]]]
print(o3)
print()

o4 = tf.strided_slice(t, [0, 3, 0], [3, 2, 3], [1, 1, 1])
print(o4)
print()