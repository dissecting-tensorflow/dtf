"""
pip install tensorflow==2.5.0
sudo rsync -avP cudnn/lib/* /usr/local/cuda-11.4/lib64/

export TF_CPP_MAX_VLOG_LEVEL=3
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64

Note: tf.gather uses underlying GatherV2 op
Executing op GatherV2 in device /job:localhost/replica:0/task:0/device:GPU:0

Description:
Gather slices from params along axis according to indices. (deprecated arguments).
More generally: The output shape has the same shape as the input, with the indexed-axis replaced by the shape of the indices.
tf.gather(
    params, indices, validate_indices=None, axis=None, batch_dims=0, name=None
)
- params may also have any shape. gather can select slices across any axis depending on the axis argument (which defaults to 0). 
- indices must be an integer tensor of any dimension

"""

import tensorflow as tf
params = tf.constant([[0, 1.0, 2.0],
                      [10.0, 11.0, 12.0],
                      [20.0, 21.0, 22.0],
                      [30.0, 31.0, 32.0]])

output_tensor = tf.gather(params, indices=[3,1])
print(output_tensor)
"""
Output:
tf.Tensor(
[[30. 31. 32.]
 [10. 11. 12.]], shape=(2, 3), dtype=float32)
"""

output_tensor = tf.gather(params, indices=[2,1], axis=1)
print(output_tensor)
"""
Output:
tf.Tensor(
[[ 2.  1.]
 [12. 11.]
 [22. 21.]
 [32. 31.]], shape=(4, 2), dtype=float32)
"""

print()
print("=" * 80)
print("indices=[[2,1], [1,2]], axis=0")
print("=" * 80)
output_tensor = tf.gather(params, indices=[[2,1], [1,2]], axis=0)
print(output_tensor)
print("=" * 80)
"""
Output:
"""

output_tensor = tf.gather(params, indices=[[2,1], [1,2]], axis=1)
print(output_tensor)
"""
tf.Tensor(
[[[ 2.  1.]
  [ 1.  2.]]

 [[12. 11.]
  [11. 12.]]

 [[22. 21.]
  [21. 22.]]

 [[32. 31.]
  [31. 32.]]], shape=(4, 2, 2), dtype=float32)
"""

# 3D params
# shape = (3, 2, 3)
params = tf.constant([[[0,     1.0,  2.0],
                       [10.0, 11.0, 12.0]],

                      [[20.0, 21.0, 22.0],
                       [30.0, 31.0, 32.0]],

                      [[40.0, 41.0, 42.0],
                       [50.0, 51.0, 52.0]]])

output_tensor = tf.gather(params, indices=[[1,0], [0,1]], axis=1)
print(output_tensor)
