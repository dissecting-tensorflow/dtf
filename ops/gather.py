"""
pip install tensorflow==2.5.0
sudo rsync -avP cudnn/lib/* /usr/local/cuda-11.4/lib64/

export TF_CPP_MAX_VLOG_LEVEL=3
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64

Note: tf.gather uses underlying GatherV2 op
Executing op GatherV2 in device /job:localhost/replica:0/task:0/device:GPU:0

The params may also have any shape. gather can select slices across any axis depending on the axis argument (which defaults to 0). 
Below it is used to gather first rows, then columns from a matrix:
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
