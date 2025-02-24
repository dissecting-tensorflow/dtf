"""
export TF_CPP_MAX_VLOG_LEVEL=3
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64
"""

import json
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

data = tf.constant([[1, 2, 3], [4, 5, 6]])
axis = tf.rank(data) - 1
out = tf.math.reduce_sum(data, axis=axis, name="Sum_1")
print(out)

graph = tf.get_default_graph()
graph_def = graph.as_graph_def()

sum_op = graph.g.get_operation_by_name("Sum_1")
print(json.dumps(dir(sum_op), indent=2))

graph_file_path = "./sum.pbtxt"
graph_writer = open(graph_file_path, "w")
graph_writer.write(str(graph_def))
graph_writer.close()

graph_file_path = "./sum.pb"
with tf.gfile.GFile(graph_file_path, 'wb') as f:
    data = graph_def.SerializeToString()
    f.write(data)
