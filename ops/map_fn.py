import tensorflow as tf

def process_element(x):
    return x * 2

data = tf.constant([[1, 2], [3, 4]])
result = tf.map_fn(process_element, data, dtype=tf.int32)

# launch the graph in a session
graph_pbtxt_path = "map_fn.pbtxt"
with tf.Session() as sess:
  graph = tf.get_default_graph()
  graph_def = graph.as_graph_def()
  graph_writer = open(graph_pbtxt_path, "w")
  graph_writer.write(str(graph_def))
  graph_writer.close()
  print(graph_pbtxt_path)
  graph_pb_path = "map_fn.pb"
  with tf.gfile.GFile(graph_pb_path, 'wb') as f:
    data = graph_def.SerializeToString()
    f.write(data)
  print(graph_pb_path)

