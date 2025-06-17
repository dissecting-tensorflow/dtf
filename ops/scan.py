import tensorflow as tf

# Input sequence
data = tf.constant([1, 2, 3, 4], dtype=tf.float32)

# Define the scan function
def scan_fn(accumulator, element):
    return accumulator + element

# Run tf.scan
result = tf.scan(fn=scan_fn, elems=data, initializer=0.0)

# Output: [1, 3, 6, 10]
print(result)

# launch the graph in a session
graph_pbtxt_path = "scan.pbtxt"
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
