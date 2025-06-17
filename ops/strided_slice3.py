import tensorflow._api.v2.compat.v1 as tf
tf.disable_eager_execution()

print()
print()
ph = tf.placeholder(tf.float32, shape=[None, 1000, 1000], name="ph")

s = tf.shape(ph, name="shape_ph")
print(f"s.op: {s.op}")
                         #begin     #end       #strides
o1 = tf.strided_slice(s, [1],       [2],       [1], shrink_axis_mask=1, name="s1")
print(o1)
print()

g = tf.get_default_graph()
s1 = g.get_operation_by_name("s1")
input_op = s1.inputs[0].op
print(f"input_op:\n{input_op}\n")
if input_op.type != "Shape":
  raise ValueError("The input operation to strided_slice is not a Shape operation.")

shape = input_op.inputs[0].shape
print(shape)
b = s1.inputs[1]
e = s1.inputs[2]
s = s1.inputs[3]
cond1 = b.op.type == "Const" and e.op.type == "Const" and s.op.type == "Const"
cond2 = len(b.shape) == 1 and len(e.shape) == 1 and len(s.shape) == 1
if not (cond1 and cond2):
  raise ValueError("Inputs to strided_slice are not constants or have unexpected shapes.")

with tf.Session() as sess:
  bo, eo, so = sess.run([b, e, s])

bv = bo[0]
ev = eo[0]
sv = so[0]

bm = int(s1.get_attr("begin_mask"))
if bm & 1:
   bv = 0

em = int(s1.get_attr("end_mask"))
if em & 1:
   ev = len(shape)

output = shape[bv : ev : sv]
for e in output:
  if e in [None, -1]:
    raise ValueError("Output of strided_slice contains None or -1, which is not allowed.")

shrink_axis_mask = int(s1.get_attr("shrink_axis_mask"))
if len(output) == 1 and shrink_axis_mask & 1:
  output = output[0]

print(f"output: {output}")

# Output pbtxt file
graph_pbtxt_path = "strided_slice3.pbtxt"
with open(graph_pbtxt_path, "w") as graph_writer:
    graph_writer.write(str(g.as_graph_def()))
print(graph_pbtxt_path)