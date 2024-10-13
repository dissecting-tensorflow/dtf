import json
import tensorflow.compat.v1 as tf

solib = "/path/to/libstupid_op.so"
print("Loading {}".format(solib))
try:
    custom_op = tf.load_op_library(solib)
    print(json.dumps(dir(custom_op), indent=2))
    print()
except Exception as ex:
    print(ex)