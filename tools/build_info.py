import json
import tensorflow as tf

info = tf.sysconfig.get_build_info()
print(json.dumps(info, indent=2, default=str))
