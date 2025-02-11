import os
import json
import logging
import coloredlogs
from collections import OrderedDict

# install a handler on the root logger
coloredlogs.install(
    level=logging.DEBUG,
    fmt="%(levelname)s %(message)s"
)

import tensorflow.compat.v1 as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework.ops import Operation

###################################################################################################
# Global variables
###################################################################################################
filename = "model_dir/graph"
bs_config_path = "model_dir/predict_online.bs.json"
###################################################################################################

###################################################################################################
# Load custom ops
###################################################################################################
import custom_operators
custom_operators.load_ops()
###################################################################################################

file_content = file_io.read_file_to_string(filename, True)
graph_def = tf.GraphDef()
try:
    graph_def.ParseFromString(file_content)
    tf.import_graph_def(graph_def, name="")
    logging.debug("Load [%s] from [%s] as binary ok.", type(graph_def), filename)
except Exception as ex:
    logging.debug("Failed to load %s, %s", filename, ex)
    raise ex

default_graph = tf.get_default_graph()

batch_size = 137
op_count = 0
ph_count = 0
custom_shape_infer_graph = tf.Graph()

bs_config = {}
if len(bs_config_path) and os.path.exists(bs_config_path):
    with open(bs_config_path, "r") as fd:
        bs_config = json.load(fd)
###################################################################################################


def op_generator(ops):
    for op in ops:
        yield op


def find_op(target_op_name, ops, name2ops):
    global ph_count
    for op in ops:
        node_def = op.node_def
        if op.type == "Placeholder":
            ph_count += 1
            logging.debug("[ph_count={}] Processing Placeholder {}".format(ph_count, op.name))
            node_def = set_symbolic_shape(op, batch_size, bs_config)

        inputs = []
        for inp in op.inputs:
            name, idx = inp.name.split(":")
            inputs.append(name2ops[name].outputs[int(idx)])

        try:
            shape_infer_op = Operation(node_def, custom_shape_infer_graph, inputs)
            custom_shape_infer_graph._create_op_helper(shape_infer_op)
            if op.name == "DynamicStitch":
                print("")
                logging.debug("Found op DynamicStitch:")
                for inp in inputs:
                    print("Input: {}".format(inp))
                for out in shape_infer_op.outputs:
                    print("Output: {}".format(out))
                print("")

            name2ops[op.name] = shape_infer_op
        except Exception as ex:
            logging.debug("op.name={} node_def.name={}".format(op.name, node_def.name))
            logging.debug(str(node_def))
            logging.fatal(ex)

        if op.name == target_op_name:
            return op

    return None

def is_valid_batch_size(shape):
    if len(shape) == 0:
        return True

    cnt = sum(
        [
            1
            for elem in shape
            if elem == -1
            or elem is None
            or (hasattr(elem, "value") and elem.value is None)
        ]
    )
    bs = shape[0]
    if hasattr(bs, "value"):
        bs = bs.value
    if cnt != 1 or bs not in [-1, None]:
        return False

    return True

def check_valid_batch_size_dynamic_shape(node_def, shape, v1=None):
    cnt = sum(
        [
            1
            for elem in shape
            if elem == -1
            or elem is None
            or (hasattr(elem, "value") and elem.value is None)
        ]
    )
    bs = shape[0]
    if hasattr(bs, "value"):
        bs = bs.value
    if cnt != 1 or bs not in [-1, None]:
        print(str(node_def))
        if v1:
            print(v1)
        raise RuntimeError(
            "ph with more than one -1 in shape or -1 is not in dimension. shape is  {}".format(
                shape
            )
        )
    # list-like
    # input may be TensorShape obj
    return [elem for elem in shape]


def set_symbolic_shape(ph_op, batch_size, bs_config):
    node_def = ph_op.node_def
    new_op_def = tf.NodeDef()
    new_op_def.CopyFrom(node_def)
    shape = [d.size for d in node_def.attr["shape"].shape.dim]
    check_valid_batch_size_dynamic_shape(node_def, shape)
    ts_bs = bs_config.get(node_def.name, [-1])[0]
    # tensorsharp set bs 16 by default
    if ts_bs == -1 or ts_bs == 16:
        shape[0] = batch_size
    else:
        shape[0] = ts_bs
    new_op_def.attr["shape"].CopyFrom(
        tf.AttrValue(shape=tensor_shape.as_shape(shape).as_proto())
    )
    return new_op_def


def dfs_graph(target_op_name, ops, name2ops, results):
    # Break cycle
    if target_op_name in results:
        return

    # Find the target op
    op = None
    if target_op_name in name2ops:
        op = name2ops[target_op_name]
    else:
        op = find_op(target_op_name, ops, name2ops)
    if op is None:
        raise Exception("Invalid op {}".format(target_op_name))

    # Scan inputs
    for inp in op.inputs:
        name, _ = inp.name.split(":")
        dfs_graph(name, ops, name2ops, results)

    # Save the found op
    results[op.name] = op
    shape_infer_op = name2ops[op.name]
    if shape_infer_op.type not in ["Placeholder", "Const"]:
        output0_shape = shape_infer_op.outputs[0].shape
        logging.debug("Node {}".format(shape_infer_op.name))
        print("output0_shape = {}".format(output0_shape))
        print(str(shape_infer_op.node_def))
        print("")
        # if not is_valid_batch_size(output0_shape):
        #     logging.warning("==> Found faulty node {}".format(shape_infer_op.name))
        #     print("output0_shape = {}".format(output0_shape))
        #     print(str(shape_infer_op.node_def))
        #     print("")
        # else:
        #     logging.debug("Normal node {}".format(shape_infer_op.name))
        #     print("output0_shape = {}".format(output0_shape))
        #     print(str(shape_infer_op.node_def))
        #     print("")




ops = op_generator(default_graph.get_operations())
name2ops = OrderedDict()
results = OrderedDict()

target_op_list = [
    # "Squeeze_5",
    # "Squeeze_6",
    # "DynamicStitch",
    "strided_slice_107_repeat"
]

for target_op_name in target_op_list:
    results.clear()
    dfs_graph(target_op_name, ops, name2ops, results)
    print("")
    print("Op {}:".format(target_op_name))
    for inp in name2ops[target_op_name].inputs:
        name, idx = inp.name.split(":")
        print("Input {}, shape={}".format(inp.name, name2ops[name].outputs[int(idx)].shape))
    for out in name2ops[target_op_name].outputs:
        print("Output {}".format(out))
    print("")

    # Build sub graph
    sub_graph_def = tf.GraphDef()
    for name, op in results.items():
        sub_graph_def.node.append(op.node_def)

    sub_graph_pb = os.path.join(os.path.dirname(filename), "sub_graph_{}.pb".format(target_op_name))
    sub_graph_txt = os.path.join(os.path.dirname(filename), "sub_graph_{}.pbtxt".format(target_op_name))
    txt_writer = open(sub_graph_txt, "w")
    txt_writer.write(str(sub_graph_def))
    txt_writer.close()

    pb_writer = open(sub_graph_pb, "wb")
    pb_writer.write(sub_graph_def.SerializeToString())
    pb_writer.close()

    print(sub_graph_txt)
    print(sub_graph_pb)
