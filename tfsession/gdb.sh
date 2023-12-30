#!/bin/bash

THIRD_PARTY_PATH=/data00/home/son.nguyen/workspace/common
export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$THIRD_PARTY_PATH/cudnn/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$THIRD_PARTY_PATH/tensorflow/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$THIRD_PARTY_PATH/protobuf/lib
echo "LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH | tr ':' '\n'
export TF_CPP_MAX_VLOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=0,1,2,3
gdb bazel-bin/main

b main.cpp:161
run
set scheduler-locking on
set scheduler-locking step
set pagination off

b tensorflow/core/common_runtime/executor.cc:782 #v2.9.0
commands
python import time
python s=time.time_ns()
continue
end

b tensorflow/core/common_runtime/executor.cc:834 #v2.9.0
commands
p (*(*(*item.kernel).props_._M_ptr).node_def.name_.ptr_)._M_dataplus
python print("Duration: {}".format(time.time_ns() - s))
continue
end
