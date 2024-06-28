#!/bin/bash

THIRD_PARTY_PATH=/data00/home/son.nguyen/workspace/common
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-11.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$THIRD_PARTY_PATH/cudnn/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$THIRD_PARTY_PATH/tensorflow/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$THIRD_PARTY_PATH/protobuf/lib
echo "LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH | tr ':' '\n'
export TF_CPP_MAX_VLOG_LEVEL=3
export CUDA_VISIBLE_DEVICES=0,1,2,3
bazel-bin/main