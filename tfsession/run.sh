#!/bin/bash

THIRD_PARTY_PATH=/data00/home/son.nguyen/workspace/common/cpp3rdlib
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$THIRD_PARTY_PATH/protobuf/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$THIRD_PARTY_PATH/icu/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$THIRD_PARTY_PATH/cuda/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$THIRD_PARTY_PATH/tensorflow/lib
echo "LD_LIBRARY_PATH:"
echo $LD_LIBRARY_PATH | tr ':' '\n'
export TF_CPP_MAX_VLOG_LEVEL=3
bazel-bin//main