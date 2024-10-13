#!/bin/bash

########################################################
# Prerequisites
# Convert BUILD to bazel format
########################################################

cwd=$(pwd)
project_root_dir=$(dirname $(dirname $cwd))

export PATH=/usr/local/cuda/bin:$PATH
bazel build :stupid_op
