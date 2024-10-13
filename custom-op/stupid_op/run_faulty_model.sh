#!/bin/bash

cwd=$(pwd)
base_dir=$(basename $(dirname $cwd))
workspace_dir=$(dirname $(dirname $cwd))
echo workspace_dir=${workspace_dir}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/nvidia/current
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/nccl/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/extras/CUPTI/lib64

export STUPID_SOLIB_PATH=${workspace_dir}/build64_release/${base_dir}/stupid_op/libstupid_op.so
echo STUPID_SOLIB_PATH=${STUPID_SOLIB_PATH}
python run_faulty_model.py -m faulty_model_v1.pb