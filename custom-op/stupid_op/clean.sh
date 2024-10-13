#!/bin/bash

cwd=$(pwd)
blade_root_dir=$(dirname $(dirname $cwd))
rm -rvf ${blade_root_dir}/build64_release/AutoFusion/stupid_op