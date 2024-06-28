# https://www.tensorflow.org/install/source
# https://github.com/tensorflow/tensorflow/issues/48919#issuecomment-866438774

# TensorFlow
git clone -b v2.9.0 https://github.com/tensorflow/tensorflow.git
git checkout -b v2.9.0 tags/v2.9.0

# Install Python and the TensorFlow package dependencies
sudo apt install python3-dev python3-pip
pip install -U pip numpy wheel packaging requests opt_einsum
pip install -U keras_preprocessing --no-deps

# Install Bazel
go install github.com/bazelbuild/bazelisk@latest
cp -fv $(which bazelisk) ~/bin/bazel

# GCC
gcc --version
gcc (Debian 8.3.0-6) 8.3.0

# NVIDIA dependencies
cd ~/workspace/common
# cudnn 8.2.4
git clone -b cudnn8.2.4_cuda11.0-gcc8 --depth 1 git@code.byted.org:cpp3rdlib/cudnn.git

# tensorrt 8.2.3
git clone -b TensorRT-8.2.3.0.Linux.x86_64-gnu.cuda-11.0.cudnn8.2-cpp3rdlib --depth 1 git@code.byted.org:cpp3rdlib/tensorrt.git



############################################################################################################################
# Configure the build
############################################################################################################################
./configure

# CUDA path
/usr/local/cuda-11.0,/data00/home/son.nguyen/workspace/common/cudnn,/data00/home/son.nguyen/workspace/common/tensorrt

# CUDA Compute Capabilities
7.0,7.5,8.0

# Don't use clang

cat .tf_configure.bazelrc
build --action_env PYTHON_BIN_PATH="/data00/home/son.nguyen/.pyenv/versions/3.7.3/bin/python3"
build --action_env PYTHON_LIB_PATH="/data00/home/son.nguyen/.pyenv/versions/3.7.3/lib/python3.7/site-packages"
build --python_path="/data00/home/son.nguyen/.pyenv/versions/3.7.3/bin/python3"
build --config=tensorrt
build --action_env TF_CUDA_VERSION="11.0"
build --action_env TF_CUDNN_VERSION="8.2.4"
build --action_env TF_TENSORRT_VERSION="8.2.3"
build --action_env TF_NCCL_VERSION=""
build --action_env TF_CUDA_PATHS="/usr/local/cuda-11.0,/data00/home/son.nguyen/workspace/common/cudnn,/data00/home/son.nguyen/workspace/common/tensorrt"
build --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda-11.0"
build --action_env TF_TENSORRT_STATIC_PATH="/data00/home/son.nguyen/workspace/common/tensorrt"
build --action_env TF_CUDA_COMPUTE_CAPABILITIES="7.0,7.5,8.0,8.6"
build --action_env LD_LIBRARY_PATH="/data00/home/son.nguyen/.gvm/pkgsets/go1.19.4/global/overlay/lib"
build:opt --copt=-Wno-sign-compare
build:opt --host_copt=-Wno-sign-compare
test --flaky_test_attempts=3
test --test_size_filters=small,medium
test --test_env=LD_LIBRARY_PATH
test:v1 --test_tag_filters=-benchmark-test,-no_oss,-no_gpu,-oss_serial
test:v1 --build_tag_filters=-benchmark-test,-no_oss,-no_gpu
test:v2 --test_tag_filters=-benchmark-test,-no_oss,-no_gpu,-oss_serial,-v1only
test:v2 --build_tag_filters=-benchmark-test,-no_oss,-no_gpu,-v1only
############################################################################################################################



# Build with debug symbols
export CPLUS_INCLUDE_PATH=/data00/home/son.nguyen/workspace/common/tensorrt/include
bazel build --config=cuda --per_file_copt=+tensorflow.*,-tensorflow/core/kernels.*@-O0,-g --strip=never //tensorflow:libtensorflow_cc.so --verbose_failures


# Show command details
# -s --copt="-DGOOGLE_CUDA=1"
bazel build -s --explain=bazel.log --config=cuda --copt="-DGOOGLE_CUDA=1" --linkopt=-g --per_file_copt=+tensorflow/core/common_runtime/gpu/gpu_device_factory.cc,+tensorflow/core/framework/device_factory.cc,+tensorflow/core/common_runtime/gpu/gpu_device.cc,+tensorflow/core/common_runtime/executor.cc,+tensorflow/core/common_runtime/gpu/gpu_device.cc,+tensorflow/stream_executor/stream.cc,+tensorflow/stream_executor/cuda/cuda_blas.cc,+tensorflow/core/framework/op_kernel.cc,+tensorflow/core/kernels/sendrecv_ops.cc,+tensorflow/core/kernels/matmul_op_real.cc,+tensorflow/core/common_runtime/gpu/gpu_init.cc@-O0,-g,-fno-inline --strip=never //tensorflow:libtensorflow_cc.so --verbose_failures




############################################################################################################################
# Error and Solution
############################################################################################################################
# fatal error: cuda_runtime_api.h: No such file or directory
ERROR: /data00/home/son.nguyen/workspace/tensorflow_dev/tensorflow/tensorflow/compiler/tf2tensorrt/BUILD:66:11: Compiling tensorflow/compiler/tf2tensorrt/stub/nvinfer_stub.cc failed: (Exit 1): gcc failed: error executing command /usr/bin/gcc -U_FORTIFY_SOURCE -fstack-protector -Wall -Wunused-but-set-parameter -Wno-free-nonheap-object -fno-omit-frame-pointer -g0 -O2 '-D_FORTIFY_SOURCE=1' -DNDEBUG -ffunction-sections ... (remaining 141 arguments skipped)
In file included from bazel-out/k8-opt/bin/external/local_config_tensorrt/_virtual_includes/tensorrt_headers/third_party/tensorrt/NvInferLegacyDims.h:53,
                 from bazel-out/k8-opt/bin/external/local_config_tensorrt/_virtual_includes/tensorrt_headers/third_party/tensorrt/NvInfer.h:53,
                 from tensorflow/compiler/tf2tensorrt/stub/nvinfer_stub.cc:17:
bazel-out/k8-opt/bin/external/local_config_tensorrt/_virtual_includes/tensorrt_headers/third_party/tensorrt/NvInferRuntimeCommon.h:56:10: fatal error: cuda_runtime_api.h: No such file or directory
 #include <cuda_runtime_api.h>
          ^~~~~~~~~~~~~~~~~~~~
# Solution
export CPLUS_INCLUDE_PATH=/usr/local/cuda-11.4/include


############################################################################################################################
# Packaging
############################################################################################################################
# Sync artifacts
# headers
rsync -avP -m --include="*/" --include "*.h" --exclude="*" tensorflow /data00/home/son.nguyen/workspace/common/tensorflow/include
rsync -avP -m --include="*/" --include "*.inc" --exclude="*" tensorflow /data00/home/son.nguyen/workspace/common/tensorflow/include
rsync -avP -m --include="*/" --include "*.h" --exclude="*" third_party /data00/home/son.nguyen/workspace/common/tensorflow/include
rsync -avP -m --include="*/" --exclude="BUILD" --exclude="*.BUILD" --exclude="LICENSE" --exclude="*.bzl" third_party/eigen3 /data00/home/son.nguyen/workspace/common/tensorflow/include/third_party

cd bazel-out/k8-opt/bin
rsync -avP -m --include="*/" --include "*.h" --exclude="*" tensorflow /data00/home/son.nguyen/workspace/common/tensorflow/include
cd -

cd bazel-tensorflow/external/com_google_absl/
rsync -avP -m --include="*/" --include "*.h" --include "*.inc" --exclude="*" absl /data00/home/son.nguyen/workspace/common/tensorflow/include/
rsync -avP -m --include="*/" --include "*.h" --include "*.inc" --exclude="*" absl /data00/home/son.nguyen/workspace/dtf/tensorflow
cd -

cd bazel-tensorflow/external/eigen_archive/
rsync -avP -m --include="*/" --include "*" Eigen /data00/home/son.nguyen/workspace/common/tensorflow/include/
rsync -avP -m --include="*/" --include "*" Eigen /data00/home/son.nguyen/workspace/dtf/tensorflow
rsync -avP -m --include="*/" --include "*" unsupported /data00/home/son.nguyen/workspace/common/tensorflow/include/
rsync -avP -m --include="*/" --include "*" unsupported /data00/home/son.nguyen/workspace/dtf/tensorflow
cd -

# For IntelliSense
# target_dir=/data00/son.nguyen/workspace/pet_model_dev/ref

cd bazel-out/k8-opt/bin
rsync -avP -m --include="*/" --include "*.h" --exclude="*" tensorflow $target_dir

rsync -avP -m --include="*/" --include "*.h" --exclude="*" third_party $target_dir
rsync -avP -m --include="*/" --exclude="BUILD" --exclude="*.BUILD" --exclude="LICENSE" --exclude="*.bzl" third_party/eigen3 $target_dir/third_party

# cd bazel-tensorflow/external/com_google_absl/
# rsync -avP -m --include="*/" --include "*.h" --include "*.inc" --exclude="*" absl $target_dir

# cd bazel-tensorflow/external/eigen_archive/
# rsync -avP -m --include="*/" --include "*" Eigen $target_dir
# rsync -avP -m --include="*/" --include "*" unsupported $target_dir

# target_dir=/data00/son.nguyen/workspace/pet_model_dev/include/tensorflow/third_party/
# rsync -avP -m -L --include="*/" gpus $target_dir

# solib
rsync -avP bazel-bin/tensorflow/*.so.* /data00/home/son.nguyen/workspace/common/tensorflow/lib && rsync -avP bazel-bin/tensorflow/*.so /data00/home/son.nguyen/workspace/common/tensorflow/lib

if [ "$?" == "0" ]; then
  echo "DONE"
else
  echo "FAILED"
fi