#!/bin/bash

set -e

debug_dir=/data00/son.nguyen/workspace/auto_fusion_dev/laniakea/lib/

output_dir=/data00/son.nguyen/workspace/common/cpp3rdlib
# output_dir=/data00/son.nguyen/workspace/auto_fusion_dev/cpp3rdlib
echo "output_dir=$output_dir"

# rm -rf $output_dir
mkdir -p $output_dir/tensorflow/include/third_party
mkdir -p $output_dir/tensorflow/lib

# headers
rsync -avP -m --include="*/" --include "*.h" --exclude="*" tensorflow $output_dir/tensorflow/include
rsync -avP -m --include="*/" --include "*.inc" --exclude="*" tensorflow $output_dir/tensorflow/include
rsync -avP -m --include="*/" --include "*.proto" --exclude="*" tensorflow $output_dir/tensorflow/include
rsync -avP -m --include="*/" --include "*.h" --exclude="*" third_party $output_dir/tensorflow/include
rsync -avP -m --include="*/" --exclude="BUILD" --exclude="*.BUILD" --exclude="LICENSE" --exclude="*.bzl" third_party/eigen3 $output_dir/tensorflow/include/third_party

cd bazel-out/k8-opt/bin
rsync -avP -m --include="*/" --include "*.h" --exclude="*" tensorflow $output_dir/tensorflow/include
cd -

cd bazel-tensorflow/external/com_google_absl/
rsync -avLP -m --include="*/" --include "*.h" --exclude="*" absl $output_dir/tensorflow/include/
rsync -avLP -m --include="*/" --include "*.inc" --exclude="*" absl $output_dir/tensorflow/include/
cd -

cd bazel-tensorflow/external/eigen_archive/
rsync -rvP --copy-unsafe-links Eigen $output_dir/tensorflow/include/
rsync -rvP --copy-unsafe-links unsupported $output_dir/tensorflow/include/
cd -

# cuda
cd bazel-bin/external/local_config_cuda/cuda/_virtual_includes/cuda_headers_virtual/third_party/
rsync -avLP -m --include="*/" --include "*.h" --exclude="*" gpus $output_dir/tensorflow/include/third_party/
rsync -avLP -m --include="*/" --include "*.inc" --exclude="*" gpus $output_dir/tensorflow/include/third_party/
cd -

# cudnn
cd bazel-bin/external/local_config_cuda/cuda/_virtual_includes/cudnn_header/third_party/
rsync -avLP -m --include="*/" --include "*.h" --exclude="*" gpus $output_dir/tensorflow/include/third_party/
rsync -avLP -m --include="*/" --include "*.inc" --exclude="*" gpus $output_dir/tensorflow/include/third_party/
cd -

# cudnn_frontend
cd bazel-bin/external/cudnn_frontend_archive/_virtual_includes/cudnn_frontend/third_party/
rsync -avLP -m --include="*/" --include "*.h" --exclude="*" cudnn_frontend $output_dir/tensorflow/include/third_party/
rsync -avLP -m --include="*/" --include "*.hpp" --exclude="*" cudnn_frontend $output_dir/tensorflow/include/third_party/
rsync -avLP -m --include="*/" --include "*.inc" --exclude="*" cudnn_frontend $output_dir/tensorflow/include/third_party/
cd -

# solib
cd bazel-bin/tensorflow/
ln -sf libtensorflow_framework.so.1 libtensorflow_framework.so
cd -
rsync -avP bazel-bin/tensorflow/*.so $output_dir/tensorflow/lib
rsync -avP bazel-bin/tensorflow/*.so.* $output_dir/tensorflow/lib
rsync -LvP bazel-bin/tensorflow/libtensorflow_cc.so.1 $debug_dir
rsync -LvP bazel-bin/tensorflow/libtensorflow_framework.so.1 $debug_dir

# BUILD
cp -fv .tf_configure.bazelrc $output_dir/tensorflow/
cp -v BUILD.blade $output_dir/tensorflow/BUILD

# version
cat <<EOF > $output_dir/tensorflow/version.txt
1.15.5+nv
EOF

if [ "$?" == "0" ]; then
  echo "DONE"
  echo "$output_dir"
else
  echo "FAILED"
fi
