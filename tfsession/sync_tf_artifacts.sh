#!/bin/bash

set -e

output_dir=`pwd`/output
echo "output_dir=$output_dir"

rm -rf $output_dir
mkdir -p $output_dir/tensorflow/include/third_party
mkdir -p $output_dir/tensorflow/lib

# headers
rsync -avP -m --include="*/" --include "*.h" --exclude="*" tensorflow $output_dir/tensorflow/include
rsync -avP -m --include="*/" --include "*.inc" --exclude="*" tensorflow $output_dir/tensorflow/include
rsync -avP -m --include="*/" --include "*.h" --exclude="*" third_party $output_dir/tensorflow/include
rsync -avP -m --include="*/" --exclude="BUILD" --exclude="*.BUILD" --exclude="LICENSE" --exclude="*.bzl" third_party/eigen3 $output_dir/tensorflow/include/third_party

cd bazel-out/k8-opt/bin
rsync -avP -m --include="*/" --include "*.h" --exclude="*" tensorflow $output_dir/tensorflow/include
cd -

cd bazel-tensorflow/external/com_google_absl/
rsync -avP -m --include="*/" --include "*.h" --include "*.inc" --exclude="*" absl $output_dir/tensorflow/include/
cd -

cd bazel-tensorflow/external/eigen_archive/
rsync -rvP --copy-unsafe-links Eigen $output_dir/tensorflow/include/
rsync -rvP --copy-unsafe-links unsupported $output_dir/tensorflow/include/
cd -

# solib
rsync -avP bazel-bin/tensorflow/*.so $output_dir/tensorflow/lib
rsync -avP bazel-bin/tensorflow/*.so.* $output_dir/tensorflow/lib

if [ "$?" == "0" ]; then
  echo "DONE"
  echo "$output_dir"
else
  echo "FAILED"
fi
