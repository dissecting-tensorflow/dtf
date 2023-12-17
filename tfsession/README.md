## Dependencies
```bash
pip install tensorflow=2.5.0
# mkdir .blade_tools
# cd .blade_tools
# ln -sf /opt/tiger/typhoon-blade/bpt
# cd ..
# export DEVREGION='cn'
# .blade_tools/bpt/bpt install tensorflow:2.5.0-cuda11.4-cudnn8.1-61-86: cuda:cuda_11.4.4_470.82.01_linux: cudnn:cudnn-11.2-linux-x64-v8.1.0: --platform=x86_64-clang1101 --install-directory=/data00/home/son.nguyen/workspace/common/cpp3rdlib -y

# protobuf
git clone -b 3.9.2 --depth 1 git@code.byted.org:cpp3rdlib/protobuf.git /data00/home/son.nguyen/workspace/common/cpp3rdlib/

# tensorflow
git clone --branch 2.5.0-cuda11.0-cudnn8.0-70-80-gcc8 --depth 1 git@code.byted.org:cpp3rdlib/tensorflow.git /data00/home/son.nguyen/workspace/common/cpp3rdlib/
cp -r /data00/home/son.nguyen/.pyenv/versions/3.8.0/lib/python3.8/site-packages/tensorflow/include/Eigen ~/workspace/common/cpp3rdlib/tensorflow/include/
cp -r /data00/home/son.nguyen/.pyenv/versions/3.8.0/lib/python3.8/site-packages/tensorflow/include/unsupported ~/workspace/common/cpp3rdlib/tensorflow/include/

# Remove bazel files
rm -vf /data00/home/son.nguyen/workspace/common/cpp3rdlib/tensorflow/include/third_party/eigen3/BUILD
rm -vf /data00/home/son.nguyen/workspace/common/cpp3rdlib/tensorflow/include/third_party/eigen3/LICENSE
rm -vf /data00/home/son.nguyen/workspace/common/cpp3rdlib/tensorflow/include/third_party/eigen3/workspace.bzl
rm -vf /data00/home/son.nguyen/workspace/common/cpp3rdlib/tensorflow/include/third_party/eigen3/eigen_archive.BUILD


cd /data00/home/son.nguyen/workspace/common/cpp3rdlib/tensorflow/include/third_party
mkdir -p gpus/cuda/include
cd gpus/cuda/include
ln -sf /usr/local/cuda/include/cuComplex.h .
ln -sf /usr/local/cuda/include/cuda.h .

rm -rf /data00/home/son.nguyen/workspace/common/cpp3rdlib/protobuf/BUILD
```

# Error
## Error 1
```
./main: relocation error: ./main: symbol _ZN10tensorflow8GraphDefC1Ev version tensorflow not defined in file libtensorflow_cc.so.2 with link time reference
```

### What does it mean?
