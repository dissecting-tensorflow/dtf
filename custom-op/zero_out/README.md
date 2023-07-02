# How to build

## Prerequisites
### 1. GCC
```
$ gcc --version
gcc (Debian 8.3.0-6) 8.3.0
Copyright (C) 2018 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### 1. Python 
Version 3.8.0

### 2. Bazelisk
Version: development  
Build label: 6.2.1

### 3. TensorFlow
Version 2.5.0
```
pip install tensorflow==2.5.0
```

## Bazel build
```bash
cd dtf/custom-op
bazel build --copt='-D_GLIBCXX_USE_CXX11_ABI=0' //zero_out:zero_out
```

## Test
```bash
cd dtf/custom-op
export LD_LIBRARY_PATH=/data00/home/son.nguyen/.pyenv/versions/3.8.0/lib/python3.8/site-packages/tensorflow
export TF_CPP_MAX_VLOG_LEVEL=3
python zero_out/test.py
```

## Appendix
The `--copt='-D_GLIBCXX_USE_CXX11_ABI=0'` helps resolve the following error:
```
tensorflow.python.framework.errors_impl.NotFoundError: bazel-bin/zero_out/libzero_out.so: undefined symbol: _ZNK10tensorflow8OpKernel11TraceStringB5cxx11ERKNS_15OpKernelContextEb
```
`-D_GLIBCXX_USE_CXX11_ABI=0` disables new C++11 ABI. For more info, please check out https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
