#####################################################################################################################
# For CUDA 10.x
#####################################################################################################################
cu_library(
  name="stupid_op_cu100",
  srcs=["kernels/stupid_kernel.cu"],
  deps=[
    "third_party/Eigen:tf1.15.0@//third_party/Eigen:Eigen",
    "third_party/tensorflow:1.15.0-cpu@//third_party/tensorflow:tensorflow_framework",
    "third_party/cuda:10.0.130@//third_party/cuda:cublas,cuda,cudart,cufft,curand,cusolver,cusparse",
  ],
  defs=["GOOGLE_CUDA", "NDEBUG", "_GLIBCXX_USE_CXX11_ABI=0"],
  extra_flags=[
    "--expt-relaxed-constexpr",
    "-std=c++11",
    "-gencode arch=compute_75,code=sm_75",
  ],
  extra_cppflags=["-O2", "-fPIC"],
)

cc_library(
  name="stupid_op_abi0",
  srcs=["ops/stupid.cc", "kernels/stupid_kernel.cc"],
  deps=[
    ":stupid_op_cu100",
    "third_party/Eigen:tf1.15.0@//third_party/Eigen:Eigen",
    "third_party/tensorflow:1.15.0-cpu@//third_party/tensorflow:tensorflow_framework",
  ],
  defs=["GOOGLE_CUDA", "NDEBUG", "_GLIBCXX_USE_CXX11_ABI=0"],
  optimize=["-O2", "-fPIC"],
  link_all_symbols=True,
)

cc_plugin(name="stupid_op", deps=[":stupid_op_abi0"], bundle_path="./lib")


#####################################################################################################################
# For CUDA 11.4
#####################################################################################################################
cu_library(
  name="stupid_op_cu114",
  srcs=["kernels/stupid_kernel.cu"],
  deps=[
    "third_party/Eigen:tf2.5@//third_party/Eigen:Eigen",
    "third_party/cuda:cuda_11.4.4_470.82.01_linux@//third_party/cuda:cublas,cuda,cudart,cufft,curand,cusolver,cusparse",
    "third_party/tensorflow:2.5.0-cuda11.4-cudnn8.1-61-86@//third_party/tensorflow:tensorflow_framework",
  ],
  defs=["GOOGLE_CUDA", "NDEBUG"],
  extra_flags=[
    "--expt-relaxed-constexpr",
    "-std=c++11",
    "-gencode arch=compute_75,code=sm_75",
    "-gencode arch=compute_80,code=sm_80",
    "-gencode arch=compute_86,code=sm_86",
  ],
  extra_cppflags=["-O0", "-fPIC"],
)

cc_library(
  name="stupid_op_gpu",
  srcs=["ops/stupid.cc", "kernels/stupid_kernel.cc"],
  deps=[
    ":stupid_op_cu114",
    "third_party/Eigen:tf2.5@//third_party/Eigen:Eigen",
    "third_party/tensorflow:2.5.0-cuda11.4-cudnn8.1-61-86@//third_party/tensorflow:tensorflow_framework",
  ],
  defs=["GOOGLE_CUDA", "NDEBUG"],
  optimize=["-O0", "-fPIC"],
  link_all_symbols=True,
)
