/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in1 compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in1 writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/


#include "stupid.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void StupidCudaKernel(const int size, const T* in, T* out) {
  if (in[0] < 0.5) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += blockDim.x * gridDim.x) {
      out[i] = 100 * in[i];
    }
  } else {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    while (true) {
      out[i] = 2 * in[i];
      i++;
    }
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct StupidFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const GPUDevice& d, int size, const T* in, T* out) {
    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    int block_count = 1024;
    int thread_per_block = 20;
    std::cout << "Launch StupidCudaKernel" << std::endl;
    StupidCudaKernel<T><<<block_count, thread_per_block, 0, d.stream()>>>(size, in, out);
    std::cout << "Scheduled StupidCudaKernel" << std::endl;
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct StupidFunctor<GPUDevice, float>;

} // namespace functor
} // namespace tensorflow
