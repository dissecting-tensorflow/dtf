#pragma once

#include "tensorflow/core/framework/op_kernel.h"

#define EIGEN_USE_GPU

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct StupidFunctor {
  void operator()(OpKernelContext* context, const Device& d, int size, const T* in, T* out);
};

} // namespace functor
} // namespace tensorflow
