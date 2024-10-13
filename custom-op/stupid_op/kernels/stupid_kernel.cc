#include "stupid.h"

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class StupidOp : public OpKernel {
 public:
  explicit StupidOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));

    // Do the computation.
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max, errors::InvalidArgument("Too many elements in tensor"));
    LOG(INFO) << "Execute StupidFunctor";
    StupidFunctor<Device, T>()(
      context,
      context->eigen_device<Device>(),
      static_cast<int>(input_tensor.NumElements()),
      input_tensor.flat<T>().data(),
      output_tensor->flat<T>().data()
    );
  }
};

// Register the GPU kernels.
REGISTER_KERNEL_BUILDER(
  Name("Stupid").Device(DEVICE_GPU)
    .TypeConstraint<float>("T"),
  StupidOp<GPUDevice, float>
);

} // namespace functor
} // namespace tensorflow
