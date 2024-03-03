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

## Session Run
2023-12-24 16:15:57.879394: W tensorflow/core/util/dump_graph.cc:134] Failed to dump encapsulate_subgraphs_after because dump location is not  specified through either TF_DUMP_GRAPH_PREFIX environment variable or function argument.
2023-12-24 16:15:57.879603: I tensorflow/compiler/jit/xla_cluster_util.cc:559] # iterations = 1
2023-12-24 16:15:57.879680: I tensorflow/compiler/jit/xla_cluster_util.cc:559] # iterations = 1
2023-12-24 16:15:57.879690: I tensorflow/compiler/jit/xla_cluster_util.cc:591] GetNodesRelatedToRefVariables() found 0 nodes
2023-12-24 16:15:57.879794: I tensorflow/compiler/jit/encapsulate_subgraphs_pass.cc:1307] Has ref vars = 0, node: name: "_SOURCE"
op: "NoOp"
attr {
  key: "_XlaHasReferenceVars"
  value {
    b: false
  }
}

2023-12-24 16:15:57.879830: I tensorflow/compiler/jit/encapsulate_subgraphs_pass.cc:1307] Has ref vars = 0, node: name: "_SINK"
op: "NoOp"
attr {
  key: "_XlaHasReferenceVars"
  value {
    b: false
  }
}

2023-12-24 16:15:57.879929: I tensorflow/compiler/jit/encapsulate_subgraphs_pass.cc:1307] Has ref vars = 0, node: name: "Weight"
op: "Const"
device: "/device:GPU:0"
attr {
  key: "_XlaHasReferenceVars"
  value {
    b: false
  }
}
attr {
  key: "dtype"
  value {
    type: DT_FLOAT
  }
}
attr {
  key: "value"
  value {
    tensor {
      dtype: DT_FLOAT
      tensor_shape {
        dim {
          size: 5
        }
        dim {
          size: 7
        }
      }
      tensor_content: "\0004\232<\300p-\274a \205<\024\331\007;B\373g\274\223\323C;\346C\356\273\244]\260:m\331\263\273\205K\252;\345\013\350:\341\236\r<a%\230<\242\262\023<@n\033< Z\322:\362\275\222;\333\270\003\274\343\232\005\273\227@\367\273\2234\351\273\031>\002\273\317\"\035\272\327:\026<+\301T\274T\300\036\274\241g\216<U\210\262\272F\003\250\273\373\224\000\274!8\220<\200\032!<,-Z<\324I3\274\2547\210:"
    }
  }
}



2023-12-17 14:29:17.026136: I tensorflow/core/common_runtime/direct_session.cc:1720] Created 
() -> () {
  n4 = Const[_XlaHasReferenceVars=false, dtype=float, value=Tensor<type: float shape: [7] values: 0 0 0...>, device=GPU:0]()
  n5 = Identity[T=float, _XlaHasReferenceVars=false, _class=["loc:@Bias"], device=GPU:0](n4)
  n2 = Const[_XlaHasReferenceVars=false, dtype=float, value=Tensor<type: float shape: [5,7] values: [0.0188236237 -0.0105859637 0.016250791...]...>, device=GPU:0]()
  n3 = Identity[T=float, _XlaHasReferenceVars=false, _class=["loc:@Weight"], device=GPU:0](n2)
  n6 = _Recv[_dst="MatMul", _src="_arg_X_0_0", client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device_incarnation=1, tensor_name="edge_13__arg_X_0_0", tensor_type=float, device=GPU:0]()
  n7 = MatMul[T=float, _XlaHasReferenceVars=false, transpose_a=false, transpose_b=false, device=GPU:0](n6, n3)
  n8 = Add[T=float, _XlaHasReferenceVars=false, device=GPU:0](n7, n5)
  n9 = Sigmoid[T=float, _XlaHasReferenceVars=false, device=GPU:0](n8)
  n10 = _Send[T=float, _dst="_retval_Sigmoid_0_0", _src="Sigmoid", client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_14_Sigmoid", device=GPU:0](n9)
}
 for /job:localhost/replica:0/task:0/device:GPU:0
2023-12-17 14:29:17.026186: I tensorflow/core/common_runtime/direct_session.cc:1720] Created 
(n2:float@CPU:0) -> (n4:float@CPU:0) {
  n4 = _Recv[_dst="_retval_Sigmoid_0_0", _src="Sigmoid", client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_14_Sigmoid", tensor_type=float, device=CPU:0]()
  n3 = _Send[T=float, _dst="MatMul", _src="_arg_X_0_0", client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device_incarnation=1, tensor_name="edge_13__arg_X_0_0", device=CPU:0](n2)
}
 for /job:localhost/replica:0/task:0/device:CPU:0
2023-12-17 14:29:17.026232: I tensorflow/core/common_runtime/function_utils.cc:78] Graph Initial #nodes 11 #edges 13
2023-12-17 14:29:17.026239: I tensorflow/core/common_runtime/function_utils.cc:164] Removing list array converter
2023-12-17 14:29:17.026262: I tensorflow/core/common_runtime/function_utils.cc:78] Graph ReCopy #nodes 11 #edges 14

# GPU
info functions BaseBatchMatMulOp<Eigen::GpuDevice, float, float, float>::Compute
info functions BaseBatchMatMulOp<Eigen::GpuDevice, float, float, float>::Launch

info functions LaunchBatchMatMul<Eigen::GpuDevice, float>::Launch
info functions LaunchBatchMatMul<Eigen::GpuDevice, Eigen::half>::Launch

# CPU
info functions tensorflow::LaunchBatchMatMul<Eigen::ThreadPoolDevice, float>::Launch(tensorflow::OpKernelContext*

<!-- Call Stack -->
libtensorflow_framework.so.2!stream_executor::gpu::CUDABlas::DoBlasInternalImpl<cublasStatus_t (*)(cublasContext*, cublasOperation_t, cublasOperation_t, int, int, int, float const*, float const*, int, float const*, int, float const*, float*, int), cublasOperation_t, cublasOperation_t, unsigned long, unsigned long, unsigned long, float const*, float const*, int, float const*, int, float const*, float*, int>(stream_executor::gpu::CUDABlas * const this, cublasStatus_t (*)(cublasContext *, cublasOperation_t, cublasOperation_t, int, int, int, const float *, const float *, int, const float *, int, const float *, float *, int) cublas_func, stream_executor::Stream * stream, bool pointer_mode_host, cublasMath_t math_type,  args#0,  args#1,  args#2,  args#3,  args#4,  args#5,  args#6,  args#7,  args#8,  args#9,  args#10,  args#11,  args#12) (/data00/home/son.nguyen/workspace/tensorflow_dev/tensorflow/tensorflow/stream_executor/cuda/cuda_blas.cc:573)
libtensorflow_framework.so.2!stream_executor::gpu::CUDABlas::DoBlasGemm(stream_executor::gpu::CUDABlas * const this, stream_executor::Stream * stream, stream_executor::blas::Transpose transa, stream_executor::blas::Transpose transb, uint64_t m, tensorflow::uint64 n, uint64_t k, stream_executor::dnn::DataType dtype, const void * alpha, const stream_executor::DeviceMemoryBase & a, int lda, const stream_executor::DeviceMemoryBase & b, int ldb, const void * beta, stream_executor::DeviceMemoryBase * c, int ldc) (/data00/home/son.nguyen/workspace/tensorflow_dev/tensorflow/tensorflow/stream_executor/cuda/cuda_blas.cc:1861)
libtensorflow_cc.so.2!tensorflow::Status stream_executor::Stream::ThenBlasGemm<float, float>(stream_executor::blas::Transpose, stream_executor::blas::Transpose, unsigned long, unsigned long, unsigned long, float, stream_executor::DeviceMemory<float> const&, int, stream_executor::DeviceMemory<float> const&, int, float, stream_executor::DeviceMemory<float>*, int) (Unknown Source:0)
libtensorflow_cc.so.2!stream_executor::Stream::ThenBlasGemm<float>(stream_executor::Stream * const this, stream_executor::blas::Transpose transa, stream_executor::blas::Transpose transb, uint64_t m, tensorflow::uint64 n, tensorflow::uint64 k, const stream_executor::DeviceMemory<float> & a, int lda, const stream_executor::DeviceMemory<float> & b, int ldb, stream_executor::DeviceMemory<float> * c, int ldc) (/data00/home/son.nguyen/workspace/tensorflow_dev/tensorflow/tensorflow/stream_executor/stream.h:1207)
libtensorflow_cc.so.2!tensorflow::BaseBatchMatMulOp<Eigen::GpuDevice, float, float, float>::Launch(tensorflow::OpKernelContext * context, const tensorflow::Tensor & in_x, const tensorflow::Tensor & in_y, bool adj_x, bool adj_y, bool trans_x, bool trans_y, const tensorflow::MatMulBCast & bcast, tensorflow::Tensor * out) (/data00/home/son.nguyen/workspace/tensorflow_dev/tensorflow/tensorflow/core/kernels/matmul_op_impl.h:902)
libtensorflow_cc.so.2!tensorflow::BaseBatchMatMulOp<Eigen::GpuDevice, float, float, float>::Compute(tensorflow::BaseBatchMatMulOp<Eigen::GpuDevice, float, float, float> * const this, tensorflow::OpKernelContext * ctx) (/data00/home/son.nguyen/workspace/tensorflow_dev/tensorflow/tensorflow/core/kernels/matmul_op_impl.h:1034)
libtensorflow_framework.so.2!tensorflow::BaseGPUDevice::Compute(tensorflow::BaseGPUDevice * const this, tensorflow::OpKernel * op_kernel, tensorflow::OpKernelContext * context) (/data00/home/son.nguyen/workspace/tensorflow_dev/tensorflow/tensorflow/core/common_runtime/gpu/gpu_device.cc:679)
libtensorflow_framework.so.2!tensorflow::(anonymous namespace)::ExecutorState<tensorflow::PropagatorState>::ProcessSync(tensorflow::(anonymous namespace)::ExecutorState<tensorflow::PropagatorState> * const this, const tensorflow::NodeItem & item, tensorflow::OpKernelContext::Params * params, tensorflow::EntryVector * outputs, tensorflow::NodeExecStatsInterface * stats) (/data00/home/son.nguyen/workspace/tensorflow_dev/tensorflow/tensorflow/core/common_runtime/executor.cc:597)
libtensorflow_framework.so.2!tensorflow::(anonymous namespace)::ExecutorState<tensorflow::PropagatorState>::Process(tensorflow::(anonymous namespace)::ExecutorState<tensorflow::PropagatorState> * const this, tensorflow::(anonymous namespace)::ExecutorState<tensorflow::PropagatorState>::TaggedNode tagged_node, int64_t scheduled_nsec) (/data00/home/son.nguyen/workspace/tensorflow_dev/tensorflow/tensorflow/core/common_runtime/executor.cc:830)
libtensorflow_framework.so.2!tensorflow::(anonymous namespace)::ExecutorState<tensorflow::PropagatorState>::<lambda()>::operator()(void) const(tensorflow::(anonymous namespace)::ExecutorState<tensorflow::PropagatorState> * const this) (/data00/home/son.nguyen/workspace/tensorflow_dev/tensorflow/tensorflow/core/common_runtime/executor.cc:1197)
libtensorflow_framework.so.2!tensorflow::(anonymous namespace)::ExecutorState<tensorflow::PropagatorState>::<lambda()>::operator()(void)(tensorflow::(anonymous namespace)::ExecutorState<tensorflow::PropagatorState>::<lambda()> * const this) (/data00/home/son.nguyen/workspace/tensorflow_dev/tensorflow/tensorflow/core/common_runtime/executor.cc:468)
libtensorflow_framework.so.2!std::_Function_handler<void(), tensorflow::(anonymous namespace)::ExecutorState<PropagatorStateType>::RunTask(Closure&&) [with Closure = tensorflow::(anonymous namespace)::ExecutorState<PropagatorStateType>::ScheduleReady(tensorflow::(anonymous namespace)::ExecutorState<PropagatorStateType>::TaggedNodeSeq*, tensorflow::(anonymous namespace)::ExecutorState<PropagatorStateType>::TaggedNodeReadyQueue*) [with PropagatorStateType = tensorflow::PropagatorState; tensorflow::(anonymous namespace)::ExecutorState<PropagatorStateType>::TaggedNodeSeq = absl::lts_20211102::InlinedVector<tensorflow::PropagatorState::TaggedNode, 8>; tensorflow::(anonymous namespace)::ExecutorState<PropagatorStateType>::TaggedNodeReadyQueue = tensorflow::PropagatorState::TaggedNodeReadyQueue]::<lambda()>; PropagatorStateType = tensorflow::PropagatorState]::<lambda()> >::_M_invoke(const std::_Any_data &)(const std::_Any_data & __functor) (/usr/include/c++/8/bits/std_function.h:297)
libtensorflow_cc.so.2!Eigen::ThreadPoolTempl<tensorflow::thread::EigenEnvironment>::WorkerLoop(int) (Unknown Source:0)
libtensorflow_cc.so.2!std::_Function_handler<void (), tensorflow::thread::EigenEnvironment::CreateThread(std::function<void ()>)::{lambda()#1}>::_M_invoke(std::_Any_data const&) (Unknown Source:0)
libtensorflow_framework.so.2!tensorflow::(anonymous namespace)::PThread::ThreadFn(void*)() (/data00/home/son.nguyen/workspace/tensorflow_dev/tensorflow/tensorflow/core/platform/default/logging.h:374)
libpthread.so.0!start_thread(void * arg) (/build/glibc-6iIyft/glibc-2.28/nptl/pthread_create.c:486)
libc.so.6!clone() (/build/glibc-6iIyft/glibc-2.28/sysdeps/unix/sysv/linux/x86_64/clone.S:95)

rb tensorflow::BaseGPUDevice::Compute(tensorflow::BaseGPUDevice * const this, tensorflow::OpKernel * op_kernel, tensorflow::OpKernelContext * context)
p op_kernel.name()