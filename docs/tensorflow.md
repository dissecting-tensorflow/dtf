# TensorFlow Internals
## How does TensorFlow internally map an operation in computational graph to the corresponding registered C++ operation?
### Operation Registration in C++:
TensorFlow uses the `REGISTER_OP` macro to register operations. This registration includes the operation's name, input types, output types, and shape functions.
```C++
REGISTER_OP("Add")
    .Input("a: T")
    .Input("b: T")
    .Output("sum: T")
    .Attr("T: {float, double, int32, int64} = DT_FLOAT")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
        shape_inference::ShapeHandle shape;
        TF_RETURN_IF_ERROR(c->MergeInputShapes(0, 1, &shape));
        c->set_output(0, shape);
        return Status::OK();
    });
```

#### How REGISTER_OP Works
* Parsing the Macro: When `REGISTER_OP` is invoked, it parses the provided operation name, inputs, outputs, attributes, and shape function.
* Creating the `OpDef`: TensorFlow creates an `OpDef` structure that describes the operation based on the information provided in `REGISTER_OP`.
* Adding to Registry: The `OpDef` is added to TensorFlow's global operation registry, making the operation available for use in both Python and C++ code.
* Shape Inference: If a shape inference function is provided, it is used to compute the shapes of the output tensors during graph construction.

### Kernel Registration:
TensorFlow uses the `REGISTER_KERNEL_BUILDER` macro to register the implementation of the operation for different devices (e.g., CPU, GPU).
```C++
REGISTER_KERNEL_BUILDER(Name("Add").Device(DEVICE_CPU), AddOp);
REGISTER_KERNEL_BUILDER(Name("Add").Device(DEVICE_GPU), AddOpGpu);
```

### Kernel Implementation:
Each kernel implements the actual computation of the operation. This is done by overriding the Compute method of the OpKernel class.
```C++
class AddOp : public OpKernel {
public:
    explicit AddOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& a = context->input(0);
        const Tensor& b = context->input(1);
        Tensor* sum = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, a.shape(), &sum));

        auto a_flat = a.flat<float>();
        auto b_flat = b.flat<float>();
        auto sum_flat = sum->flat<float>();

        for (int i = 0; i < a_flat.size(); ++i) {
            sum_flat(i) = a_flat(i) + b_flat(i);
        }
    }
};
```

### Graph Construction:
TensorFlow constructs a GraphDef from the Python code. Each operation in the Python code corresponds to a node in the GraphDef, with the operation's name matching the registered C++ operation name.
```pbtxt
node {
    name: "Add"
    op: "Add"
    input: "a"
    input: "b"
    attr { key: "T" value { type: DT_FLOAT } }
}
```

### Graph Execution:
* During execution, TensorFlow uses the GraphDef protocol buffer, which contains serialized information about the computational graph, including operation types and dependencies.
* The TensorFlow executor reads the GraphDef and uses the operation names (e.g., "Add") to find the corresponding C++ implementation.

#### Operation Lookup:
* When TensorFlow executes the graph, it looks up the operation name ("Add") in its registry to find the corresponding C++ OpKernel implementation.
* TensorFlow maintains a registry of all operations and their kernels. This registry maps operation names to their corresponding OpKernel classes.
  ```C++
  static std::map<std::string, std::unique_ptr<OpKernel>> kernel_registry;
  ```

#### Kernel Invocation:
* The TensorFlow executor invokes the Compute method of the OpKernel class associated with the operation name. This method performs the actual computation.
  ```C++
  OpKernel* kernel = kernel_registry["Add"].get();
  kernel->Compute(context);
  ```


# TensorFlow and MLIR
To optimize a TensorFlow graph using MLIR, you would typically follow these steps:

1. Parse the TensorFlow graph: Use a library like TensorFlow's GraphDef parser to load the graph from the provided pbtxt format into a format that can be manipulated by MLIR.

2. Convert the TensorFlow graph to MLIR: Write a converter that translates the TensorFlow graph into MLIR. This involves mapping TensorFlow operations to corresponding MLIR operations and handling any differences in semantics or representation between the two.

3. Optimize the MLIR graph: Once you have the TensorFlow graph represented in MLIR, you can use MLIR's optimization passes to improve its performance. MLIR provides a variety of passes for common optimizations such as constant folding, dead code elimination, and loop optimization.

4. Convert the optimized MLIR graph back to TensorFlow: Finally, convert the optimized MLIR graph back to TensorFlow's representation if needed. This step may involve writing a reverse converter that maps MLIR operations back to TensorFlow operations.

Here's a high-level overview of what the code might look like:
```C++
// Assume you have parsed the TensorFlow graph into 'graphDef' and 'context' is your MLIR context.

// Create an MLIR module to hold the graph.
mlir::OwningModuleRef module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&context)));

// Convert the TensorFlow graph to MLIR.
convertTensorFlowToMLIR(graphDef, context, module);

// Optimize the MLIR graph.
optimizeMLIR(module.get());

// Convert the optimized MLIR graph back to TensorFlow if needed.
convertMLIRToTensorFlow(module.get(), optimizedGraphDef);
```

Note that writing the `convertTensorFlowToMLIR`, `optimizeMLIR`, and `convertMLIRToTensorFlow` functions would require a deep understanding of both TensorFlow and MLIR, as well as the ability to map operations and their semantics between the two frameworks.

Original TensorFlow GraphDef -> TensorFlow Dialect + MLIR -> MLIR IR -> Analyze -> Transform/Optimize -> MLIR IR -> TensorFlow Dialect + MLIR -> Optimized TensorFlow GraphDef

## Types
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/types.proto


# Get compile flags
```Python
import tensorflow as tf
tf.sysconfig.get_compile_flags()
```

# Load a solib
```Python
python3

# Copy and paste the following code snippet into the python3 console
import os
import sys
import json
import logging
import tensorflow.compat.v1 as tf

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.info("This is a dummy message!")

logging.info("cwd {}".format(os.getcwd()))

solib = "./libcustom_ops_gpu_abi0.so"

logging.info("Loading solib {}".format(solib))

custom_op = tf.load_op_library(solib)
print(json.dumps(dir(custom_op), indent=2))
```

# Device ids
/data00/son.nguyen/workspace/tensorflow/tensorflow/core/common_runtime/device/device_id.h
```C++
// There are three types of device ids:
// - *physical* device id: this is the integer index of a device in the
//   physical machine, it can be filtered (for e.g. using environment variable
//   CUDA_VISIBLE_DEVICES when using CUDA). Note that this id is not visible to
//   Tensorflow, but result after filtering is visible to TF and is called
//   platform device id as below.
//   For CUDA, see
//   http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
//   for more details.
// - *platform* device id (also called *visible* device id in
//   third_party/tensorflow/core/protobuf/config.proto): this is the id that is
//   visible to Tensorflow after filtering (for e.g. by CUDA_VISIBLE_DEVICES).
//   For CUDA, this id is generated by the CUDA GPU driver. It starts from 0
//   and is used for CUDA API calls like cuDeviceGet().
// - TF device id (also called *virtual* device id in
//   third_party/tensorflow/core/protobuf/config.proto): this is the id that
//   Tensorflow generates and exposes to its users. It is the id in the <id>
//   field of the device name "/device:GPU:<id>", and is also the identifier of
//   a BaseGPUDevice. Note that the configuration allows us to create multiple
//   BaseGPUDevice per GPU hardware in order to use multi CUDA streams on the
//   hardware, so the mapping between TF GPU id and platform GPU id is not a 1:1
//   mapping, see the example below.
//
// For example, assuming that in the machine we have GPU device with index 0, 1,
// 2 and 3 (physical GPU id). Setting "CUDA_VISIBLE_DEVICES=1,2,3" will create
// the following mapping between platform GPU id and physical GPU id:
//
//        platform GPU id ->  physical GPU id
//                 0  ->  1
//                 1  ->  2
//                 2  ->  3
//
// Note that physical GPU id 0 is invisible to TF so there is no mapping entry
// for it.
//
// Assuming we configure the Session to create one BaseGPUDevice per GPU
// hardware, then setting GPUOptions::visible_device_list to "2,0" will create
// the following mapping between TF device id and platform device id:
//
//                  TF GPU id  ->  platform GPU ID
//      0 (i.e. /device:GPU:0) ->  2
//      1 (i.e. /device:GPU:1) ->  0
//
// Note that platform device id 1 is filtered out by
// GPUOptions::visible_device_list, so it won't be used by the TF process.
//
// On the other hand, if we configure it to create 2 BaseGPUDevice per GPU
// hardware, then setting GPUOptions::visible_device_list to "2,0" will create
// the following mapping between TF device id and platform device id:
//
//                  TF GPU id  ->  platform GPU ID
//      0 (i.e. /device:GPU:0) ->  2
//      1 (i.e. /device:GPU:1) ->  2
//      2 (i.e. /device:GPU:2) ->  0
//      3 (i.e. /device:GPU:3) ->  0
//
// We create strong-typed integer classes for both TF device id and platform
// device id to minimize programming errors and improve code readability. Except
// for the StreamExecutor interface (as we don't change its API), whenever we
// need a TF device id (or platform device id) we should use TfDeviceId (or
// PlatformDeviceId) instead of a raw integer.
TF_LIB_GTL_DEFINE_INT_TYPE(TfDeviceId, int32);
// Expands to:
// struct TfDeviceId_tag_ {}; typedef ::tensorflow::gtl::IntType<TfDeviceId_tag_, int32> TfDeviceId;

TF_LIB_GTL_DEFINE_INT_TYPE(PlatformDeviceId, int32);
// Expands to:
// struct PlatformDeviceId_tag_ {}; typedef ::tensorflow::gtl::IntType<PlatformDeviceId_tag_, int32> PlatformDeviceId;
```

# Search for a kernel
Keyword: Name("OP_NAME"). For example, <br/>
Name("Sum")

# Profiling
```Bash
# Replace tf profiler with xprof:
pip install xprof
rm -rf profiler/pilot && mkdir -p profiler/pilot
xprof --logdir=profiler/pilot --port=6006

# On BackBook
ssh -L 6006:localhost:6006 gpudev_xx
```

# Device Placement Debugging
```Bash
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_VLOG_LEVEL=3
export TF_LOG_DEVICE_PLACEMENT=1
export TF_DUMP_GRAPH_PREFIX=/path/to/tf_dump_dir
```

# Edge
The edge `edge_517_cross_former_0_block_0_1/attn_1/strided_slice_15` is the output of node `cross_former_0_block_0_1/attn_1/strided_slice_15`.
```Bash
2025-12-02 00:50:06.446249: I tensorflow/core/common_runtime/executor.cc:764] Process node: 292 step 2 {{node cross_former_0_block_0_1/attn_1/strided_slice_15}} = StridedSlice[Index=DT_INT32, T=DT_FLOAT, _XlaHasReferenceVars=false, _byted_output_shapes=[[?,1024,384]], _output_shapes=[[?,1024,384]], _symbolic_output_shapes=[[1,51,384]], begin_mask=1, ellipsis_mask=0, end_mask=1, new_axis_mask=0, shrink_axis_mask=0, _device="/job:localhost/replica:0/task:0/device:GPU:0"](cross_former_0_block_0_1/attn_1/strided_slice_14, bias_slice/begin, cross_former_0_block_0_1/attn_1/strided_slice_7/stack_1, mix_former/mix_former_block_.1/mix_former_block_.1_attn_1/strided_slice_10/stack_2) device: /job:localhost/replica:0/task:0/device:GPU:0
2025-12-02 00:50:06.446255: I tensorflow/core/common_runtime/gpu/gpu_device.cc:587] GpuDevice::ComputeHelper cross_former_0_block_0_1/attn_1/strided_slice_15 op StridedSlice on GPU 0 stream[0]
2025-12-02 00:50:06.446261: I tensorflow/core/kernels/strided_slice_op.cc:113] Strided slice identity 
2025-12-02 00:50:06.446265: I tensorflow/core/common_runtime/gpu/gpu_device.cc:617] GpuDevice::ComputeHelper scheduled cross_former_0_block_0_1/attn_1/strided_slice_15 op StridedSlice on GPU 0 stream[0]
2025-12-02 00:50:06.446276: I tensorflow/core/common_runtime/executor.cc:764] Process node: 293 step 2 {{node cross_former_0_block_0_1/attn_1/strided_slice_15/_280}} = _Send[T=DT_FLOAT, _dst="_retval_cross_former_0_block_0_1/attn_1/strided_slice_15_0_0", _src="cross_former_0_block_0_1/attn_1/strided_slice_15", client_terminated=false, recv_device="/job:localhost/replica:0/task:0/device:CPU:0", send_device="/job:localhost/replica:0/task:0/device:GPU:0", send_device_incarnation=1, tensor_name="edge_517_cross_former_0_block_0_1/attn_1/strided_slice_15", _device="/job:localhost/replica:0/task:0/device:GPU:0"](cross_former_0_block_0_1/attn_1/strided_slice_15) device: /job:localhost/replica:0/task:0/device:GPU:0
2025-12-02 00:50:06.446283: I tensorflow/core/common_runtime/gpu/gpu_device.cc:587] GpuDevice::ComputeHelper cross_former_0_block_0_1/attn_1/strided_slice_15/_280 op _Send on GPU 0 stream[0]
2025-12-02 00:50:06.446289: I tensorflow/core/common_runtime/rendezvous_mgr.cc:180] IntraProcessRendezvous Send 0x96e23fb0 /job:localhost/replica:0/task:0/device:GPU:0;0000000000000001;/job:localhost/replica:0/task:0/device:CPU:0;edge_517_cross_former_0_block_0_1/attn_1/strided_slice_15;0:0
2025-12-02 00:50:06.446298: I tensorflow/core/common_runtime/bfc_allocator.cc:310] AllocateRaw gpu_host_bfc  78336
2025-12-02 00:50:06.446304: I tensorflow/core/common_runtime/copy_tensor.cc:211] Copy edge_517_cross_former_0_block_0_1/attn_1/strided_slice_15
2025-12-02 00:50:06.446310: I tensorflow/core/common_runtime/gpu/gpu_util.cc:258] CopyGPUTensorToCPU
2025-12-02 00:50:06.446318: I tensorflow/stream_executor/stream.cc:1397] [stream=0x91936450,impl=0x35ae5b70] Called Stream::ThenWaitFor(other=0x35ae0770)
2025-12-02 00:50:06.446330: I tensorflow/stream_executor/stream.cc:4563] [stream=0x91936450,impl=0x35ae5b70] Called Stream::ThenMemcpy(host_dst=0x7fd28de00900, gpu_src=0x7fd28ebc6300, size=78336)
2025-12-02 00:50:06.446348: I tensorflow/stream_executor/stream.cc:330] [stream=0x91936450,impl=0x35ae5b70] Called Stream::ThenRecordEvent(event=0x7fd20400cd50)
2025-12-02 00:50:06.446366: I tensorflow/core/common_runtime/gpu/gpu_device.cc:617] GpuDevice::ComputeHelper scheduled cross_former_0_block_0_1/attn_1/strided_slice_15/_280 op _Send on GPU 0 stream[0]
```