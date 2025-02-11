# Find op name
```Bash
export TF_CPP_MAX_VLOG_LEVEL=3
python
```

For example, StridedSlice op:
```Python
>>> import tensorflow as tf
>>> a = tf.constant([1, 2, 3, 4, 5])
2025-02-11 06:37:44.473192: I tensorflow/core/framework/log_memory.cc:34] __LOG_MEMORY__ MemoryLogTensorAllocation { step_id: -6 kernel_name: "Unknown" tensor { dtype: DT_INT32 shape { dim { size: 5 } } allocation_description { requested_bytes: 20 allocated_bytes: 20 allocator_name: "cpu" allocation_id: 5 has_single_reference: true ptr: 2377352512 } } }
>>> e1 = a[1]
...
2025-02-11 06:37:50.503771: I tensorflow/core/framework/op_kernel.cc:1626] Instantiating kernel for node: {{node StridedSlice}} = StridedSlice[Index=DT_INT32, T=DT_INT32, begin_mask=0, ellipsis_mask=0, end_mask=0, new_axis_mask=0, shrink_axis_mask=1](dummy_input, dummy_input, dummy_input, dummy_input)
2025-02-11 06:37:50.503810: I tensorflow/core/common_runtime/eager/execute.cc:733] Executing op StridedSlice in device /job:localhost/replica:0/task:0/device:GPU:0
2025-02-11 06:37:50.503834: I tensorflow/core/common_runtime/gpu/gpu_device.cc:587] GpuDevice::ComputeHelper StridedSlice op StridedSlice on GPU 0 stream[0]
2025-02-11 06:37:50.503844: I tensorflow/stream_executor/cuda/cuda_driver.cc:250] ScopedActivateContext switching to 1
2025-02-11 06:37:50.503876: I tensorflow/core/framework/log_memory.cc:34] __LOG_MEMORY__ MemoryLogTensorAllocation { kernel_name: "StridedSlice" tensor { dtype: DT_INT32 shape { } allocation_description { requested_bytes: 4 allocated_bytes: 4 allocator_name: "cpu" allocation_id: 9 has_single_reference: true ptr: 2377363328 } } }
2025-02-11 06:37:50.503909: I tensorflow/core/common_runtime/gpu/gpu_device.cc:617] GpuDevice::ComputeHelper scheduled StridedSlice op StridedSlice on GPU 0 stream[0]
...
```
<br/>

Switch op:
```Python
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
x_0, x_1 = control_flow_ops.switch(tf.constant(2), False)

2025-02-11 06:41:12.585707: I tensorflow/core/framework/op_kernel.cc:1626] Instantiating kernel for node: {{node Switch}} = Switch[T=DT_INT32](dummy_input, dummy_input)
2025-02-11 06:41:12.585740: I tensorflow/core/common_runtime/eager/execute.cc:733] Executing op Switch in device /job:localhost/replica:0/task:0/device:GPU:0
2025-02-11 06:41:12.591515: I tensorflow/core/common_runtime/bfc_allocator.cc:276] AllocateRaw GPU_0_bfc  1028
2025-02-11 06:41:12.591527: I tensorflow/stream_executor/cuda/cuda_driver.cc:250] ScopedActivateContext switching to 1
2025-02-11 06:41:12.603266: I tensorflow/stream_executor/cuda/cuda_driver.cc:885] allocated 0x7f65fe000000 for context 0x4645ce0 of 21544435712 bytes
2025-02-11 06:41:12.603291: I tensorflow/stream_executor/stream_executor_pimpl.cc:515] Called StreamExecutor::Allocate(size=21544435712, memory_space=0) returns 0x7f65fe000000
2025-02-11 06:41:12.603301: I tensorflow/core/common_runtime/bfc_allocator.cc:174] Extending allocation by 20.06GiB bytes.
2025-02-11 06:41:12.603307: I tensorflow/core/common_runtime/bfc_allocator.cc:178] Total allocated bytes: 20.06GiB
2025-02-11 06:41:12.603312: I tensorflow/core/common_runtime/bfc_allocator.cc:181] Allocated memory at 0x7f65fe000000 to 0x7f6b02260000
2025-02-11 06:41:12.826907: I tensorflow/stream_executor/stream_executor_pimpl.cc:611] Called StreamExecutor::SynchronousMemZero(location=0x7ffef461b980, size=1028)
2025-02-11 06:41:12.826934: I tensorflow/stream_executor/cuda/cuda_driver.cc:250] ScopedActivateContext switching to 1
2025-02-11 06:41:12.826989: I tensorflow/core/common_runtime/gpu/gpu_device.cc:587] GpuDevice::ComputeHelper Switch op Switch on GPU 0 stream[0]
```
