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