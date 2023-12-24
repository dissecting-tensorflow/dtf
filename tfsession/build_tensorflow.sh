# https://www.tensorflow.org/install/source

# TensorFlow
git clone https://github.com/tensorflow/tensorflow.git
git checkout -b v2.9.0 tags/v2.9.0

# Install Python and the TensorFlow package dependencies
sudo apt install python3-dev python3-pip
pip install -U pip numpy wheel packaging requests opt_einsum
pip install -U keras_preprocessing --no-deps

# Install Bazel
go install github.com/bazelbuild/bazelisk@latest
cp $(which bazelisk) /data00/home/son.nguyen/.local/bin/bazel

# GCC
gcc --version
gcc (GCC) 9.5.0

# Configure the build

