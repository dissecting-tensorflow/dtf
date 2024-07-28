export BAZEL_CXXOPTS="-D_GLIBCXX_USE_CXX11_ABI=0"
FILE_PATTERNS=+tensorflow/core/framework/op_kernel.cc,+tensorflow/core/framework/kernel_def.pb.cc,+tensorflow/core/framework/op.cc
FILE_PATTERNS=${FILE_PATTERNS},+tensorflow/compiler/tf2xla/xla_op_registry.cc
DESIRED_FLAGS=@-O0,-g,-fno-inline
PER_FILE_OPTS=${FILE_PATTERNS}${DESIRED_FLAGS}
bazel build --subcommands --config=cuda --per_file_copt=${PER_FILE_OPTS} --linkopt=-g --strip=never //tensorflow:libtensorflow_cc.so --verbose_failures --jobs 128
