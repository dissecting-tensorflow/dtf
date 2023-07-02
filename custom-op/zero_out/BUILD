load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "zero_out",
    srcs = [
        "zero_out_op.cc",
        "zero_out_kernel_cpu.cc"
    ],
    linkshared = 1,
    copts = ["-Itensorflow/include"],
    deps = [
        "//tensorflow:include",
        "//tensorflow:libtensorflow_framework",
    ],
)
