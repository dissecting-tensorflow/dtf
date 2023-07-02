load("@rules_cc//cc:defs.bzl", "cc_library")

cc_library(
    name = "include",
    hdrs = glob(["include/**"]),
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libtensorflow_framework",
    srcs = ["lib/libtensorflow_framework.so.2"],
    visibility = ["//visibility:public"],
)
