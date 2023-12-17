load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    copts = [
      "-Iexternal/cuda",
      "-Iexternal/protobuf/include",
      "-Iexternal/tensorflow/include"
    ],
    deps = [
      "@cuda//:cuda",
      "@protobuf//:protobuf",
      "@tensorflow//:tensorflow"
    ]
)
