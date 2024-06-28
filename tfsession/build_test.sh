#!/bin/bash

bazel build -c dbg //:test --verbose_failures --sandbox_debug
if [ "$?" == "0" ]; then
  cp -vf bazel-bin/test test
fi