#!/bin/bash

bazel build -c dbg //:main --verbose_failures --sandbox_debug
bazel build -c dbg //:test --verbose_failures --sandbox_debug