#include <iostream>
#include <tensorflow/core/platform/cpu_info.h>

int main(int argc, char** argv) {
  int m = tensorflow::port::MaxParallelism();
  std::cout << "port::MaxParallelism() = " << m << std::endl;
}
