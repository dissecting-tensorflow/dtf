#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <cstring>

#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/graph/default_device.h>

using InputTensors = std::vector<std::pair<std::string, tensorflow::Tensor>>;
using OutputTensors = std::vector<tensorflow::Tensor>;

#define LOG_ERROR(...) \
  fprintf(stderr, __VA_ARGS__); \
  fprintf(stderr, "\n")
#define LOG_INFO(...) \
  fprintf(stdout, __VA_ARGS__); \
  fprintf(stdout, "\n")

enum ExitCode {
  EXIT_CODE_0 = 0,
  EXIT_CODE_1 = 1
};

const std::string DEVICE_NAME = "/device:GPU:0";

// Read all content from file
static int read_file(const std::string& file_name, std::string& data) {
  FILE* fp = fopen(file_name.c_str(), "rb");
  if (fp == nullptr) {
    return EXIT_CODE_1;
  }

  char buffer[1024];
  int count;
  do {
    count = fread(buffer, sizeof(char), 1024, fp);
    if (count <= 0) {
      break;
    }
    data.append(buffer, count);
  } while (true);
  fclose(fp);
  return 0;
}

int main(int argc, char** argv) {  
  // Model name
  std::string model_name = "test_model_v1";

  // Model graph file
  std::string graphFile = "models/test/test_model_v1/gpu/graph.pb";
  std::cout << "Graph file: " << graphFile << std::endl;

  std::string graph_data;
  int ret = read_file(graphFile, graph_data);
  if (ret != 0) {
    std::cout << "ERROR: Load graph data failed!" << std::endl;
    return EXIT_CODE_1;
  }

  tensorflow::GraphDef graph_def;
  if (!graph_def.ParseFromString(graph_data)) {
    std::cout << "ERROR: Failed to parse graph data" << std::endl;
    return EXIT_CODE_1;
  }

  // Set default device for model graph
  tensorflow::graph::SetDefaultDevice(DEVICE_NAME, &graph_def);

  // Create tensorflow session options
  tensorflow::SessionOptions options;
  options.config.set_inter_op_parallelism_threads(10);
  options.config.set_intra_op_parallelism_threads(2);
  options.config.set_log_device_placement(true);
  options.config.set_allow_soft_placement(false);
  options.config.set_use_per_session_threads(false);
  auto* optimizer_options = options.config.mutable_graph_options()->mutable_optimizer_options();
  optimizer_options->set_opt_level(tensorflow::OptimizerOptions::L0);
    
  // Disable Grappler optimizations.
  auto rewriter_config = options.config.mutable_graph_options()->mutable_rewrite_options();
  rewriter_config->set_disable_meta_optimizer(true);

  options.config.mutable_graph_options()->mutable_optimizer_options()->set_global_jit_level(tensorflow::OptimizerOptions::OFF);
  tensorflow::GPUOptions* gpu_options = options.config.mutable_gpu_options();
  gpu_options->set_allow_growth(true);
  gpu_options->set_per_process_gpu_memory_fraction((float)42 / 100.0);
  // gpu_options->set_force_gpu_compatible(true);
  // gpu_options->set_allocator_type("BFC");

  // tensorflow::SetDefaultLocalSessionImpl(tensorflow::LocalSessionImpl::kDirectSession);
  tensorflow::Session* session;
  std::cout << "\n\n";
  std::cout << std::endl << std::endl;
  std::cout << std::string(120, '=') << std::endl;
  std::cout << "1. Create a new TF Session object" << std::endl;
  std::cout << std::string(120, '=') << std::endl;
  auto status = tensorflow::NewSession(options, &session);
  if (!status.ok()) {
    std::cout << "ERROR: Failed to new TF session! " << status.error_message() << std::endl;
    return EXIT_CODE_1;
  }
  LOG_INFO("Successfully created a new TF Session object");

  std::cout << "\n\n";
  std::cout << std::endl << std::endl;
  std::cout << std::string(120, '=') << std::endl;
  std::cout << "2. Create the Session object with graph" << std::endl;
  std::cout << std::string(120, '=') << std::endl;
  status = session->Create(graph_def);
  if (!status.ok()) {
    LOG_ERROR("Failed to create session! model=%s, err=%s", model_name.c_str(), status.error_message().c_str());
    return EXIT_CODE_1;
  }
  LOG_INFO("Successfully created session! model=%s", model_name.c_str());

  std::cout << "\n\n";
  std::cout << std::endl << std::endl;
  std::cout << std::string(120, '=') << std::endl;
  std::cout << "3. Run the Session object with input tensors" << std::endl;
  std::cout << std::string(120, '=') << std::endl;

  // Build input tensors
  InputTensors input_tensors;
  std::vector<std::string> fetches;
  std::vector<std::string> targets;

  std::string name = "X:0";
  int batch_size = 3;
  int element_size = 5;
  LOG_INFO("Input %s with shape{%d, %d}", name.c_str(), batch_size, element_size);
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0.0, 1.0);
  tensorflow::Tensor input(tensorflow::DT_FLOAT, tensorflow::TensorShape({batch_size, element_size}));
  float* p = input.flat<float>().data();
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < element_size; j++) {
      *p = dis(gen);
      p++;
    }
  }

  float* p2 = input.flat<float>().data();
  for (int i = 0; i < batch_size; i++) {
    std::cout << ">> ";
    for (int j = 0; j < element_size; j++) {
      std::cout << *p2++ << " ";
    }
    std::cout << std::endl;
  }
  input_tensors.emplace_back(name, input);

  LOG_INFO("\nFetch name: Sigmoid:0\n");
  fetches.push_back("Sigmoid:0");

  // Session Run
  OutputTensors output_tensors;
  LOG_INFO(">> Start Session Run");
  status = session->Run(input_tensors, fetches, targets, &output_tensors);
  if (!status.ok()) {
    LOG_ERROR("Session Run failed! model=%s, ret=%d, err=%s", model_name.c_str(), status.code(), status.error_message().c_str());
    return status.code();
  }
  LOG_INFO("Successfully ran session! model=%s", model_name.c_str());

  // Print out output tensors
  std::cout << "\n\n";
  for (size_t i = 0; i < fetches.size(); i++) {
    auto& tensor = output_tensors[i];
    std::cout << "Output " << fetches[i] << ":" << std::endl;
    std::cout << tensor.DebugString() << std::endl;
    float* p = tensor.flat<float>().data();
    for (size_t i = 0; i < tensor.dim_size(0); i++) {
      std::cout << ">> ";
      for (size_t j = 0; j < tensor.dim_size(1); j++) {
        std::cout << *p++ << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }

  std::cout << "FINISHED" << std::endl;
}
