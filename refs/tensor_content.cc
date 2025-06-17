InferenceInput input;
if (input.ParseFromString(req->input)) {
  for (size_t i = 0; i < input.feed_size(); i++) {
    auto& named_tensor = input.feed(i);
    const tensorflow::TensorProto& tensor = named_tensor.tensor();
    auto& name = named_tensor.name();
    if (name != "group_candidates_size:0") continue;
    LOG_INFO("feed name: %s", name.c_str());
    std::cout << "  dtype: " << tensorflow::DataType_Name(tensor.dtype()) << std::endl;
    if (tensor.dtype() == tensorflow::DT_INT64 && tensor.tensor_content().empty()) {
      std::cout << "  value: " << tensor.int64_val().at(0) << std::endl;
      std::vector<int64_t> array(1, tensor.int64_val().at(0));
      auto mtensor = input.mutable_feed(i)->mutable_tensor();
      auto shape = mtensor->mutable_tensor_shape();
      if (shape->dim_size() == 0) {
        shape->add_dim()->set_size(1);
      }
      mtensor->set_tensor_content(
        std::string(reinterpret_cast<const char*>(array.data()), 1 * sizeof(int64_t))
      );

      req->input = input.SerializeAsString();
      break;
    }
  }
} else {
  LOG_ERROR("Failed to parse input data from req->input");
}