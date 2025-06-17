const tensorflow::TensorProto& tensor = named_tensor.tensor();
std::cout << "dtype: " << tensorflow::DataType_Name(tensor.dtype()) << std::endl;
