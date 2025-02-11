# REGISTER_KERNEL_BUILDER
```C++
// Register the GPU kernels.
REGISTER_KERNEL_BUILDER(
  Name("Stupid").Device(DEVICE_GPU)
    .TypeConstraint<float>("T"),
  StupidOp<GPUDevice, float>
);
```


# REGISTER_OP
```C++
REGISTER_OP("Stupid")
    .Attr("T: {float}")
    .Input("in: T")
    .Output("output: T")
    .SetShapeFn(
        [](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        }
    );

// REGISTER_OP("Stupid") expands to:
static ::tensorflow::InitOnStartupMarker const register_op24 __attribute__((unused)) = (::std::integral_constant<bool, !(false || true)>::value) ? ::tensorflow::InitOnStartupMarker{} : ::tensorflow::InitOnStartupMarker {} << ::tensorflow::register_op::OpDefBuilderWrapper("Stupid")
    .Attr("T: {float}")
    .Input("in: T")
    .Output("output: T")
    .SetShapeFn(
        [](::tensorflow::shape_inference::InferenceContext* c) {
            c->set_output(0, c->input(0));
            return Status::OK();
        }
    );
```
