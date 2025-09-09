#include <torch/extension.h>
#include <vector>

void axpby_launcher(torch::Tensor x, torch::Tensor y, float a, float b);

torch::Tensor axpby(torch::Tensor x, double a, double b) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.dtype() == torch::kFloat32 || x.dtype() == torch::kFloat64,
              "x must be float32/float64");

  // Allocate output on same device/dtype/shape:
  auto y = torch::empty_like(x);

  // Ensure kernels run on the right device/stream:
//   auto device_guard(x.get_device());

  axpby_launcher(x, y, static_cast<float>(a), static_cast<float>(b));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("axpby", &axpby, "y = a*x + b (CUDA)");
}
