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

  axpby_launcher(x, y, static_cast<float>(a), static_cast<float>(b));
  return y;
}

void one_dim_conv_launcher(torch::Tensor kernel, torch::Tensor in, torch::Tensor out);

torch::Tensor one_dim_conv(torch::Tensor kernel, torch::Tensor in) {

  TORCH_CHECK(in.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(in.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(in.dtype() == torch::kFloat32 || in.dtype() == torch::kFloat64,
              "in must be float32/float64");

  // Allocate output on same device/dtype/shape:
  auto out = torch::empty_like(in);

  one_dim_conv_launcher(kernel, in, out);
  return out;
}

// void two_dim_transpose_launcher(torch::Tensor in, torch::Tensor out);

// torch::Tensor two_dim_transpose(torch::Tensor in) {

//   TORCH_CHECK(in.is_cuda(), "in must be a CUDA tensor");
//   TORCH_CHECK(in.is_contiguous(), "in must be contiguous");
//   TORCH_CHECK(in.dtype() == torch::kFloat32 || in.dtype() == torch::kFloat64,
//               "in must be float32/float64");

//   // Allocate output on same device/dtype/shape:
//   auto out = torch::empty({in.size(1), in.size(0)}, in.options());

//   two_dim_transpose_launcher(in, out);
//   return out;
// }

void two_dim_prefix_sum_launcher(torch::Tensor in, torch::Tensor out);

torch::Tensor two_dim_prefix_sum(torch::Tensor in) {

  TORCH_CHECK(in.is_cuda(), "in must be a CUDA tensor");
  TORCH_CHECK(in.is_contiguous(), "in must be contiguous");
  TORCH_CHECK(in.dtype() == torch::kFloat32 || in.dtype() == torch::kFloat64,
              "in must be float32/float64");

  // Allocate output on same device/dtype/shape:
  auto out = torch::empty_like(in);

  two_dim_prefix_sum_launcher(in, out);
  return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("axpby", &axpby, "y = a*x + b (CUDA)");
  m.def("one_dim_conv", &one_dim_conv, "one dim convolution (CUDA)");
  // m.def("two_dim_transpose", &two_dim_transpose, "two dim blocked transpose (CUDA)");
  m.def("two_dim_prefix_sum", &two_dim_prefix_sum, "two dim prefix sum (CUDA)");
}
