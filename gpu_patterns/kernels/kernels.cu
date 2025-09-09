#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
__global__ void one_dim_conv_kernel(const T* __restrict__ kernel, const T* __restrict__ in, const T* __restrict__ out) {

}


template <typename T>
__global__ void axpby_kernel(const T* __restrict__ x,
                             T* __restrict__ y,
                             T a, T b,
                             int64_t n) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;
  for (; i < n; i += stride) {
    y[i] = a * x[i] + b;
  }
}

void one_dim_conv_launcher(torch::Tensor kernel, torch::Tensor in, torch::Tensor out) {

    const int64_t n = in.numel();
    const int threads = 128;
    const int blocks = (int)((n + threads-1) / threads); // ceil div

    AT_DISPATCH_FLOATING_TYPES(kernel.scalar_type(), "one_dim_conv_kernel", [&] {
        one_dim_conv_kernel<scalar_t><<<blocks, threads>>>(kernel.data_ptr<scalar_t>(), in.data_ptr<scalar_t>(), out.data_ptr<scalar_t>());
    });

  cudaDeviceSynchronize(); TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");

}

void axpby_launcher(torch::Tensor x, torch::Tensor y, float a, float b) {
  const int64_t n = x.numel();
  const int threads = 256;
  const int blocks = (int)((n + threads - 1) / threads);

  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "axpby_kernel", [&] {
    axpby_kernel<scalar_t><<<blocks, threads>>>(
        x.data_ptr<scalar_t>(),
        y.data_ptr<scalar_t>(),
        static_cast<scalar_t>(a),
        static_cast<scalar_t>(b),
        n);
  });

  cudaDeviceSynchronize(); TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");
}