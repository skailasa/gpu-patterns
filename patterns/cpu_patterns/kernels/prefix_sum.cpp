// cpu_kernels.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <stdexcept>
#include "omp.h"

namespace py = pybind11;

// ---- kernel (contiguous C-order) ----
template <typename T>
void two_dim_prefix_sum_kernel(const T* in, T* out, py::ssize_t H, py::ssize_t W) {
    // Inclusive prefix sum: out[i,j] = sum_{u<=i, v<=j} in[u,v]
    for (py::ssize_t i = 0; i < H; ++i) {
        T row_acc = T(0);
        const py::ssize_t base = i * W;
        const py::ssize_t prev = (i - 1) * W;
        for (py::ssize_t j = 0; j < W; ++j) {
            row_acc += in[base + j];
            if (i == 0) {
                out[base + j] = row_acc;
            } else {
                out[base + j] = row_acc + out[prev + j];
            }
        }
    }
}

// ---- typed wrapper returning numpy array ----
template <typename T>
py::array_t<T> two_dim_prefix_sum_t(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
    auto in_buf = arr.request();
    if (in_buf.ndim != 2) throw std::runtime_error("Expected a 2D array");
    const py::ssize_t H = in_buf.shape[0];
    const py::ssize_t W = in_buf.shape[1];

    // allocate output with same shape
    py::array_t<T> out({H, W});
    auto out_buf = out.request();

    const T* in_ptr = static_cast<const T*>(in_buf.ptr);
    T* out_ptr = static_cast<T*>(out_buf.ptr);

    {
        // release the GIL during the heavy loop
        py::gil_scoped_release release;
        two_dim_prefix_sum_kernel(in_ptr, out_ptr, H, W);
    }

    return out;
}

// ---- dtype-dispatching entry point ----
py::array two_dim_prefix_sum(py::array arr) {
    if (py::isinstance<py::array_t<float>>(arr)) {
        return two_dim_prefix_sum_t<float>(arr);
    } else if (py::isinstance<py::array_t<double>>(arr)) {
        return two_dim_prefix_sum_t<double>(arr);
    } else if (py::isinstance<py::array_t<std::int32_t>>(arr)) {
        return two_dim_prefix_sum_t<std::int32_t>(arr);
    } else if (py::isinstance<py::array_t<std::int64_t>>(arr)) {
        return two_dim_prefix_sum_t<std::int64_t>(arr);
    } else {
        throw std::runtime_error("Unsupported dtype; use float32/float64/int32/int64.");
    }
}

PYBIND11_MODULE(cpu_kernels, m) {  // <- module name and variable
    m.doc() = "CPU kernels";
    m.def("two_dim_prefix_sum", &two_dim_prefix_sum,
          "Inclusive 2D prefix sum (integral image) over a 2D NumPy array");
}
