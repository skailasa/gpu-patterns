// cpu_kernels.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <iostream>
#include "omp.h"

namespace py = pybind11;

template <typename T>
void two_dim_transpose_kernel(const T* in, T* out, py::ssize_t H, py::ssize_t W) {

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < H; ++i) {
        // iterate over rows (row major)
        for (size_t j=0; j < W; ++j) {
            // iterate over columns
            out[j*H + i] = in[i*W + j];
        }
    }
}


/// restrict stops aliasing of in/out telling compiler that they refer to different memory locations
// reduce false sharing (when two threads access data in the same cache line)
// no cache lines are shared (or fewer are) when threads access distinct tiles
template <typename T>
void two_dim_transpose_blocked_kernel(const T* __restrict in, T* __restrict out, size_t H, size_t W,  size_t TS) {


    // Parallelise across tiles
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t ii = 0; ii < H; ii += TS) {
        for (size_t jj = 0; jj < W; jj += TS) {

            const size_t i_max = std::min(ii + TS, H);
            const size_t j_max = std::min(jj + TS, W);

            // make inner loop contiguous for writes
            for (size_t j = jj; j < j_max; ++j) {

                // compiler hint to use SIMD extensions
                #pragma omp simd
                for (size_t i = ii; i < i_max; ++i) {
                    out[j * H + i] = in[i * W + j];
                }
            }
        }
    }
}

// ---- kernel (contiguous C-order) ----
template <typename T>
void two_dim_prefix_sum_kernel(const T* __restrict in, T*  __restrict out, py::ssize_t H, py::ssize_t W) {

    // parallelise across rows, heap allocated
    // can make stack allocated with std::vector<T, size> can't do push back
    std::vector<T> tmp(H*W);
    std::vector<T> tmp2(H*W);

    // int arr[10]; // stack allocated
    // int* arr = new int[10]; // heap allocated; delete with delete[] arr;

    // compute first cumulative sum
    #pragma omp parallel for schedule(static)
    for (py::ssize_t i = 0; i < H; ++i) {

        size_t row_offset = (size_t)(i*W);
        // compute prefix sum on each row

        T curr = 0.;
        for (size_t j = 0; j < W; ++j) {
            curr += in[row_offset + j];
            tmp[row_offset + j] = curr;
        }
    }

    // Compute transpose
    two_dim_transpose_blocked_kernel(tmp.data(), tmp2.data(), H, W, 64);

    // second cumulative sum
    #pragma omp parallel for schedule(static)
    for (py::ssize_t i = 0; i < W; ++i) {

        size_t row_offset = (size_t)(i*H);
        // compute prefix sum on each row

        T curr = 0.;
        for (size_t j = 0; j < H; ++j) {
            curr += tmp2[row_offset + j];
            tmp[row_offset + j] = curr;
        }
    }

    // Compute transpose
    two_dim_transpose_blocked_kernel(tmp.data(), out, W, H, 64);

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

template <typename T>
py::array_t<T> two_dim_transpose_t(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
    auto in_buf = arr.request();

    if (in_buf.ndim != 2) throw std::runtime_error("Expected a 2D array");
    const py::ssize_t H = in_buf.shape[0];
    const py::ssize_t W = in_buf.shape[1];

    py::array_t<T> out({W, H});
    auto out_buf = out.request();

    const T* in_ptr = static_cast<const T*>(in_buf.ptr);
    T* out_ptr = static_cast<T*>(out_buf.ptr);

    {
        // release the GIL during the heavy loop
        py::gil_scoped_release release;
        two_dim_transpose_kernel(in_ptr, out_ptr, H, W);
    }

    return out;
}

template <typename T>
py::array_t<T> two_dim_transpose_blocked_t(py::array_t<T, py::array::c_style | py::array::forcecast> arr,  py::ssize_t blocksize) {
    auto in_buf = arr.request();

    if (in_buf.ndim != 2) throw std::runtime_error("Expected a 2D array");
    const py::ssize_t H = in_buf.shape[0];
    const py::ssize_t W = in_buf.shape[1];

    py::array_t<T> out({W, H});
    auto out_buf = out.request();

    const T* in_ptr = static_cast<const T*>(in_buf.ptr);
    T* out_ptr = static_cast<T*>(out_buf.ptr);

    {
        // release the GIL during the heavy loop
        py::gil_scoped_release release;
        two_dim_transpose_blocked_kernel(in_ptr, out_ptr, (size_t)H, (size_t)W, size_t(blocksize));
    }

    return out;
}

// ---- dtype-dispatching entry point ----
py::array two_dim_prefix_sum(py::array arr) {
    if (py::isinstance<py::array_t<float>>(arr)) {
        return two_dim_prefix_sum_t<float>(arr);
    } else if (py::isinstance<py::array_t<double>>(arr)) {
        return two_dim_prefix_sum_t<double>(arr);
    } else {
        throw std::runtime_error("Unsupported dtype; use float32/float64.");
    }
}

py::array two_dim_transpose(py::array arr) {
    if (py::isinstance<py::array_t<float>>(arr)) {
        return two_dim_transpose_t<float>(arr);
    } else if (py::isinstance<py::array_t<double>>(arr)) {
        return two_dim_transpose_t<double>(arr);
    } else {
        throw std::runtime_error("Unsupported dtype; use float32/float64.");
    }
}

py::array two_dim_transpose_blocked(py::array arr, py::ssize_t blocksize) {
    if (py::isinstance<py::array_t<float>>(arr)) {
        return two_dim_transpose_blocked_t<float>(arr, blocksize);
    } else if (py::isinstance<py::array_t<double>>(arr)) {
        return two_dim_transpose_blocked_t<double>(arr, blocksize);
    } else {
        throw std::runtime_error("Unsupported dtype; use float32/float64.");
    }
}


PYBIND11_MODULE(cpu_kernels, m) {  // <- module name and variable
    m.doc() = "CPU kernels";
    m.def("two_dim_prefix_sum", &two_dim_prefix_sum,
          "Inclusive 2D prefix sum (integral image) over a 2D NumPy array");
    m.def("two_dim_transpose", &two_dim_transpose,
          "Two dimensional transpose");
    m.def("two_dim_transpose_blocked", &two_dim_transpose_blocked,
          "Two dimensional transpose");
}
