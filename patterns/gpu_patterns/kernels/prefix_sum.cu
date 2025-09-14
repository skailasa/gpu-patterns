#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


int ceil_div(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}


// blocked transpose algorithm, bring in tiles from global to shared memory
// transpose in shared memory
template <typename T, int TS>
__global__ void two_dim_transpose_kernel(T* __restrict__ in,  T* __restrict__ out, int H, int W) {

    __shared__ T tile[TS][TS+1]; // avoid bank conflicts with padding

    // global element index mapping to threads
    const int in_col = blockIdx.x*TS + threadIdx.x; // fastest changing
    const int in_row = blockIdx.y*TS + threadIdx.y;

    // read tile into shared memory
    if (in_row < H && in_col < W) {
        tile[threadIdx.y][threadIdx.x] = in[in_row*W + in_col];
    }

    __syncthreads();

    // store transposed tile in out
    int out_row = in_col;
    int out_col = in_row;

    // perform transpose in shared memory
    if (out_row < W && out_col < H) {
        out[out_row * H + out_col] = tile[threadIdx.y][threadIdx.x];
    }
}


template <typename T, int TS>
__device__ void two_dim_transpose_device(T* __restrict__ in, T* __restrict__ out, int H, int W) {
    __shared__ T tile[TS][TS+1]; // avoid bank conflicts with padding

    // global element index mapping to threads
    const int in_col = blockIdx.x*TS + threadIdx.x; // fastest changing
    const int in_row = blockIdx.y*TS + threadIdx.y;

    // read tile into shared memory
    if (in_row < H && in_col < W) {
        tile[threadIdx.y][threadIdx.x] = in[in_row*W + in_col];
    }

    __syncthreads();

    // store transposed tile in out
    int out_row = in_col;
    int out_col = in_row;

    // perform transpose in shared memory
    if (out_row < W && out_col < H) {
        out[out_row * H + out_col] = tile[threadIdx.y][threadIdx.x];
    }
}


template <typename T, int TS>
__global__ void two_dim_prefix_sum_1(const T* __restrict__ in,  T* __restrict__ out, T* __restrict__ row_sums, int H, int W) {

    // Load tile of global problem into shared memory
    __shared__ T in_tile[TS][TS];

    const int in_col = blockIdx.x*blockDim.x + threadIdx.x;
    const int in_row = blockIdx.y*blockDim.y + threadIdx.y;

    if (in_row < H && in_col < W) {
        in_tile[threadIdx.y][threadIdx.x] = in[in_row*W+in_col];
    }

    __syncthreads();

    // Perform cumulative sum on shared memory tiles (each thread is responsible for one input entry)
    int lane = threadIdx.x; // each row is one warp of tile

    // mask only lanes that are valid at the right edge
    unsigned mask = __ballot_sync(0xffffffffu, in_row < H && in_col < W);

    T x = in_tile[threadIdx.y][lane];

    #pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        T n = __shfl_up_sync(mask, x, offset);
        if (lane >= offset) x+=n;
    }

    // write partial warp-wise result to global memory
    if (in_row < H && in_col < W) {
        out[in_row * W + in_col] = x;
    }

    /// if lane at the end, then write to row sums
    if (lane == TS-1 && in_row < H) {
        const int n_col_blocks = (W + TS - 1)/TS;
        const int col = blockIdx.x;
        const int row = blockIdx.y*TS + threadIdx.y;
        row_sums[row * n_col_blocks + col] = x;
    }
}

template <typename T, int TS>
__global__ void two_dim_prefix_sum_2(T* __restrict__ in_row_sums, T* __restrict__ out_row_sums, int H, int W) {

    int n_col_blocks = (W + TS - 1)/TS;

    int row = blockIdx.x; // one block (warp) per row
    int lane = threadIdx.x; // 0...31

    if (row >= H) return;

    T carry = 0; // running total

    for (int block_idx = 0; block_idx < n_col_blocks; block_idx += 1) {
        int col = block_idx*TS + lane;

        // load this tile's row sum
        T x = (col < n_col_blocks) ? in_row_sums[row * n_col_blocks + col] : T(0);

        // Compute prefix sum of first block (warp inclusive scan)
        unsigned mask = __ballot_sync(0xffffffffu, col < n_col_blocks);
        for (int offset = 1; offset < 32; offset <<= 1) {
            T n = __shfl_up_sync(mask, x, offset);
            if (lane >= offset) x += n;
        }

        // add carry from previous chunks
        x += carry;

        // write back
        if (col < n_col_blocks) {
            out_row_sums[row * n_col_blocks + col] = x;
        }

        // update carry
        int last = 31 - __clz(mask);                 // index of top set bit
        T chunk_total = __shfl_sync(mask, x, last);
        carry = chunk_total;
    }
}



template <typename T, int TS>
__global__ void two_dim_prefix_sum_3(T* __restrict__ block_offsets, T* __restrict__ in, int H, int W) {

    int col = blockIdx.x * blockDim.x + threadIdx.x; // element col
    int row = blockIdx.y * blockDim.y + threadIdx.y; // element row
    if (row >= H || col >= W) return;

    int n_block_cols = (W + TS - 1)/TS;
    int block_col = blockIdx.x;

    T add = T(0);
    if (block_col > 0) {
        add = block_offsets[row * n_block_cols + (block_col - 1)];
    }

    in[row * W + col] += add;
}


template <typename T>
void two_dim_prefix_sum_kernel(const T* __restrict__ in, T* __restrict__ out, int H, int W) {

    // 1. Compute row sums
    dim3 block(32, 32); // max block size
    dim3 grid(ceil_div(W, block.x), ceil_div(H, block.y));

    int n_col_blocks = (W + 32 - 1) / 32;
    std::vector<T> row_sums(H * n_col_blocks);
    T *in_row_sums_dev;
    cudaMalloc((void **)&in_row_sums_dev, row_sums.size() * sizeof(T));
    cudaMemset(in_row_sums_dev, 0, row_sums.size() * sizeof(T));

    two_dim_prefix_sum_1<T, 32><<<grid, block>>>(in, out, in_row_sums_dev, H, W);

    // 2. Compute offsets from row sums
    dim3 block1(32); // allocate one warp to handle each row
    dim3 grid1(H); // enough blocks to cover all rows

    std::vector<T> out_row_sums(H * n_col_blocks);
    T *out_row_sums_dev;
    cudaMalloc((void **)&out_row_sums_dev, out_row_sums.size() * sizeof(T));
    cudaMemset(out_row_sums_dev, 0, out_row_sums.size() * sizeof(T));
    two_dim_prefix_sum_2<T, 32><<<grid1, block1>>>(in_row_sums_dev, out_row_sums_dev, H, W);

    /// 3. Apply offsets
    dim3 block2(32, 32);
    dim3 grid2(ceil_div(W, block2.x), ceil_div(H, block2.y));
    two_dim_prefix_sum_3<T, 32><<<grid2, block2>>>(out_row_sums_dev, out, H, W);

}

void two_dim_prefix_sum_launcher(torch::Tensor in, torch::Tensor out) {

    int64_t H = in.size(0);
    int64_t W = in.size(1);

    torch::Tensor tmp = torch::empty_like(in);
    int64_t H_tmp = tmp.size(0);
    int64_t W_tmp = tmp.size(1);

    torch::Tensor tmp2 = torch::empty({tmp.size(1), tmp.size(0)});
    int64_t H_tmp2 = tmp2.size(0);
    int64_t W_tmp2 = tmp2.size(1);

    torch::Tensor tmp3 = torch::empty({tmp2.size(1), tmp2.size(0)});
    int64_t H_tmp3 = tmp3.size(0);
    int64_t W_tmp3 = tmp3.size(1);

    AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "two_dim_prefix_sum_kernel", [&] {
        two_dim_prefix_sum_kernel<scalar_t>(in.data_ptr<scalar_t>(), tmp.data_ptr<scalar_t>(), (int)H, (int)W);

        dim3 block(32, 32); // max block size
        dim3 grid(ceil_div(W, block.x), ceil_div(H, block.y));
        two_dim_transpose_kernel<scalar_t, 32><<<grid, block>>>(tmp.data_ptr<scalar_t>(), tmp2.data_ptr<scalar_t>(), (int)H_tmp, (int)W_tmp);

        two_dim_prefix_sum_kernel<scalar_t>(tmp2.data_ptr<scalar_t>(), tmp3.data_ptr<scalar_t>(), (int)H_tmp2, (int)W_tmp2);

        dim3 block1(32, 32); // max block size
        dim3 grid1(ceil_div(W_tmp3, block1.x), ceil_div(H_tmp3, block1.y));
        two_dim_transpose_kernel<scalar_t, 32><<<grid1, block1>>>(tmp3.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), (int)H_tmp3, (int)W_tmp3);
    });


  cudaDeviceSynchronize(); TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");

}


// void two_dim_prefix_sum_launcher(torch::Tensor in, torch::Tensor out) {

//     int64_t H = in.size(0);
//     int64_t W = in.size(1);

//     const int64_t n = in.numel();
//     dim3 block(32, 32);
//     dim3 grid(ceil_div((int)H, block.x), ceil_div((int)W, block.y));

//     AT_DISPATCH_FLOATING_TYPES(in.scalar_type(), "two_dim_prefix_sum_kernel", [&] {
//         two_dim_prefix_sum_kernel<scalar_t><<<grid, block>>>(in.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(), (int)H, (int)W);
//     });

//   cudaDeviceSynchronize(); TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel failed");

// }
