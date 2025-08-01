// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/gemms/gemv.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace mlx::core::cu {

namespace cg = cooperative_groups;

static constexpr int rows_per_block = 8;

template <typename T, int rows_per_block, int n_per_thread>
__device__ void
gemv_impl(const T* mat, const T* vec, T* out, int rows, int cols) {
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  auto g_idx = block.group_index();
  auto t_idx = block.thread_index();
  int row = g_idx.x * rows_per_block + t_idx.y;

  if (row < rows) {
    float sum = 0.0f;
    for (int col = n_per_thread * warp.thread_rank(); col < cols;
         col += (WARP_SIZE * n_per_thread)) {
      auto local_mat =
          unsafe_load_vector<n_per_thread>(mat + row * cols + col, 0);
      auto local_vec = unsafe_load_vector<n_per_thread>(vec + col, 0);
#pragma unroll
      for (int j = 0; j < n_per_thread; ++j) {
        sum +=
            static_cast<float>(local_mat[j]) * static_cast<float>(local_vec[j]);
      }
    }

    sum = cg::reduce(warp, sum, cg::plus<float>{});
    if (warp.thread_rank() == 0) {
      out[row] = static_cast<T>(sum);
    }
  }
}

template <typename T, int rows_per_block, int n_per_thread>
__global__ void
gemv_single(const T* mat, const T* vec, T* out, int rows, int cols) {
  gemv_impl<T, rows_per_block, n_per_thread>(mat, vec, out, rows, cols);
}

template <typename T, int rows_per_block, int n_per_thread>
__global__ void gemv_batched(
    const T* mat,
    const T* vec,
    T* out,
    int rows,
    int cols,
    const __grid_constant__ Shape batch_shape,
    const __grid_constant__ Strides mat_batch_strides,
    const __grid_constant__ Strides vec_batch_strides,
    int batch_ndim) {
  auto block = cg::this_thread_block();
  auto batch_idx = block.group_index().y;
  auto [vec_offset, mat_offset] = elem_to_loc(
      batch_idx,
      batch_shape.data(),
      vec_batch_strides.data(),
      mat_batch_strides.data(),
      batch_ndim);
  gemv_impl<T, rows_per_block, n_per_thread>(
      mat + mat_offset, vec + vec_offset, out + batch_idx * rows, rows, cols);
}

template <typename T, int rows_per_thread, int cols_per_step>
__device__ void
gemv_t_impl(const T* mat, const T* vec, T* out, int rows, int cols) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  auto g_idx = block.group_index();
  auto t_idx = block.thread_index();
  auto block_dims = block.dim_threads();

  int row = g_idx.x * block_dims.x + t_idx.x;
  int col_offset = (g_idx.y * block_dims.y + t_idx.y);
  int col_step = WARP_SIZE;

  mat += rows * col_offset * cols_per_step;

  float sum[rows_per_thread] = {};

  for (int col = col_offset; (cols_per_step * col) < cols; col += col_step) {
    auto local_vec = unsafe_load_vector<cols_per_step>(vec, col);
#pragma unroll
    for (int i = 0; i < cols_per_step; ++i) {
      auto local_mat = unsafe_load_vector<rows_per_thread>(mat + i * rows, row);
#pragma unroll
      for (int k = 0; k < rows_per_thread; ++k) {
        sum[k] +=
            static_cast<float>(local_mat[k]) * static_cast<float>(local_vec[i]);
      }
    }
    mat += cols_per_step * col_step * rows;
  }

  __shared__ float smem_sums[rows_per_thread * WARP_SIZE][WARP_SIZE + 1];

#pragma unroll
  for (int k = 0; k < rows_per_thread; ++k) {
    smem_sums[rows_per_thread * t_idx.x + k][t_idx.y] = sum[k];
  }
  block.sync();

  AlignedVector<T, rows_per_thread> out_vec;
#pragma unroll
  for (int k = 0; k < rows_per_thread; ++k) {
    out_vec[k] = static_cast<T>(cg::reduce(
        warp,
        smem_sums[rows_per_thread * t_idx.y + k][t_idx.x],
        cg::plus<float>{}));
  }

  if (t_idx.x == 0) {
    unsafe_store_vector(out, g_idx.x * block_dims.x + t_idx.y, out_vec);
  }
}

template <typename T, int rows_per_thread, int cols_per_step>
__global__ void
gemv_t_single(const T* mat, const T* vec, T* out, int rows, int cols) {
  gemv_t_impl<T, rows_per_thread, cols_per_step>(mat, vec, out, rows, cols);
}

template <typename T, int rows_per_thread, int cols_per_step>
__global__ void gemv_t_batched(
    const T* mat,
    const T* vec,
    T* out,
    int rows,
    int cols,
    const __grid_constant__ Shape batch_shape,
    const __grid_constant__ Strides mat_batch_strides,
    const __grid_constant__ Strides vec_batch_strides,
    int batch_ndim) {
  auto block = cg::this_thread_block();
  auto batch_idx = block.group_index().z;
  auto [vec_offset, mat_offset] = elem_to_loc(
      batch_idx,
      batch_shape.data(),
      vec_batch_strides.data(),
      mat_batch_strides.data(),
      batch_ndim);
  gemv_t_impl<T, rows_per_thread, cols_per_step>(
      mat + mat_offset, vec + vec_offset, out + batch_idx * rows, rows, cols);
}

bool can_use_gemv(int M, int N, int K, bool a_transposed, bool b_transposed) {
  if (M != 1 && N != 1) {
    return false;
  }
  if (M == 1 && !b_transposed) {
    return N % 32 == 0;
  } else if (N == 1 && a_transposed) {
    return M % 32 == 0;
  } else {
    return K % 32 == 0;
  }
}

template <typename F>
void dispatch_n_per_thread(int n_per_thread, F&& f) {
  switch (n_per_thread) {
    case 1:
      f(std::integral_constant<int, 1>{});
      break;
    case 2:
      f(std::integral_constant<int, 2>{});
      break;
    case 4:
      f(std::integral_constant<int, 4>{});
      break;
  }
}

template <typename T, typename F>
void dispatch_rows_per_thread(int rows_per_thread, F&& f) {
  if constexpr (sizeof(T) < 4) {
    switch (rows_per_thread) {
      case 1:
        f(std::integral_constant<int, 1>{});
        break;
      case 2:
        f(std::integral_constant<int, 2>{});
        break;
    }
  } else {
    f(std::integral_constant<int, 1>{});
  }
}

template <typename T, typename F>
void dispatch_cols_per_step(int cols_per_step, F&& f) {
  switch (cols_per_step) {
    case 1:
      f(std::integral_constant<int, 1>{});
      break;
    case 2:
      f(std::integral_constant<int, 2>{});
      break;
  }
}

void gemv(
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    uint32_t batch_count,
    const mlx::core::Shape& batch_shape,
    const mlx::core::Strides& a_batch_strides,
    const mlx::core::Strides& b_batch_strides,
    CommandEncoder& encoder) {
  encoder.set_input_array(a);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  dispatch_float_types(out.dtype(), "gemv", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    const DataType* mat;
    const DataType* vec;
    int rows;
    int cols = K;
    auto mat_strides = const_param(a_batch_strides);
    auto vec_strides = const_param(b_batch_strides);

    bool transpose;
    if (M == 1) {
      mat = b.data<DataType>();
      vec = a.data<DataType>();
      transpose = b.strides()[b.ndim() - 2] != 1;
      rows = N;
      std::swap(mat_strides, vec_strides);
    } else {
      mat = a.data<DataType>();
      vec = b.data<DataType>();
      transpose = a.strides().back() != 1;
      rows = M;
    }
    if (transpose) {
      dim3 block_dim{32, 32};
      int rows_per_thread;
      if (rows % 64 == 0 && out.dtype().size() < 4 && is_aligned<2>(mat)) {
        rows_per_thread = 2;
      } else {
        rows_per_thread = 1;
      }
      int cols_per_step;
      if (cols % 2 == 0 && is_aligned<2>(vec)) {
        cols_per_step = 2;
      } else {
        cols_per_step = 1;
      }

      uint32_t num_blocks_x = cuda::ceil_div(rows, rows_per_thread * WARP_SIZE);

      dispatch_rows_per_thread<DataType>(rows_per_thread, [&](auto rpt) {
        dispatch_cols_per_step<DataType>(cols_per_step, [&](auto cps) {
          if (batch_count == 1) {
            auto kernel = gemv_t_single<DataType, rpt, cps>;
            encoder.add_kernel_node(
                kernel,
                num_blocks_x,
                block_dim,
                mat,
                vec,
                out.data<DataType>(),
                rows,
                cols);
          } else {
            auto kernel = gemv_t_batched<DataType, rpt, cps>;
            encoder.add_kernel_node(
                kernel,
                dim3{num_blocks_x, 1, batch_count},
                block_dim,
                mat,
                vec,
                out.data<DataType>(),
                rows,
                cols,
                const_param(batch_shape),
                mat_strides,
                vec_strides,
                batch_shape.size());
          }
        });
      });

    } else {
      dim3 block_dims{WARP_SIZE, rows_per_block};
      uint32_t num_blocks_x = (rows + rows_per_block - 1) / rows_per_block;
      int n_per_t;
      if (K % 128 == 0 && is_aligned<4>(mat) && is_aligned<4>(vec)) {
        n_per_t = 4;
      } else if (K % 64 == 0 && is_aligned<2>(mat) && is_aligned<2>(vec)) {
        n_per_t = 2;
      } else {
        n_per_t = 1;
      }
      dispatch_n_per_thread(n_per_t, [&](auto n_per_thread) {
        if (batch_count == 1) {
          auto kernel = gemv_single<DataType, rows_per_block, n_per_thread()>;
          encoder.add_kernel_node(
              kernel,
              num_blocks_x,
              block_dims,
              mat,
              vec,
              out.data<DataType>(),
              rows,
              cols);
        } else {
          auto kernel = gemv_batched<DataType, rows_per_block, n_per_thread()>;
          encoder.add_kernel_node(
              kernel,
              dim3{num_blocks_x, batch_count},
              block_dims,
              mat,
              vec,
              out.data<DataType>(),
              rows,
              cols,
              const_param(batch_shape),
              mat_strides,
              vec_strides,
              batch_shape.size());
        }
      });
    }
  });
}

} // namespace mlx::core::cu
