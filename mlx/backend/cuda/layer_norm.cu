// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/reduce/reduce.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

inline __device__ float3 plus_f3(const float3& a, const float3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

// Similar to cub::BlockReduce, but result is broadcasted to every thread.
template <typename T, int BLOCK_DIM>
struct BlockBroadcastReduce {
  static_assert(WARP_SIZE <= BLOCK_DIM && BLOCK_DIM <= WARP_SIZE * WARP_SIZE);
  static_assert(BLOCK_DIM % WARP_SIZE == 0);
  using TempStorage = T[BLOCK_DIM / WARP_SIZE];

  cg::thread_block& block;
  TempStorage& temp;

  template <typename Op>
  __device__ T Reduce(const T& input, const Op& op, const T& init_value) {
    auto warp = cg::tiled_partition<WARP_SIZE>(block);
    T x = cg::reduce(warp, input, op);
    if (warp.thread_rank() == 0) {
      temp[warp.meta_group_rank()] = x;
    }
    block.sync();
    x = warp.thread_rank() < warp.meta_group_size() ? temp[warp.thread_rank()]
                                                    : init_value;
    return cg::reduce(warp, x, op);
  }

  __device__ T Sum(const T& input) {
    return Reduce(input, cg::plus<T>{}, T{});
  }
};

template <typename T, int BLOCK_DIM, int N_READS = 4>
__global__ void layer_norm(
    const T* x,
    const T* w,
    const T* b,
    T* out,
    float eps,
    int32_t axis_size,
    int64_t w_stride,
    int64_t b_stride) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  using BlockReduceT = BlockBroadcastReduce<float, BLOCK_DIM>;
  __shared__ typename BlockReduceT::TempStorage temp;

  x += grid.block_rank() * axis_size;
  out += grid.block_rank() * axis_size;

  // Sum.
  float sum = 0;
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    auto xn = load_vector<N_READS>(x, index, axis_size, T(0));
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      sum += static_cast<float>(xn[i]);
    }
  }
  sum = BlockReduceT{block, temp}.Sum(sum);

  // Mean.
  float mean = sum / axis_size;

  // Normalizer.
  float normalizer = 0;
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    if ((index + 1) * N_READS <= axis_size) {
      auto xn = load_vector<N_READS>(x, index);
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        float t = static_cast<float>(xn[i]) - mean;
        normalizer += t * t;
      }
    } else {
      for (int i = index * N_READS; i < axis_size; ++i) {
        float t = static_cast<float>(x[i]) - mean;
        normalizer += t * t;
      }
    }
  }
  normalizer = BlockReduceT{block, temp}.Sum(normalizer);
  normalizer = rsqrt(normalizer / axis_size + eps);

  // Outputs.
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    auto xn = load_vector<N_READS>(x, index, axis_size, T(0));
    auto wn = load_vector<N_READS>(w, index, axis_size, w_stride, T(0));
    auto bn = load_vector<N_READS>(b, index, axis_size, b_stride, T(0));
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      float norm = (static_cast<float>(xn[i]) - mean) * normalizer;
      xn[i] = wn[i] * static_cast<T>(norm) + bn[i];
    }
    store_vector<N_READS>(out, index, xn, axis_size);
  }
}

template <typename T, bool HAS_W, int BLOCK_DIM, int N_READS = 4>
__global__ void layer_norm_vjp(
    const T* x,
    const T* w,
    const T* g,
    T* gx,
    T* gw,
    float eps,
    int32_t axis_size,
    int64_t w_stride) {
  auto grid = cg::this_grid();
  auto block = cg::this_thread_block();

  using BlockReduceF = BlockBroadcastReduce<float, BLOCK_DIM>;
  using BlockReduceF3 = BlockBroadcastReduce<float3, BLOCK_DIM>;
  __shared__ union {
    typename BlockReduceF::TempStorage f;
    typename BlockReduceF3::TempStorage f3;
  } temp;

  x += grid.block_rank() * axis_size;
  g += grid.block_rank() * axis_size;
  gx += grid.block_rank() * axis_size;
  gw += grid.block_rank() * axis_size;

  // Sum.
  float sum = 0;
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    auto xn = load_vector<N_READS>(x, index, axis_size, T(0));
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      sum += static_cast<float>(xn[i]);
    }
  }
  sum = BlockReduceF{block, temp.f}.Sum(sum);

  // Mean.
  float mean = sum / axis_size;

  // Normalizer.
  float3 factors = {};
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    auto gn = load_vector<N_READS>(g, index, axis_size, T(0));
    auto wn = load_vector<N_READS>(w, index, axis_size, w_stride, T(0));

    if ((index + 1) * N_READS <= axis_size) {
      auto xn = load_vector<N_READS>(x, index);
#pragma unroll
      for (int i = 0; i < N_READS; ++i) {
        float t = static_cast<float>(xn[i]) - mean;
        float wi = wn[i];
        float gi = gn[i];
        float wg = wi * gi;
        factors = plus_f3(factors, {wg, wg * t, t * t});
      }
    } else {
      for (int i = index * N_READS; i < axis_size; ++i) {
        float t = static_cast<float>(x[i]) - mean;
        float wi = wn[i];
        float gi = gn[i];
        float wg = wi * gi;
        factors = plus_f3(factors, {wg, wg * t, t * t});
      }
    }
  }
  factors = BlockReduceF3{block, temp.f3}.Reduce(factors, plus_f3, {});
  float meanwg = factors.x / axis_size;
  float meanwgxc = factors.y / axis_size;
  float normalizer2 = 1 / (factors.z / axis_size + eps);
  float normalizer = sqrt(normalizer2);

  // Outputs.
  for (int r = 0; r < cuda::ceil_div(axis_size, BLOCK_DIM * N_READS); ++r) {
    auto index = r * BLOCK_DIM + block.thread_rank();
    auto xn = load_vector<N_READS>(x, index, axis_size, T(0));
    auto gn = load_vector<N_READS>(g, index, axis_size, T(0));
    auto wn = load_vector<N_READS>(w, index, axis_size, w_stride, T(0));

    for (int i = 0; i < N_READS; i++) {
      float xi = (static_cast<float>(xn[i]) - mean) * normalizer;
      float wi = wn[i];
      float gi = gn[i];
      xn[i] = normalizer * (wi * gi - meanwg) - xi * meanwgxc * normalizer2;
      if constexpr (HAS_W) {
        wn[i] = gi * xi;
      }
    }
    store_vector<N_READS>(gx, index, xn, axis_size);
    if constexpr (HAS_W) {
      store_vector<N_READS>(gw, index, wn, axis_size);
    }
  }
}

} // namespace cu

namespace fast {

bool LayerNorm::use_fallback(Stream s) {
  return s.device == Device::cpu;
}

// TODO: There are duplicate code with backend/metal/normalization.cpp
void LayerNorm::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("LayerNorm::eval_gpu");
  auto& s = stream();
  auto& out = outputs[0];

  // Make sure that the last dimension is contiguous.
  auto set_output = [&s, &out](const array& x) {
    bool no_copy = x.flags().contiguous && x.strides()[x.ndim() - 1] == 1;
    if (no_copy && x.ndim() > 1) {
      auto s = x.strides()[x.ndim() - 2];
      no_copy &= (s == 0 || s == x.shape().back());
    }
    if (no_copy) {
      if (x.is_donatable()) {
        out.copy_shared_buffer(x);
      } else {
        out.set_data(
            allocator::malloc(x.data_size() * x.itemsize()),
            x.data_size(),
            x.strides(),
            x.flags());
      }
      return x;
    } else {
      array x_copy = contiguous_copy_gpu(x, s);
      out.copy_shared_buffer(x_copy);
      return x_copy;
    }
  };

  const array x = set_output(inputs[0]);
  const array& w = inputs[1];
  const array& b = inputs[2];

  int32_t axis_size = x.shape().back();
  int32_t n_rows = x.data_size() / axis_size;
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;
  int64_t b_stride = (b.ndim() == 1) ? b.strides()[0] : 0;

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(b);
  encoder.set_output_array(out);
  dispatch_float_types(out.dtype(), "layernorm", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    constexpr int N_READS = 16 / sizeof(DataType);
    dispatch_block_dim(cuda::ceil_div(axis_size, N_READS), [&](auto block_dim) {
      auto kernel = cu::layer_norm<DataType, block_dim(), N_READS>;
      encoder.add_kernel_node(
          kernel,
          n_rows,
          block_dim(),
          x.data<DataType>(),
          w.data<DataType>(),
          b.data<DataType>(),
          out.data<DataType>(),
          eps_,
          axis_size,
          w_stride,
          b_stride);
    });
  });
}

void LayerNormVJP::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  nvtx3::scoped_range r("LayerNormVJP::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  // Ensure row contiguity. We could relax this step by checking that the array
  // is contiguous (no broadcasts or holes) and that the input strides are the
  // same as the cotangent strides but for now this is simpler.
  auto check_input = [&s](const array& x, bool& copied) {
    if (x.flags().row_contiguous) {
      copied = false;
      return x;
    }
    copied = true;
    return contiguous_copy_gpu(x, s);
  };
  bool donate_x = inputs[0].is_donatable();
  bool donate_g = inputs[3].is_donatable();
  bool copied;
  auto x = check_input(inputs[0], copied);
  donate_x |= copied;
  const array& w = inputs[1];
  const array& b = inputs[2];
  bool g_copied;
  auto g = check_input(inputs[3], g_copied);
  donate_g |= g_copied;
  array& gx = outputs[0];
  array& gw = outputs[1];
  array& gb = outputs[2];

  // Check whether we had a weight.
  bool has_w = w.ndim() != 0;

  // Allocate space for the outputs.
  bool g_in_gx = false;
  if (donate_x) {
    gx.copy_shared_buffer(x);
  } else if (donate_g) {
    gx.copy_shared_buffer(g);
    g_in_gx = true;
  } else {
    gx.set_data(allocator::malloc(gx.nbytes()));
  }
  if (g_copied && !g_in_gx) {
    encoder.add_temporary(g);
  }

  int32_t axis_size = x.shape().back();
  int32_t n_rows = x.data_size() / axis_size;
  int64_t w_stride = (w.ndim() == 1) ? w.strides()[0] : 0;

  // Allocate a temporary to store the gradients for w and allocate the output
  // gradient accumulators.
  array gw_temp =
      (has_w) ? array({n_rows, x.shape().back()}, gw.dtype(), nullptr, {}) : w;
  bool g_in_gw = false;
  if (has_w) {
    if (!g_in_gx && donate_g) {
      g_in_gw = true;
      gw_temp.copy_shared_buffer(g);
    } else {
      gw_temp.set_data(allocator::malloc(gw_temp.nbytes()));
      encoder.add_temporary(gw_temp);
    }
  }

  // The gradient for b in case we had a b.
  bool has_gb = (gb.ndim() == 1 && gb.size() == axis_size);
  if (has_gb) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    col_reduce(encoder, g, gb, Reduce::ReduceType::Sum, {0}, plan);
  }

  // Insert dependency if `g` was donated
  if ((g_in_gx || g_in_gw) && has_gb) {
    encoder.set_input_array(gb);
  }
  encoder.set_input_array(x);
  encoder.set_input_array(w);
  encoder.set_input_array(g);
  encoder.set_output_array(gx);
  encoder.set_output_array(gw_temp);
  dispatch_float_types(gx.dtype(), "layernorm_vjp", [&](auto type_tag) {
    dispatch_bool(has_w, [&](auto has_w_constant) {
      using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;
      constexpr int N_READS = 16 / sizeof(DataType);
      dispatch_block_dim(
          cuda::ceil_div(axis_size, N_READS), [&](auto block_dim) {
            auto kernel = cu::layer_norm_vjp<
                DataType,
                has_w_constant.value,
                block_dim(),
                N_READS>;
            encoder.add_kernel_node(
                kernel,
                n_rows,
                block_dim(),
                x.data<DataType>(),
                w.data<DataType>(),
                g.data<DataType>(),
                gx.data<DataType>(),
                gw_temp.data<DataType>(),
                eps_,
                axis_size,
                w_stride);
          });
    });
  });

  if (has_w) {
    ReductionPlan plan(
        ReductionOpType::ContiguousStridedReduce, {n_rows}, {axis_size});
    col_reduce(encoder, gw_temp, gw, Reduce::ReduceType::Sum, {0}, plan);
  }
}

} // namespace fast

} // namespace mlx::core
