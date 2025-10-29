// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/matmul.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/gemms/cublas_gemm.h"
#include "mlx/backend/cuda/gemms/gemv.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <numeric>

namespace mlx::core {

namespace {

std::tuple<bool, int64_t, array>
check_transpose(cu::CommandEncoder& enc, const Stream& s, const array& arr) {
  auto stx = arr.strides()[arr.ndim() - 2];
  auto sty = arr.strides()[arr.ndim() - 1];
  if (sty == 1 && stx >= arr.shape(-1)) {
    return std::make_tuple(false, stx, arr);
  } else if (stx == 1 && sty >= arr.shape(-2)) {
    return std::make_tuple(true, sty, arr);
  } else {
    array arr_copy = contiguous_copy_gpu(arr, s);
    enc.add_temporary(arr_copy);
    return std::make_tuple(false, arr.shape(-1), arr_copy);
  }
}

void gemm_and_bias(
    cu::CommandEncoder& encoder,
    int M,
    int N,
    int K,
    bool a_transposed,
    int64_t lda,
    bool b_transposed,
    int64_t ldb,
    array& out,
    const array& a,
    const array& b,
    const std::optional<array>& bias = std::nullopt,
    float alpha = 1.0f) {
  // Check and collapse batch dimensions
  auto [batch_shape, a_batch_strides, b_batch_strides] = collapse_batches(a, b);

  auto batch_count = out.size() / (M * N);

  // Collapse batches into M if needed
  if (batch_count > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && a_batch_strides.back() == M * K &&
      b_batch_strides.back() == 0) {
    M *= batch_shape.back();
    batch_count = 1;

    a_batch_strides = {0};
    b_batch_strides = {0};
    batch_shape = {1};
  }

  // Use gemmv when possible
  if (!bias && cu::can_use_gemv(M, N, K, a_transposed, b_transposed)) {
    out.set_data(allocator::malloc(out.nbytes()));
    cu::gemv(
        a,
        b,
        out,
        M,
        N,
        K,
        batch_count,
        batch_shape,
        a_batch_strides,
        b_batch_strides,
        encoder);
    return;
  }

  // Pad the output if needed for large enough gemms
  if (M > 16 && K >= 16 && M >= 16 && N % 8 != 0) {
    int N_up = ((N + 8 - 1) / 8) * 8;
    // a, b, c, d+3
    auto strides = out.strides();
    strides[out.ndim() - 2] = N_up;
    for (int i = out.ndim() - 3; i >= 0; --i) {
      strides[i] = strides[i + 1] * out.shape(i + 1);
    }
    size_t data_size = strides[0] * out.shape(0);
    out.set_data(
        allocator::malloc(data_size * out.itemsize()),
        data_size,
        strides,
        {false});
  } else {
    out.set_data(allocator::malloc(out.nbytes()));
  }
  std::cout << "IN A " << a.strides() << " SHAPE " << a.shape() << std::endl;
  std::cout << "IN B " << b.strides() << " SHAPE " << b.shape() << std::endl;
  std::cout << "OUT " << out.shape() << " STRIDES " << out.strides()
            << std::endl;

  // Invoke cublasLt
  CublasGemm gemm(
      encoder.device(),
      a.dtype(),
      a_transposed,
      M,
      K,
      lda,
      b_transposed,
      K,
      N,
      ldb,
      batch_shape.back(),
      a_batch_strides.back(),
      b_batch_strides.back());
  if (bias) {
    if (a.dtype() == complex64) {
      throw std::runtime_error(
          "[gemm_and_bias] complex64 bias epilogue isn’t supported in cublasLtMatmul.");
    }
    gemm.set_bias(encoder, *bias);
  }
  gemm.run(
      encoder, out, a, b, batch_shape, a_batch_strides, b_batch_strides, alpha);
}

} // namespace

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Matmul::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 2);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  // Return 0s if either input is empty.
  if (a_pre.size() == 0 || b_pre.size() == 0) {
    array zero(0, a_pre.dtype());
    encoder.add_temporary(zero);
    fill_gpu(zero, out, s);
    return;
  }

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  gemm_and_bias(
      encoder, M, N, K, a_transposed, lda, b_transposed, ldb, out, a, b);
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("AddMM::eval_gpu");
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  assert(inputs.size() == 3);
  auto& a_pre = inputs[0];
  auto& b_pre = inputs[1];
  auto c = inputs[2];

  /////////////////////////////////////////////////////////////////////////////
  // Init checks and prep

  int M = a_pre.shape(-2);
  int N = b_pre.shape(-1);
  int K = a_pre.shape(-1);

  // Keep a vector with copies to be cleared in the completed buffer to release
  // the arrays
  auto [a_transposed, lda, a] = check_transpose(encoder, s, a_pre);
  auto [b_transposed, ldb, b] = check_transpose(encoder, s, b_pre);

  /////////////////////////////////////////////////////////////////////////////
  // Dispatch to GEMM with epilogue or AddMM

  if (beta_ == 1 && a.dtype() != complex64 && c.strides(-1) == 1 &&
      c.data_size() == out.shape(-1)) {
    gemm_and_bias(
        encoder,
        M,
        N,
        K,
        a_transposed,
        lda,
        b_transposed,
        ldb,
        out,
        a,
        b,
        c,
        alpha_);
    return;
  }

  int64_t ldc;
  {
    auto stx = c.strides()[c.ndim() - 2];
    auto sty = c.strides()[c.ndim() - 1];
    if (sty == 1 && stx == c.shape(-1)) {
      ldc = stx;
      out.set_data(allocator::malloc(out.nbytes()));
    } else if (sty == 1 && stx == 0) {
      ldc = 0;
      out.set_data(allocator::malloc(out.nbytes()));
    } else {
      // Copy C into out and set C to out
      ldc = c.shape(-1);
      copy_gpu(c, out, CopyType::General, s);
      c = out;
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Check and collapse batch dimensions

  auto [batch_shape, a_batch_strides, b_batch_strides, c_batch_strides] =
      collapse_batches(a, b, c);

  auto batch_count = out.size() / (M * N);

  // Collapse batches into M if needed
  if (batch_count > 1 && !a_transposed && batch_shape.size() == 1 &&
      a.strides()[a.ndim() - 2] == K && a_batch_strides.back() == M * K &&
      c_batch_strides.back() == M * c.strides()[c.ndim() - 2] &&
      b_batch_strides.back() == 0) {
    M *= batch_shape.back();
    batch_count = 1;

    a_batch_strides = {0};
    b_batch_strides = {0};
    c_batch_strides = {0};
    batch_shape = {1};
  }

  /////////////////////////////////////////////////////////////////////////////
  // Invoke cublasLt with AddMM settings

  CublasGemm gemm(
      cu::device(s.device),
      a.dtype(),
      a_transposed,
      M,
      K,
      lda,
      b_transposed,
      K,
      N,
      ldb,
      ldc,
      batch_shape.back(),
      a_batch_strides.back(),
      b_batch_strides.back(),
      c_batch_strides.back());
  gemm.run(
      encoder,
      out,
      a,
      b,
      c,
      batch_shape,
      a_batch_strides,
      b_batch_strides,
      c_batch_strides,
      alpha_,
      beta_);
}

} // namespace mlx::core
