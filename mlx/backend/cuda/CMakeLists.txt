# Filename rules in cuda backend:
#
# * Use .cu/.cuh if code contains device code, and .cpp/.h if not.
# * Device-only code should be put in device/ subdir.
# * Files in device/ subdir should not include files outside.
target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/arange.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/arg_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/binary.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/binary_two.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/compiled.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_contiguous.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general_dynamic.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general_input.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/cuda.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/gemv.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/gemms/cublas_gemm.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/jit_module.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/indexing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/kernel_utils.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/logsumexp.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/random.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/all_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/col_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/init_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/row_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/rms_norm.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/rope.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/scan.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/sort.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/ternary.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/unary.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/affine_quantize.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/quantized/quantized.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp)

if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9.0)
  target_sources(
    mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gemms/cublas_batched_gemm_12_9.cu)
else()
  target_sources(
    mlx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/gemms/cublas_batched_gemm_12_0.cpp)
endif()

target_compile_definitions(mlx PRIVATE MLX_USE_CUDA)

# Embed kernel sources in binary for JIT compilation.
file(
  GLOB MLX_JIT_SOURCES
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "${CMAKE_CURRENT_SOURCE_DIR}/device/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/device/*.cuh")
string(JOIN ":" MLX_JIT_SOURCES_ARG ${MLX_JIT_SOURCES})
add_custom_command(
  OUTPUT gen/cuda_jit_sources.h
  COMMAND
    ${CMAKE_COMMAND} -DMLX_SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}
    -DMLX_JIT_SOURCES=${MLX_JIT_SOURCES_ARG} -P
    "${CMAKE_CURRENT_SOURCE_DIR}/bin2h.cmake"
  DEPENDS bin2h.cmake ${MLX_JIT_SOURCES})
add_custom_target(cuda_jit_sources DEPENDS gen/cuda_jit_sources.h)
add_dependencies(mlx cuda_jit_sources)
target_include_directories(mlx PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/gen")

# Enable defining device lambda functions.
target_compile_options(mlx
                       PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")

# Enable calling host constexpr functions from device. This is needed because
# the constexpr version of isnan is host only.
target_compile_options(
  mlx PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>")

# CUDA 12.8 emits warning #20280-D for copy kernels which is a false positive.
# Explicitly pass this flag to suppress the warning, it is safe to set it to
# true but the warning wouldn't be suppressed.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8.0)
  target_compile_options(
    mlx
    PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--static-global-template-stub=false>")
endif()

# Suppress warning when building for compute capability 7 used by V100.
target_compile_options(
  mlx PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--Wno-deprecated-gpu-targets>")

# Use stronger binaries compression. This feature was introduced in CUDA 12.8
# and requires drivers released after CUDA 12.4.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8.0)
  target_compile_options(
    mlx PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--compress-mode=size>")
endif()

# Compute capability >= 7.0 is required for synchronization between CPU/GPU with
# managed memory.
if(NOT DEFINED MLX_CUDA_ARCHITECTURES)
  set(MLX_CUDA_ARCHITECTURES "native")
endif()
message(STATUS "CUDA architectures: ${MLX_CUDA_ARCHITECTURES}")
set_target_properties(mlx PROPERTIES CUDA_ARCHITECTURES
                                     "${MLX_CUDA_ARCHITECTURES}")

# Use fixed version of CCCL.
FetchContent_Declare(
  cccl
  URL "https://github.com/NVIDIA/cccl/releases/download/v2.8.1/cccl-v2.8.1.zip")
FetchContent_MakeAvailable(cccl)
target_include_directories(mlx BEFORE PRIVATE "${cccl_SOURCE_DIR}/include")

# Use fixed version of NVTX.
FetchContent_Declare(
  nvtx3
  GIT_REPOSITORY https://github.com/NVIDIA/NVTX.git
  GIT_TAG v3.1.1
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR c EXCLUDE_FROM_ALL)
FetchContent_MakeAvailable(nvtx3)
target_link_libraries(mlx PUBLIC $<BUILD_INTERFACE:nvtx3-cpp>)

# Make cuda runtime APIs available in non-cuda files.
find_package(CUDAToolkit REQUIRED)
target_include_directories(mlx PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# Use cublasLt.
target_link_libraries(mlx PRIVATE CUDA::cublasLt)

# Use NVRTC and driver APIs.
target_link_libraries(mlx PRIVATE CUDA::nvrtc CUDA::cuda_driver)

# Use the frontend APIs of cuDNN.
FetchContent_Declare(
  cudnn
  GIT_REPOSITORY https://github.com/NVIDIA/cudnn-frontend.git
  GIT_TAG v1.12.1
  GIT_SHALLOW TRUE
  EXCLUDE_FROM_ALL)
set(CUDNN_FRONTEND_SKIP_JSON_LIB ON)
set(CUDNN_FRONTEND_BUILD_SAMPLES OFF)
set(CUDNN_FRONTEND_BUILD_TESTS OFF)
set(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF)
FetchContent_MakeAvailable(cudnn)
target_link_libraries(mlx PRIVATE cudnn_frontend)
# Link with the actual cuDNN libraries.
include(${cudnn_frontend_SOURCE_DIR}/cmake/cuDNN.cmake)
target_link_libraries(mlx PRIVATE CUDNN::cudnn_all)

# Suppress nvcc warnings on MLX headers.
target_compile_options(mlx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe
                                   --diag_suppress=997>)

# Install CCCL headers for JIT.
install(DIRECTORY ${cccl_SOURCE_DIR}/include/cuda
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cccl)
