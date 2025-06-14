# Filename rules in cuda backend:
#
# * Use .cu/.cuh if code contains device code, and .cpp/.h if not.
# * Device-only kernel code should be put in kernels/ subdir.
# * Files in kernels/ subdir should not include files outside.
target_sources(
  mlx
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/allocator.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/arg_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/binary.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_contiguous.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general_dynamic.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/copy/copy_general_input.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/cuda.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/device.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/eval.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/event.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/fence.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/kernel_utils.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/matmul.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/layer_norm.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/logsumexp.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/primitives.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/random.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/col_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/row_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/reduce/segmented_reduce.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/slicing.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/softmax.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/sort.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/unary.cu
          ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
          ${CMAKE_CURRENT_SOURCE_DIR}/worker.cpp)

target_compile_definitions(mlx PRIVATE MLX_USE_CUDA)

# Enable defining device lambda functions.
target_compile_options(mlx
                       PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")

# CUDA 12.8 emits warning #20280-D for copy kernels which is a false positive.
# Explicitly pass this flag to suppress the warning, it is safe to set it to
# true but the warning wouldn't be suppressed.
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
  target_compile_options(
    mlx
    PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--static-global-template-stub=false>")
endif()

# Compute capability 7 is required for synchronization between CPU/GPU with
# managed memory. TODO: Add more architectures for potential performance gain.
set(MLX_CUDA_ARCHITECTURES
    "70;80"
    CACHE STRING "CUDA architectures")
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

# Suppress nvcc warnings on MLX headers.
target_compile_options(mlx PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe
                                   --diag_suppress=997>)
