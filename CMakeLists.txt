cmake_minimum_required(VERSION 3.16)
project(edgecert_kernels C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenMP QUIET)

# SumFold product-sumcheck kernel, host driver (CLI) and transcript helper.
add_executable(sumfold_cpu
    kernels/cpu/main.c
    kernels/cpu/sumfold.c
    kernels/cpu/sha256.c)
target_include_directories(sumfold_cpu PRIVATE kernels/cpu)
target_compile_options(sumfold_cpu PRIVATE -O3 -funroll-loops -Wall)
if(OpenMP_C_FOUND)
    target_link_libraries(sumfold_cpu PRIVATE OpenMP::OpenMP_C)
endif()

# MLE contraction helpers, loaded from Python through ctypes (kernel_cpu_bench / matmul_sumcheck).
add_library(mle_shared SHARED
    kernels/cpu/mle.c
    kernels/cpu/sha256.c)
target_include_directories(mle_shared PRIVATE kernels/cpu)
target_compile_options(mle_shared PRIVATE -O3 -fPIC -funroll-loops -Wall)

# CUDA kernels are optional: configure with -DEDGECERT_CUDA=ON on a machine with the CUDA toolkit.
option(EDGECERT_CUDA "build the CUDA SumFold kernel (needs nvcc)" OFF)
if(EDGECERT_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_library(sumfold_cuda STATIC kernels/cuda/sumfold.cu)
    target_compile_options(sumfold_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_link_libraries(sumfold_cuda PRIVATE CUDA::cudart)
    add_library(cuda_host_shim SHARED kernels/cuda/shim/shim_launch.cpp)
    target_include_directories(cuda_host_shim PRIVATE kernels/cuda/shim)
    target_link_libraries(cuda_host_shim PRIVATE sumfold_cuda CUDA::cudart)
endif()
