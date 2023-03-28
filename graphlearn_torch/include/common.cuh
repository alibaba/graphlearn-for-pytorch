/* Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef GRAPHLEARN_TORCH_INCLUDE_COMMON_CUH_
#define GRAPHLEARN_TORCH_INCLUDE_COMMON_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace graphlearn_torch
{

enum class DataType {
  Int32 = 0,
  Int64 = 1,
  Float32 = 2,
  Float64 = 3
};

inline void Check(bool val, const char* err_msg) {
  if (val) { return; }
  throw std::runtime_error(err_msg);
}

template <typename T>
inline void CheckEq(const T &x, const T &y) {
  if (x == y) { return; }
  throw std::runtime_error(std::string("CheckEq failed"));
}

inline void CUDARegisterByBlock(void* ptr, size_t size, unsigned int flags) {
  constexpr size_t BLOCK = 1000000000;
  auto* register_ptr = static_cast<int8_t*>(ptr);
  for (size_t pos = 0; pos < size; pos += BLOCK) {
    auto s = std::min(size - pos, BLOCK);
    cudaHostRegister(register_ptr + pos, s, flags);
  }
}

inline cudaError_t CUDAMemcpy(void* dst, const void* src, size_t count,
    cudaMemcpyKind kind, cudaStream_t stream = 0) {
  #ifdef CUDA_VERSION_LT112
  return cudaMemcpy(dst, src, count, kind);
  #else
  return cudaMemcpyAsync(dst, src, count, kind, stream);
  #endif
}

inline cudaError_t CUDAFree(void* ptr, cudaStream_t stream = 0) {
  #ifdef CUDA_VERSION_LT112
  return cudaFree(ptr);
  #else
  return cudaFreeAsync(ptr, stream);
  #endif
}

#define CUDACheckError()                                                   \
{                                                                          \
  cudaError_t e = cudaGetLastError();                                      \
  if ((e != cudaSuccess) && (e != cudaErrorPeerAccessAlreadyEnabled)) {    \
    printf("CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__,               \
    cudaGetErrorString(e));                                                \
    exit(EXIT_FAILURE);                                                    \
  }                                                                        \
}

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_INCLUDE_COMMON_CUH_
