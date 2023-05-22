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

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <stdexcept>

#include "graphlearn_torch/include/common.h"


namespace graphlearn_torch
{

inline void CUDARegisterByBlock(void* ptr, size_t size, unsigned int flags) {
  constexpr size_t BLOCK = 1000000000;
  auto* register_ptr = static_cast<int8_t*>(ptr);
  for (size_t pos = 0; pos < size; pos += BLOCK) {
    auto s = std::min(size - pos, BLOCK);
    cudaHostRegister(register_ptr + pos, s, flags);
  }
}

inline void* CUDAAlloc(size_t nbytes, cudaStream_t stream = 0) {
  at::globalContext().lazyInitCUDA();
  return c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(nbytes, stream);
}

inline void CUDADelete(void* ptr) {
  c10::cuda::CUDACachingAllocator::raw_delete(ptr);
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

class CUDAAllocator {
 public:
   typedef char value_type;

  explicit CUDAAllocator(cudaStream_t stream) : stream_(stream) {}

  void operator()(void* ptr) const {
    CUDADelete(ptr);
  }

  template <typename T>
  std::unique_ptr<T, CUDAAllocator> alloc_unique(std::size_t size) const {
    return std::unique_ptr<T, CUDAAllocator>(
        reinterpret_cast<T*>(CUDAAlloc(size, stream_)), *this);
  }

  char* allocate(std::ptrdiff_t size) const {
    return reinterpret_cast<char*>(CUDAAlloc(size, stream_));
  }

  void deallocate(char* ptr, std::size_t) const {
    CUDADelete(ptr);
  }
private:
  cudaStream_t stream_;
};

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_INCLUDE_COMMON_CUH_
