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

#include "graphlearn_torch/include/common.h"
#include "graphlearn_torch/csrc/cuda/random_negative_sampler.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include "graphlearn_torch/include/common.cuh"


namespace graphlearn_torch {

inline __device__ bool EdgeInCSR(const int64_t* row_ptr,
                                 const int64_t* col_idx,
                                 int64_t r,
                                 int64_t c) {
  int64_t low = row_ptr[r];
  int64_t high = row_ptr[r + 1] - 1;
  while(low <= high) {
    int64_t mid = (low + high) / 2;
    if(col_idx[mid] == c) {
      return true;
    } else if (col_idx[mid] > c) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return false;
}

__global__ void RandomNegativeSampleKernel(const int64_t rand_seed,
                                           const int64_t* row_ptr,
                                           const int64_t* col_idx,
                                           int64_t row_num,
                                           int64_t col_num,
                                           int32_t req_num,
                                           int32_t trials_num,
                                           bool strict,
                                           int64_t* out_row,
                                           int64_t* out_col,
                                           int32_t* out_prefix) {
  curandState rng1, rng2;
  int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(rand_seed * gridDim.x + blockIdx.x, tid, 0, &rng1);
  curand_init(rand_seed * gridDim.x + blockIdx.x, tid, 0, &rng2);

  const int stride = gridDim.x * blockDim.x;
  while (tid < req_num) {
    if (strict) {
      for (int32_t i = 0; i < trials_num; ++i) {
        int64_t r = curand(&rng1) % (row_num);
        int64_t c = curand(&rng2) % (col_num);
        if (!EdgeInCSR(row_ptr, col_idx, r, c)) {
          out_row[tid] = r;
          out_col[tid] = c;
          out_prefix[tid] = 1;
          break;
        }
      }
    } else { // non-strict negative sampling.
      unsigned int r = curand(&rng1) % (row_num);
      unsigned int c = curand(&rng2) % (col_num);
      out_row[tid] = r;
      out_col[tid] = c;
    }
    tid += stride;
  }
}

void SortByIndex(int64_t* row_data,
                 int64_t* col_data,
                 const int32_t* out_prefix,
                 int32_t req_num) {
  int32_t* keys = static_cast<int32_t*>(CUDAAlloc(sizeof(int32_t) * req_num));
  int64_t* rows = static_cast<int64_t*>(CUDAAlloc(sizeof(int64_t) * req_num));
  int64_t* cols = static_cast<int64_t*>(CUDAAlloc(sizeof(int64_t) * req_num));
  cudaMemsetAsync((void*)keys, 0, sizeof(int32_t) * req_num);
  cudaMemcpyAsync(rows, row_data, sizeof(int64_t) * req_num, cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync(cols, col_data, sizeof(int64_t) * req_num, cudaMemcpyDeviceToDevice);
  thrust::copy_if(thrust::device,
                  thrust::make_counting_iterator<int32_t>(0),
                  thrust::make_counting_iterator<int32_t>(req_num),
                  out_prefix,
                  keys,
                  thrust::placeholders::_1 == 1);
  thrust::gather(thrust::device, keys, keys + req_num, rows, row_data);
  thrust::gather(thrust::device, keys, keys + req_num, cols, col_data);
  CUDADelete((void*) keys);
  CUDADelete((void*) rows);
  CUDADelete((void*) cols);
}

std::tuple<torch::Tensor, torch::Tensor>
CUDARandomNegativeSampler::Sample(int32_t req_num,
                                  int32_t trials_num,
                                  bool padding) {
  int current_device = 0;
  cudaGetDevice(&current_device);
  auto stream = ::at::cuda::getDefaultCUDAStream();
  ::at::cuda::CUDAStreamGuard guard(stream);

  const int64_t* row_ptr = graph_->GetRowPtr();
  const int64_t* col_idx = graph_->GetColIdx();
  int64_t row_num = graph_->GetRowCount();
  int64_t col_num = graph_->GetColCount();
  int64_t* row_data = static_cast<int64_t*>(CUDAAlloc(sizeof(int64_t) * req_num));
  int64_t* col_data = static_cast<int64_t*>(CUDAAlloc(sizeof(int64_t) * req_num));
  int32_t* out_prefix = static_cast<int32_t*>(CUDAAlloc(sizeof(int32_t) * req_num));
  cudaMemsetAsync((void*)out_prefix, 0, sizeof(int32_t) * req_num, stream);

  int block_size = 0;
  int grid_size = 0;
  uint32_t seed = RandomSeedManager::getInstance().getSeed();
  thread_local static std::mt19937 engine(seed);
  std::uniform_int_distribution<int64_t> dist(0, 1e10);
  cudaOccupancyMaxPotentialBlockSize(
    &grid_size, &block_size, RandomNegativeSampleKernel);
  RandomNegativeSampleKernel<<<grid_size, block_size, 0, stream>>>(
    dist(engine), row_ptr, col_idx, row_num, col_num, req_num,
    trials_num, true, row_data, col_data, out_prefix);
  CUDACheckError();
  // sort by index.
  SortByIndex(row_data, col_data, out_prefix, req_num);

  CUDAAllocator allocator(stream);
  const auto policy = thrust::cuda::par(allocator).on(stream);
  auto sampled_num = thrust::reduce(policy, out_prefix, out_prefix + req_num);
  if((sampled_num < req_num) && padding) { // non-strict negative sampling.
    RandomNegativeSampleKernel<<<grid_size, block_size, 0, stream>>>(
      dist(engine), row_ptr, col_idx, row_num, col_num,
      req_num-sampled_num, 0, false,
      row_data + sampled_num, col_data + sampled_num, out_prefix);
    CUDACheckError();
    sampled_num = req_num;
  }

  torch::Tensor rows = torch::empty(
    sampled_num, torch::dtype(torch::kInt64).device(torch::kCUDA, current_device));
  torch::Tensor cols = torch::empty(
    sampled_num, torch::dtype(torch::kInt64).device(torch::kCUDA, current_device));
  cudaMemcpyAsync((void*)rows.data_ptr<int64_t>(), (void*)row_data,
      sizeof(int64_t) * sampled_num, cudaMemcpyDeviceToDevice);
  cudaMemcpyAsync((void*)cols.data_ptr<int64_t>(), (void*)col_data,
      sizeof(int64_t) * sampled_num, cudaMemcpyDeviceToDevice);
  CUDADelete((void*) out_prefix);
  CUDADelete((void*) row_data);
  CUDADelete((void*) col_data);
  return std::make_tuple(rows, cols);
}

} // namespace graphlearn_torch