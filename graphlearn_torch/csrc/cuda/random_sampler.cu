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
#include "graphlearn_torch/csrc/cuda/random_sampler.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <cub/cub.cuh>
#include <curand.h>
#include <curand_kernel.h>

#include "graphlearn_torch/include/common.cuh"


namespace graphlearn_torch {

constexpr int32_t WARP_SIZE = 128;
constexpr int32_t BLOCK_WARPS = 1;
// The number of rows covered by each threadblock.
constexpr int32_t TILE_SIZE = BLOCK_WARPS;

__global__ void CSRRowWiseFillNbrsNumKernel(const int64_t* nodes,
                                            const int32_t bs,
                                            const int32_t req_num,
                                            const int64_t row_count,
                                            const int64_t* row_ptr,
                                            int64_t* out_nbr_num) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < bs) {
    auto v = nodes[tid];
    if (v < row_count) {
      out_nbr_num[tid] = min(static_cast<int64_t>(req_num),
                            row_ptr[v+1]-row_ptr[v]);
    } else {
      out_nbr_num[tid] = 0;
    }
    if (tid == bs - 1) {
      // make the prefixsum work
      out_nbr_num[bs] = 0;
    }
  }
}

// We refer to the dgl implementation
__global__ void CSRRowWiseSampleKernel(const int64_t rand_seed,
                                       const int32_t req_num,
                                       const int64_t bs,
                                       const int64_t row_count,
                                       const int64_t* nodes,
                                       const int64_t* row_ptr,
                                       const int64_t* col_idx,
                                       const int64_t* nbrs_offset,
                                       int64_t* out_nbrs) {
  // each warp take charge of one-row sampling.
  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  int64_t last_row = min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, bs);
  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x,
              threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);

  while ((out_row < last_row) && (nodes[out_row] < row_count)) {
    int64_t row = nodes[out_row];
    int64_t in_row_start = row_ptr[row];
    int64_t deg = row_ptr[row + 1] - in_row_start;
    int64_t out_row_start = nbrs_offset[out_row];
    // without replacement
    if (deg <= req_num) {
      for (int32_t idx = threadIdx.x; idx < deg; idx += WARP_SIZE) {
        int64_t in_idx = in_row_start + idx;
        out_nbrs[out_row_start + idx] = col_idx[in_idx];
      }
    } else {
      // reservoir algorithm
      for (int32_t idx = threadIdx.x; idx < req_num; idx += WARP_SIZE) {
        out_nbrs[out_row_start + idx] = idx;
      }
      __syncthreads();
      for (int32_t idx = req_num + threadIdx.x; idx < deg; idx += WARP_SIZE) {
        int32_t j = curand(&rng) % (idx + 1);
        if (j < req_num) {
          using T = unsigned long long int;
          // keep the order of the sample stream
          atomicMax(reinterpret_cast<T*>(out_nbrs + out_row_start + j),
                    static_cast<T>(idx));
        }
      }
      __syncthreads();
      for (int32_t idx = threadIdx.x; idx < req_num; idx += WARP_SIZE) {
        int64_t perm_idx = out_nbrs[out_row_start + idx] + in_row_start;
        out_nbrs[out_row_start + idx] = col_idx[perm_idx];
      }
    }
    out_row += BLOCK_WARPS;
  }
}

__global__ void CSRRowWiseSampleKernel(const int64_t rand_seed,
                                       const int32_t req_num,
                                       const int64_t bs,
                                       const int64_t row_count,
                                       const int64_t* nodes,
                                       const int64_t* row_ptr,
                                       const int64_t* col_idx,
                                       const int64_t* edge_ids,
                                       const int64_t* nbrs_offset,
                                       int64_t* out_nbrs,
                                       int64_t* out_eid) {
  // each warp take charge of one-row sampling.
  int64_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  int64_t last_row = min(static_cast<int64_t>(blockIdx.x + 1) * TILE_SIZE, bs);
  curandStatePhilox4_32_10_t rng;
  curand_init(rand_seed * gridDim.x + blockIdx.x,
              threadIdx.y * WARP_SIZE + threadIdx.x, 0, &rng);

  while ((out_row < last_row) && (nodes[out_row] < row_count)) {
    int64_t row = nodes[out_row];
    int64_t in_row_start = row_ptr[row];
    int64_t deg = row_ptr[row + 1] - in_row_start;
    int64_t out_row_start = nbrs_offset[out_row];
    // without replacement
    if (deg <= req_num) {
      for (int32_t idx = threadIdx.x; idx < deg; idx += WARP_SIZE) {
        int64_t in_idx = in_row_start + idx;
        out_nbrs[out_row_start + idx] = col_idx[in_idx];
        out_eid[out_row_start + idx] = edge_ids[in_idx];
      }
    } else {
      // reservoir algorithm
      for (int32_t idx = threadIdx.x; idx < req_num; idx += WARP_SIZE) {
        out_nbrs[out_row_start + idx] = idx;
      }
      __syncthreads();
      for (int32_t idx = req_num + threadIdx.x; idx < deg; idx += WARP_SIZE) {
        int32_t j = curand(&rng) % (idx + 1);
        if (j < req_num) {
          using T = unsigned long long int;
          // keep the order of the sample stream
          atomicMax(reinterpret_cast<T*>(out_nbrs + out_row_start + j),
                    static_cast<T>(idx));
        }
      }
      __syncthreads();
      for (int32_t idx = threadIdx.x; idx < req_num; idx += WARP_SIZE) {
        int64_t perm_idx = out_nbrs[out_row_start + idx] + in_row_start;
        out_nbrs[out_row_start + idx] = col_idx[perm_idx];
        out_eid[out_row_start + idx] = edge_ids[perm_idx];
      }
    }
    out_row += BLOCK_WARPS;
  }
}

template<typename T, typename S>
__global__ void CalNbrProbKernel(const S *last_prob,
                                 const S* nbr_last_prob,
                                 int row_count,
                                 int row_count_nbr,
                                 int k,
                                 const T *const in_ptr,
                                 const T *const in_index,
                                 const T *const nbr_in_ptr,
                                 S *cur_prob) {
  int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t step = gridDim.x * blockDim.x;
  while (row < row_count) {
    const int64_t in_row_start = in_ptr[row];
    const int64_t deg = in_ptr[row + 1] - in_row_start;
    S acc = 1.0;
    if (deg == 0) {
      cur_prob[row] = 0;
      row += step;
      continue;
    }
    for (int64_t i = in_row_start; i < in_row_start + deg; i++) {
      int64_t upper = in_index[i];
      int64_t upper_deg = 0;
      if (upper < row_count_nbr) {
        int64_t upper_start = nbr_in_ptr[upper];
        upper_deg = nbr_in_ptr[upper + 1] - upper_start;
      }
      S skip;
      if (upper_deg == 0) {
          skip = 1;
      } else if (upper_deg <= k) {
          skip = 1 - nbr_last_prob[upper];
      } else {
          skip = 1 - nbr_last_prob[upper] +
            nbr_last_prob[upper] * (upper_deg - k) / upper_deg;
      }
      acc *= skip;
    }
    cur_prob[row] = 1 - (1 - last_prob[row]) * acc;
    row += step;
  }
}

void FillNbrsNum(cudaStream_t stream,
                 const int64_t* nodes,
                 const int32_t bs,
                 const int32_t req_num,
                 const int64_t row_count,
                 const int64_t* row_ptr,
                 int64_t* out_nbr_num) {
  const dim3 block(512);
  const dim3 grid((bs+block.x-1)/block.x);
  CSRRowWiseFillNbrsNumKernel<<<grid, block, 0, stream>>>(
    nodes, bs, req_num, row_count, row_ptr, out_nbr_num);
  CUDACheckError();
}

void CSRRowWiseSample(cudaStream_t stream,
                      const int64_t* nodes,
                      const int64_t* nbrs_offset,
                      const int32_t bs,
                      const int32_t req_num,
                      const int64_t row_count,
                      const int64_t* row_ptr,
                      const int64_t* col_idx,
                      int64_t* out_nbrs) {
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  const dim3 grid((bs + TILE_SIZE - 1) / TILE_SIZE);
  uint32_t seed = RandomSeedManager::getInstance().getSeed();
  thread_local static std::mt19937 engine(seed);
  std::uniform_int_distribution<int64_t> dist(0, 1e10);
  CSRRowWiseSampleKernel<<<grid, block, 0, stream>>>(
    dist(engine), req_num, bs, row_count,
    nodes, row_ptr, col_idx, nbrs_offset, out_nbrs);
  CUDACheckError();
}

void CSRRowWiseSample(cudaStream_t stream,
                      const int64_t* nodes,
                      const int64_t* nbrs_offset,
                      const int32_t bs,
                      const int32_t req_num,
                      const int64_t row_count,
                      const int64_t* row_ptr,
                      const int64_t* col_idx,
                      const int64_t* edge_ids,
                      int64_t* out_nbrs,
                      int64_t* out_eid) {
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  const dim3 grid((bs + TILE_SIZE - 1) / TILE_SIZE);
  uint32_t seed = RandomSeedManager::getInstance().getSeed();
  thread_local static std::mt19937 engine(seed);
  std::uniform_int_distribution<int64_t> dist(0, 1e10);
  CSRRowWiseSampleKernel<<<grid, block, 0, stream>>>(
    dist(engine), req_num, bs, row_count, nodes, row_ptr, col_idx, edge_ids,
    nbrs_offset, out_nbrs, out_eid);
  CUDACheckError();
}

std::tuple<torch::Tensor, torch::Tensor>
CUDARandomSampler::Sample(const torch::Tensor& nodes, int32_t req_num) {
  auto stream = ::at::cuda::getDefaultCUDAStream();
  cudaEvent_t copyEvent;
  cudaEventCreate(&copyEvent);

  if (req_num < 0) req_num = std::numeric_limits<int32_t>::max();
  const auto nodes_ptr = nodes.data_ptr<int64_t>();
  int64_t bs = nodes.size(0);
  const auto row_ptr = graph_->GetRowPtr();
  const auto col_idx = graph_->GetColIdx();
  const auto row_count = graph_->GetRowCount();

  torch::Tensor nbrs_num = torch::empty(bs + 1, nodes.options());
  auto nbrs_num_ptr = nbrs_num.data_ptr<int64_t>();
  FillNbrsNum(stream, nodes_ptr, bs, req_num, row_count, row_ptr, nbrs_num_ptr);

  int64_t* nbrs_offset = static_cast<int64_t*>(
      CUDAAlloc(sizeof(int64_t) * (bs + 1), stream));
  cudaMemsetAsync((void*)nbrs_offset, 0, sizeof(int64_t) * (bs + 1), stream);

  size_t prefix_temp_size = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr, prefix_temp_size, nbrs_num_ptr, nbrs_offset, bs + 1, stream);
  void* prefix_temp = CUDAAlloc(prefix_temp_size, stream);
  cub::DeviceScan::ExclusiveSum(
      prefix_temp, prefix_temp_size, nbrs_num_ptr, nbrs_offset, bs + 1, stream);
  CUDADelete(prefix_temp);
  int64_t total_nbrs_num = 0;
  cudaMemcpyAsync((void*)&total_nbrs_num, (void*)(nbrs_offset + bs),
      sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaEventRecord(copyEvent, stream);
  cudaEventSynchronize(copyEvent);
  cudaEventDestroy(copyEvent);

  torch::Tensor nbrs = torch::empty(total_nbrs_num, nodes.options());
  CSRRowWiseSample(
    stream, nodes_ptr, nbrs_offset, bs, req_num, row_count, row_ptr, col_idx,
    nbrs.data_ptr<int64_t>());
  CUDADelete((void*) nbrs_offset);
  return std::make_tuple(nbrs, nbrs_num.narrow(0, 0, bs));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CUDARandomSampler::SampleWithEdge(const torch::Tensor& nodes, int32_t req_num) {
  auto stream = ::at::cuda::getDefaultCUDAStream();
  cudaEvent_t copyEvent;
  cudaEventCreate(&copyEvent);
  if (req_num < 0) req_num = std::numeric_limits<int32_t>::max();
  const auto nodes_ptr = nodes.data_ptr<int64_t>();
  int64_t bs = nodes.size(0);
  const auto row_ptr = graph_->GetRowPtr();
  const auto col_idx = graph_->GetColIdx();
  const auto edge_ids = graph_->GetEdgeId();
  const auto row_count = graph_->GetRowCount();

  torch::Tensor nbrs_num = torch::empty(bs + 1, nodes.options());
  auto nbrs_num_ptr = nbrs_num.data_ptr<int64_t>();
  FillNbrsNum(stream, nodes_ptr, bs, req_num, row_count, row_ptr, nbrs_num_ptr);

  int64_t* nbrs_offset = static_cast<int64_t*>(
      CUDAAlloc(sizeof(int64_t) * (bs + 1), stream));
  cudaMemsetAsync((void*)nbrs_offset, 0, sizeof(int64_t) * (bs + 1), stream);

  size_t prefix_temp_size = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr, prefix_temp_size, nbrs_num_ptr, nbrs_offset, bs + 1, stream);
  void* prefix_temp = CUDAAlloc(prefix_temp_size, stream);
  cub::DeviceScan::ExclusiveSum(
      prefix_temp, prefix_temp_size, nbrs_num_ptr, nbrs_offset, bs + 1, stream);
  CUDADelete(prefix_temp);
  int64_t total_nbrs_num = 0;
  cudaMemcpyAsync((void*)&total_nbrs_num, (void*)(nbrs_offset + bs),
      sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
  cudaEventRecord(copyEvent, stream);
  cudaEventSynchronize(copyEvent);
  cudaEventDestroy(copyEvent);

  torch::Tensor nbrs = torch::empty(total_nbrs_num, nodes.options());
  torch::Tensor out_eid = torch::empty(total_nbrs_num, nodes.options());
  CSRRowWiseSample(
    stream, nodes_ptr, nbrs_offset, bs, req_num, row_count, row_ptr, col_idx,
    edge_ids, nbrs.data_ptr<int64_t>(), out_eid.data_ptr<int64_t>());
  CUDADelete((void*) nbrs_offset);
  return std::make_tuple(nbrs, nbrs_num.narrow(0, 0, bs), out_eid);
}

void CUDARandomSampler::CalNbrProb(
    int k, const torch::Tensor& last_prob, const torch::Tensor& nbr_last_prob,
    const Graph* nbr_graph_, torch::Tensor cur_prob) {
  auto stream = ::at::cuda::getDefaultCUDAStream();
  ::at::cuda::CUDAStreamGuard guard(stream);

  const dim3 block(WARP_SIZE * BLOCK_WARPS);
  const dim3 grid((last_prob.size(0) + WARP_SIZE * BLOCK_WARPS - 1) /
    (WARP_SIZE * BLOCK_WARPS));
  const auto *indptr = graph_->GetRowPtr();
  const auto *indices = graph_->GetColIdx();
  const auto *nbr_indptr = nbr_graph_->GetRowPtr();

  CalNbrProbKernel<<<grid, block, 0, stream>>>(
    last_prob.data_ptr<float>(), nbr_last_prob.data_ptr<float>(),
    graph_->GetRowCount(), nbr_graph_->GetRowCount(), k, indptr, indices,
    nbr_indptr, cur_prob.data_ptr<float>());
  CUDACheckError();
}

}  // namespace graphlearn_torch
