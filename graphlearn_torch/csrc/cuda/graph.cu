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

#include <algorithm>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <unordered_set>


#include "graphlearn_torch/include/graph.h"
#include "graphlearn_torch/include/common.cuh"

namespace graphlearn_torch {

constexpr int32_t BLOCK_SIZE = 128;
constexpr int32_t TILE_SIZE = 4;

__global__ void LookupDegreeKernel(const int64_t* nodes,
                                   int32_t nodes_size,
                                   const int64_t* row_ptr,
                                   int32_t row_count,
                                   int64_t* degrees) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t stride_x = gridDim.x * blockDim.x;
  while (tid < nodes_size) {
    if (nodes[tid] < row_count) {
      int64_t in_row = nodes[tid];
      int64_t row_start = row_ptr[in_row];
      int64_t degree = row_ptr[in_row + 1] - row_start;
      degrees[tid] = degree;
    } else {
      degrees[tid] = 0;
    }
    tid += stride_x;
  }
}

Graph::~Graph() {
  for (auto& ptr : registered_ptrs_) {
    if (ptr != nullptr) {
      cudaHostUnregister(ptr);
    }
  }
}

void Graph::LookupDegree(const int64_t* nodes,
                         int32_t nodes_size,
                         int64_t* degrees) const {
  auto stream = ::at::cuda::getDefaultCUDAStream();
  ::at::cuda::CUDAStreamGuard guard(stream);
  const dim3 grid((nodes_size + TILE_SIZE - 1) / TILE_SIZE);
  const dim3 block(BLOCK_SIZE);
  LookupDegreeKernel<<<grid, block, 0, stream>>>(
      nodes, nodes_size, row_ptr_, row_count_, degrees);
}

void Graph::InitCUDAGraphFromCSR(const torch::Tensor& indptr,
                                 const torch::Tensor& indices,
                                 int device,
                                 GraphMode graph_mode,
                                 const torch::Tensor& edge_ids) {
  CheckEq<int64_t>(indptr.dim(), 1);
  CheckEq<int64_t>(indices.dim(), 1);
  int64_t indptr_size = indptr.size(0);
  int64_t indices_size = indices.size(0);
  const int64_t* input_indptr = indptr.data_ptr<int64_t>();
  const int64_t* input_indices = indices.data_ptr<int64_t>();
  int64_t* edge_ptr = nullptr;
  if (edge_ids.numel()) {
    CheckEq<int64_t>(edge_ids.dim(), 1);
    CheckEq<int64_t>(edge_ids.numel(), indices.numel());
    edge_ptr = edge_ids.data_ptr<int64_t>();
  }
  col_count_ = std::get<0>(at::_unique(indices)).size(0);
  InitCUDAGraphFromCSR(
    input_indptr, indptr_size, input_indices, indices_size,
    edge_ptr, device, graph_mode);
}

void Graph::InitCUDAGraphFromCSR(const int64_t* input_indptr,
                                 int64_t indptr_size,
                                 const int64_t* input_indices,
                                 int64_t indices_size,
                                 const int64_t* edge_ids,
                                 int device,
                                 GraphMode graph_mode) {
  graph_mode_ = graph_mode;
  device_id_ = device;
  cudaSetDevice(device);
  row_count_ = indptr_size - 1;
  edge_count_ = indices_size;
  if (!col_count_) {
    col_count_ = std::unordered_set<int64_t>(input_indices,
        input_indices + indices_size).size();
  }

  if (graph_mode == GraphMode::ZERO_COPY) {
    CUDARegisterByBlock((void*)input_indptr, sizeof(int64_t) * indptr_size,
                        cudaHostRegisterMapped);
    registered_ptrs_.push_back((void*)input_indptr);
    CUDACheckError();
    cudaHostGetDevicePointer((void**)&row_ptr_, (void*)input_indptr, 0);

    CUDARegisterByBlock((void*)input_indices, sizeof(int64_t) * indices_size,
                        cudaHostRegisterMapped);
    registered_ptrs_.push_back((void*)input_indices);
    CUDACheckError();
    cudaHostGetDevicePointer((void**)&col_idx_, (void*)input_indices, 0);

    if (edge_ids != nullptr) {
      CUDARegisterByBlock((void*)edge_ids, sizeof(int64_t) * indices_size,
                          cudaHostRegisterMapped);
      registered_ptrs_.push_back((void*)edge_ids);
      CUDACheckError();
      cudaHostGetDevicePointer((void**)&edge_id_, (void*)edge_ids, 0);
    }
  } else { // DMA
    cudaMalloc((void**)&row_ptr_, sizeof(int64_t) * indptr_size);
    cudaMemcpy((void*)row_ptr_, (void*)input_indptr,
               sizeof(int64_t) * indptr_size, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&col_idx_, sizeof(int64_t) * indices_size);
    cudaMemcpy((void*)col_idx_, (void *)input_indices,
               sizeof(int64_t) * indices_size, cudaMemcpyHostToDevice);
    if (edge_ids != nullptr) {
      cudaMalloc((void**)&edge_id_, sizeof(int64_t) * indices_size);
      cudaMemcpy((void*)edge_id_, (void *)edge_ids,
                 sizeof(int64_t) * indices_size, cudaMemcpyHostToDevice);
    }
  }
}

} // namespace graphlearn_torch
