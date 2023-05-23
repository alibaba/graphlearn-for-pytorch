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

#include "graphlearn_torch/csrc/cuda/subgraph_op.cuh"

#include <cub/cub.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include "graphlearn_torch/include/common.cuh"
#include "graphlearn_torch/include/hash_table.cuh"

namespace graphlearn_torch {

constexpr int32_t BLOCK_SIZE = 128;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t BLOCK_WARPS = 4;
// The number of rows covered by each threadblock.
constexpr int32_t TILE_SIZE = BLOCK_WARPS;


__global__ void GetNbrsNumKernel(DeviceHashTable table,
                                 const int64_t* nodes,
                                 const int64_t* sub_indptr,
                                 int32_t nodes_size,
                                 const int64_t* row_ptr,
                                 const int64_t* col_idx,
                                 int64_t row_count,
                                 int64_t* out_nbrs_num,
                                 int32_t* col_mask) {
  // each block takes charge of TILE_SIZE nodes.
  int32_t row_idx = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int32_t last_row = min((blockIdx.x + 1) * TILE_SIZE, nodes_size);

  typedef cub::WarpReduce<int64_t> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[BLOCK_WARPS];
  while ((row_idx < last_row) && (nodes[row_idx] < row_count)) {
    int64_t in_row = nodes[row_idx];
    int64_t row_start = row_ptr[in_row];
    int64_t degree = row_ptr[in_row + 1] - row_start;
    int64_t local_count = 0;
    for (int32_t idx = threadIdx.x; idx < degree; idx += WARP_SIZE) {
      auto new_col_idx = table.FindValue(col_idx[idx + row_start]);
      if (new_col_idx >= 0) {
        ++local_count;
        // in the input order of nodes.
        col_mask[idx + sub_indptr[row_idx]] = 1;
      }
    }
    int64_t nbr_count = WarpReduce(temp_storage[threadIdx.y]).Sum(local_count);
    if (threadIdx.x == 0) {
      out_nbrs_num[row_idx] = nbr_count;
    }
    row_idx += BLOCK_WARPS;
  }
}

__global__ void GetColIndexKernel(DeviceHashTable table,
                                  const int64_t* nodes,
                                  const int64_t* sub_indptr,
                                  const int32_t* col_offset,
                                  int32_t nodes_size,
                                  const int64_t* row_ptr,
                                  const int64_t* col_idx,
                                  const int64_t* edge_ids,
                                  int64_t row_count,
                                  bool with_edge,
                                  int64_t* out_cols,
                                  int64_t* out_eids) {
  // each block takes charge of TILE_SIZE nodes.
  int32_t row_idx = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int32_t last_row = min((blockIdx.x + 1) * TILE_SIZE, nodes_size);
  while ((row_idx < last_row) && (nodes[row_idx] < row_count)) {
    int64_t in_row = nodes[row_idx];
    int64_t row_start = row_ptr[in_row];
    int64_t degree = row_ptr[in_row + 1] - row_start;
    for (int32_t idx = threadIdx.x; idx < degree; idx += WARP_SIZE) {
      auto new_col_idx = table.FindValue(col_idx[idx + row_start]);
      if (new_col_idx >= 0) {
        auto offset = col_offset[idx + sub_indptr[row_idx]];
        out_cols[offset] = new_col_idx;
        if (with_edge) out_eids[offset] = edge_ids[idx + row_start];
      }
    }
    row_idx += BLOCK_WARPS;
  }
}


__global__ void GetRowIndexKernel(const int64_t* nodes,
                                  const int64_t* nbrs_num,
                                  const int64_t* nbrs_offset,
                                  int32_t nodes_size,
                                  int64_t* out_rows) {
  // each block takes charge of TILE_SIZE nodes.
  int32_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int32_t last_row = min((blockIdx.x + 1) * TILE_SIZE, nodes_size);
  while (out_row < last_row) {
    auto out_start = nbrs_offset[out_row];
    for (int32_t idx = threadIdx.x; idx < nbrs_num[out_row]; idx += WARP_SIZE) {
      out_rows[out_start + idx] = out_row;
    }
    out_row += BLOCK_WARPS;
  }
}


CUDASubGraphOp::CUDASubGraphOp(const Graph* graph): SubGraphOp(graph) {
  host_table_ = new HostHashTable(
      std::max(graph_->GetColCount(), graph_->GetRowCount()), 1);
  col_mask_ = static_cast<int32_t*>(
      CUDAAlloc(sizeof(int32_t) * (graph_->GetColCount() + 1)));
}

CUDASubGraphOp::~CUDASubGraphOp() {
  delete host_table_;
  host_table_ = nullptr;
  CUDADelete((void*)col_mask_);
}

SubGraph CUDASubGraphOp::NodeSubGraph(const torch::Tensor& srcs,
    bool with_edge) {
  if (srcs.numel() > host_table_->Capacity()) {
    delete host_table_;
    host_table_ = new HostHashTable(srcs.numel(), 1);
  }
  auto stream = ::at::cuda::getDefaultCUDAStream();
  ::at::cuda::CUDAStreamGuard guard(stream);
  int64_t* nodes = static_cast<int64_t*>(
      CUDAAlloc(sizeof(int64_t) * srcs.numel(), stream));
  int32_t nodes_size;
  InitNode(stream, srcs, nodes, &nodes_size);
  int64_t* nbrs_num = static_cast<int64_t*>(
      CUDAAlloc(sizeof(int64_t) * nodes_size, stream));
  cudaMemsetAsync((void*)nbrs_num, 0, sizeof(int64_t) * nodes_size, stream);
  int64_t* nbrs_offset = static_cast<int64_t*>(
      CUDAAlloc(sizeof(int64_t) * nodes_size, stream));
  cudaMemsetAsync((void*)nbrs_offset, 0, sizeof(int64_t) * nodes_size, stream);
  // nbrs offset of each row in original graph_.
  int64_t* sub_indptr = static_cast<int64_t*>(
      CUDAAlloc(sizeof(int64_t) * (nodes_size + 1), stream));
  cudaMemsetAsync((void*)sub_indptr, 0, sizeof(int64_t) * (nodes_size + 1), stream);

  // get subgraph indptr by input order of nodes.
  CSRSliceRows(stream, nodes, nodes_size, sub_indptr);
  GetNbrsNumAndColMask(stream, nodes, sub_indptr, nodes_size, nbrs_num);
  CUDAAllocator allocator(stream);
  const auto policy = thrust::cuda::par(allocator).on(stream);
  thrust::exclusive_scan(policy, nbrs_num, nbrs_num+nodes_size, nbrs_offset);
  auto edge_size = thrust::reduce(policy, nbrs_num, nbrs_num+nodes_size);
  torch::Tensor out_nodes = torch::empty(nodes_size, srcs.options());
  torch::Tensor rows = torch::empty(edge_size, srcs.options());
  torch::Tensor cols = torch::empty(edge_size, srcs.options());
  torch::Tensor eids = torch::empty(edge_size, srcs.options());
  cudaMemcpyAsync((void*)out_nodes.data_ptr<int64_t>(), (void*)nodes,
      sizeof(int64_t) * nodes_size, cudaMemcpyDeviceToDevice, stream);

  const auto row_ptr = graph_->GetRowPtr();
  const auto col_idx = graph_->GetColIdx();
  const auto row_count = graph_->GetRowCount();
  int64_t* edge_ids = nullptr;
  if (with_edge) {
    edge_ids = const_cast<int64_t*>(graph_->GetEdgeId());
  }
  auto device_table = host_table_->DeviceHandle();
  const dim3 grid((nodes_size + TILE_SIZE - 1) / TILE_SIZE);
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  GetColIndexKernel<<<grid, block, 0, stream>>>(
      device_table, nodes, sub_indptr, col_mask_, nodes_size,
      row_ptr, col_idx, edge_ids, row_count, with_edge,
      cols.data_ptr<int64_t>(), eids.data_ptr<int64_t>());
  CUDACheckError();
  GetRowIndexKernel<<<grid, block, 0, stream>>>(
      nodes, nbrs_num, nbrs_offset, nodes_size, rows.data_ptr<int64_t>());
  CUDACheckError();

  CUDADelete((void*)sub_indptr);
  CUDADelete((void*)nbrs_offset);
  CUDADelete((void*)nbrs_num);
  CUDADelete((void*)nodes);
  auto subgraph = SubGraph(out_nodes, rows, cols);
  if (with_edge) subgraph.eids = eids;
  return subgraph;
}

void CUDASubGraphOp::Reset() {
  host_table_->Clear();
}

void CUDASubGraphOp::InitNode(cudaStream_t stream,
    const torch::Tensor& srcs, int64_t* nodes, int32_t* nodes_size) {
  Reset();
  const auto src_size = srcs.numel();
  const auto src_ptr = srcs.data_ptr<int64_t>();
  host_table_->InsertDeviceHashTable(
    stream, src_ptr, src_size, nodes, nodes_size);
  CUDACheckError();
}

void CUDASubGraphOp::CSRSliceRows(
    cudaStream_t stream,
    const int64_t* rows,
    int32_t rows_size,
    int64_t* sub_indptr) {
  graph_->LookupDegree(rows, rows_size, sub_indptr);
  CUDAAllocator allocator(stream);
  const auto policy = thrust::cuda::par(allocator).on(stream);
  thrust::exclusive_scan(policy,
      sub_indptr, sub_indptr+rows_size+1, sub_indptr);
  CUDACheckError();
}

void CUDASubGraphOp::GetNbrsNumAndColMask(cudaStream_t stream,
                                          const int64_t* nodes,
                                          const int64_t* sub_indptr,
                                          int32_t nodes_size,
                                          int64_t* out_nbrs_num) {
  const auto row_ptr = graph_->GetRowPtr();
  const auto col_idx = graph_->GetColIdx();
  const auto row_count = graph_->GetRowCount();
  const auto col_count = graph_->GetColCount();
  auto device_table = host_table_->DeviceHandle();
  const dim3 grid((nodes_size + TILE_SIZE - 1) / TILE_SIZE);
  const dim3 block(WARP_SIZE, BLOCK_WARPS);
  cudaMemsetAsync((void*)col_mask_, 0, sizeof(int32_t) * (col_count+ 1));
  GetNbrsNumKernel<<<grid, block, 0, stream>>>(
      device_table, nodes, sub_indptr, nodes_size,
      row_ptr, col_idx, row_count, out_nbrs_num, col_mask_);
  CUDAAllocator allocator(stream);
  const auto policy = thrust::cuda::par(allocator).on(stream);
  // col prefix
  thrust::exclusive_scan(
    policy, col_mask_, col_mask_ + col_count + 1, col_mask_, 0);
  CUDACheckError();
}

} // namespace graphlearn_torch