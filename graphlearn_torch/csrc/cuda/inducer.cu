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

#include "graphlearn_torch/csrc/cuda/inducer.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "graphlearn_torch/include/common.cuh"
#include "graphlearn_torch/include/hash_table.cuh"

namespace graphlearn_torch {

constexpr int32_t BLOCK_SIZE = 256;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t BLOCK_WARPS = 4;
// The number of rows covered by each threadblock.
constexpr int32_t TILE_SIZE = BLOCK_WARPS;

__global__ void ReIndexColKernel(DeviceHashTable table,
                                 const int64_t* input_cols,
                                 const int32_t input_size,
                                 int64_t* out_cols) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t stride_x = gridDim.x * blockDim.x;
  while (tid < input_size) {
    out_cols[tid] = table.Lookup(input_cols[tid])->value;
    tid += stride_x;
  }
}

__global__ void ReIndexRowKernel(DeviceHashTable table,
                                 const int64_t* input_rows,
                                 int64_t* nbrs_num,
                                 int64_t* input_prefix,
                                 const int32_t input_size,
                                 int64_t* out_rows) {
  // each block takes charge of TILE_SIZE rows reindexing.
  int32_t out_row = blockIdx.x * TILE_SIZE + threadIdx.y;
  const int32_t last_row = min((blockIdx.x + 1) * TILE_SIZE, input_size);
  while (out_row < last_row) {
    int64_t row = input_rows[out_row];
    auto row_idx = table.Lookup(row)->value;
    auto out_start = input_prefix[out_row];
    // each warp takes charge of one-row reindexing.
    for (int32_t idx = threadIdx.x; idx < nbrs_num[out_row]; idx += WARP_SIZE) {
      out_rows[out_start + idx] = row_idx;
    }
    out_row += BLOCK_WARPS;
  }
}

CUDAInducer::CUDAInducer(int32_t num_nodes) : Inducer() {
  host_table_ = new HostHashTable(num_nodes, 1);
}

CUDAInducer::~CUDAInducer() {
  delete host_table_;
  host_table_ = nullptr;
}

torch::Tensor CUDAInducer::InitNode(const torch::Tensor& seed) {
  auto stream = ::at::cuda::getDefaultCUDAStream();
  ::at::cuda::CUDAStreamGuard guard(stream);
  Reset();
  const auto seed_size = seed.numel();
  const auto seed_ptr = seed.data_ptr<int64_t>();
  int64_t* out_nodes = static_cast<int64_t*>(
      CUDAAlloc(sizeof(int64_t) * seed_size, stream));
  int32_t out_nodes_num = 0;

  host_table_->InsertDeviceHashTable(
    stream, seed_ptr, seed_size, out_nodes, &out_nodes_num);
  CUDACheckError();
  torch::Tensor nodes = torch::empty(out_nodes_num, seed.options());
  cudaMemcpy((void*)nodes.data_ptr<int64_t>(), (void*)out_nodes,
             sizeof(int64_t) * out_nodes_num, cudaMemcpyDeviceToDevice);
  CUDADelete((void*)out_nodes);
  return nodes;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CUDAInducer::InduceNext(const torch::Tensor& srcs,
                        const torch::Tensor& nbrs,
                        const torch::Tensor& nbrs_num) {
  auto stream = ::at::cuda::getDefaultCUDAStream();

  CUDAAllocator allocator(stream);
  const auto policy = thrust::cuda::par(allocator).on(stream);
  const auto edge_size = nbrs.numel();
  const auto src_size = srcs.numel();
  const auto rows_ptr = srcs.data_ptr<int64_t>();
  const auto cols_ptr = nbrs.data_ptr<int64_t>();
  const auto nbrs_num_ptr = nbrs_num.data_ptr<int64_t>();

  int64_t* row_prefix = static_cast<int64_t*>(
      CUDAAlloc(sizeof(int64_t) * src_size, stream));
  int64_t* out_nodes = static_cast<int64_t*>(
      CUDAAlloc(sizeof(int64_t) * edge_size, stream));
  int32_t out_nodes_num = 0;
  thrust::exclusive_scan(
    policy, nbrs_num_ptr, nbrs_num_ptr + src_size, row_prefix);
  host_table_->InsertDeviceHashTable(
    stream, cols_ptr, edge_size, out_nodes, &out_nodes_num);
  CUDACheckError();

  torch::Tensor nodes = torch::empty(out_nodes_num, srcs.options());
  torch::Tensor row_idx = torch::empty(edge_size, srcs.options());
  torch::Tensor col_idx = torch::empty(edge_size, srcs.options());
  cudaMemcpy((void*)nodes.data_ptr<int64_t>(), (void*)out_nodes,
             sizeof(int64_t) * out_nodes_num, cudaMemcpyDeviceToDevice);

  auto device_table = host_table_->DeviceHandle();
  const dim3 grid1((edge_size + TILE_SIZE - 1) / TILE_SIZE);
  const dim3 block1(BLOCK_SIZE);
  ReIndexColKernel<<<grid1, block1, 0, stream>>>(
    device_table, cols_ptr, edge_size, col_idx.data_ptr<int64_t>());
  CUDACheckError();

  const dim3 grid2((src_size + TILE_SIZE - 1) / TILE_SIZE);
  const dim3 block2(WARP_SIZE, BLOCK_WARPS);
  ReIndexRowKernel<<<grid2, block2, 0, stream>>>(
    device_table, rows_ptr, nbrs_num_ptr, row_prefix, src_size,
    row_idx.data_ptr<int64_t>());
  CUDACheckError();

  CUDADelete((void*)row_prefix);
  CUDADelete((void*)out_nodes);
  return std::make_tuple(nodes, row_idx, col_idx);
}

void CUDAInducer::Reset() {
  host_table_->Clear();
}


// heterogeneous graph.
CUDAHeteroInducer::CUDAHeteroInducer(
    std::unordered_map<std::string, int32_t> num_nodes) : HeteroInducer() {
  for (auto i : num_nodes) {
    host_table_[i.first] = new HostHashTable(i.second, 1);
  }
}

CUDAHeteroInducer::~CUDAHeteroInducer() {
  for (auto& i : host_table_) {
    delete i.second;
  }
  CUDACheckError();
}

TensorMap CUDAHeteroInducer::InitNode(const TensorMap& seed) {
  auto stream = ::at::cuda::getDefaultCUDAStream();

  Reset();
  TensorMap out_nodes_dict;
  for (const auto& iter : seed) {
    const auto seed_size = iter.second.numel();
    const auto seed_ptr = iter.second.data_ptr<int64_t>();
    int64_t* out_nodes = static_cast<int64_t*>(
        CUDAAlloc( sizeof(int64_t) * seed_size, stream));
    int32_t out_nodes_num = 0;

    host_table_[iter.first]->InsertDeviceHashTable(
      stream, seed_ptr, seed_size, out_nodes, &out_nodes_num);
    CUDACheckError();
    torch::Tensor nodes = torch::empty(out_nodes_num, iter.second.options());
    cudaMemcpy((void*)nodes.data_ptr<int64_t>(), (void*)out_nodes,
               sizeof(int64_t) * out_nodes_num, cudaMemcpyDeviceToDevice);
    CUDADelete((void*)out_nodes);
    out_nodes_dict.emplace(iter.first, std::move(nodes));
  }
  return out_nodes_dict;
}

void CUDAHeteroInducer::GroupNodesByType(
    const std::string& type,
    const int64_t* nodes,
    const int64_t nodes_size,
    std::unordered_map<std::string, std::vector<Array<int64_t>>>& nodes_dict) {
  auto iter = nodes_dict.find(type);
  if (iter == nodes_dict.end()) {
    std::vector<Array<int64_t>> vec({Array<int64_t>(nodes, nodes_size)});
    nodes_dict.emplace(type, std::move(vec));
  } else {
    iter->second.push_back(Array<int64_t>(nodes, nodes_size));
  }
}

void CUDAHeteroInducer::InsertHostHashTable(
    const cudaStream_t stream,
    const std::unordered_map<std::string, std::vector<Array<int64_t>>>& input_ptr_dict,
    std::unordered_map<std::string, int64_t*>& out_nodes_dict,
    std::unordered_map<std::string, int64_t>& out_nodes_num_dict) {
  for (const auto& iter : input_ptr_dict) {
    int64_t input_nodes_num = 0;
    const auto& nodes_vec = iter.second;
    for (const auto& arr : nodes_vec) {
      input_nodes_num += arr.size;
    }
    auto& host_table = host_table_[iter.first];
    int64_t* input_nodes = static_cast<int64_t*>(
        CUDAAlloc(sizeof(int64_t) * input_nodes_num, stream));
    int64_t count = 0;
    for (const auto& arr : nodes_vec) {
      cudaMemcpy((void*)(input_nodes + count), (void*)(arr.data),
                 sizeof(int64_t) * arr.size, cudaMemcpyDeviceToDevice);
      count += arr.size;
    }

    int64_t* out_nodes = static_cast<int64_t*>(
        CUDAAlloc(sizeof(int64_t) * input_nodes_num, stream));
    int32_t out_nodes_num = 0;
    host_table->InsertDeviceHashTable(
      stream, input_nodes, input_nodes_num, out_nodes, &out_nodes_num);
    out_nodes_dict.emplace(iter.first, out_nodes);
    out_nodes_num_dict.emplace(iter.first, out_nodes_num);
    CUDADelete((void*)input_nodes);
  }
}

void CUDAHeteroInducer::BuildNodesDict(
    const std::unordered_map<std::string, int64_t*>& out_nodes_dict,
    const std::unordered_map<std::string, int64_t>& out_nodes_num_dict,
    const torch::TensorOptions& option,
    std::unordered_map<std::string, torch::Tensor>& nodes_dict) {
  for (auto& iter : out_nodes_dict) {
    torch::Tensor nodes = torch::empty(out_nodes_num_dict.at(iter.first), option);
    cudaMemcpy((void*)nodes.data_ptr<int64_t>(),
               (void*)(iter.second),
               sizeof(int64_t) * out_nodes_num_dict.at(iter.first),
               cudaMemcpyDeviceToDevice);
    nodes_dict.emplace(iter.first, std::move(nodes));
  }
}

void CUDAHeteroInducer::BuildEdgeIndexDict(
    const cudaStream_t stream,
    const HeteroNbr& nbrs,
    const std::unordered_map<EdgeType, int64_t*, EdgeTypeHash>& row_prefix_dict,
    const torch::TensorOptions& option,
    std::unordered_map<EdgeType, torch::Tensor, EdgeTypeHash>& rows_dict,
    std::unordered_map<EdgeType, torch::Tensor, EdgeTypeHash>& cols_dict) {
  for (const auto& iter : nbrs) {
    const auto& src_type = std::get<0>(iter.first);
    const auto& dst_type = std::get<2>(iter.first);
    const auto nbrs_num_ptr = std::get<2>(iter.second).data_ptr<int64_t>();
    const auto rows_ptr = std::get<0>(iter.second).data_ptr<int64_t>();
    const auto cols_ptr = std::get<1>(iter.second).data_ptr<int64_t>();
    const auto src_size = std::get<0>(iter.second).numel();
    const auto edge_size = std::get<1>(iter.second).numel();

    torch::Tensor row_idx = torch::empty(edge_size, option);
    torch::Tensor col_idx = torch::empty(edge_size, option);
    auto col_table = (host_table_.at(dst_type))->DeviceHandle();
    const dim3 grid1((edge_size + TILE_SIZE - 1) / TILE_SIZE);
    const dim3 block1(BLOCK_SIZE);
    ReIndexColKernel<<<grid1, block1, 0, stream>>>(
      col_table, cols_ptr, edge_size, col_idx.data_ptr<int64_t>());

    auto row_table = (host_table_.at(src_type))->DeviceHandle();
    const dim3 grid2((src_size + TILE_SIZE - 1) / TILE_SIZE);
    const dim3 block2(WARP_SIZE, BLOCK_WARPS);
    ReIndexRowKernel<<<grid2, block2, 0, stream>>>(
      row_table, rows_ptr, nbrs_num_ptr, row_prefix_dict.at(iter.first),
      src_size, row_idx.data_ptr<int64_t>());
    rows_dict.emplace(iter.first, std::move(row_idx));
    cols_dict.emplace(iter.first, std::move(col_idx));
  }
}

HeteroCOO CUDAHeteroInducer::InduceNext(const HeteroNbr& nbrs) {
  auto stream = ::at::cuda::getDefaultCUDAStream();

  CUDAAllocator allocator(stream);
  const auto policy = thrust::cuda::par(allocator).on(stream);
  std::unordered_map<std::string, std::vector<Array<int64_t>>> input_ptr_dict;
  std::unordered_map<std::string, int64_t*> out_nodes_dict;
  std::unordered_map<std::string, int64_t> out_nodes_num_dict;
  std::unordered_map<EdgeType, int64_t*, EdgeTypeHash> row_prefix_dict;

  for (const auto& iter : nbrs) {
    const auto& src_type = std::get<0>(iter.first);
    const auto& dst_type = std::get<2>(iter.first);
    const auto nbrs_num_ptr = std::get<2>(iter.second).data_ptr<int64_t>();
    const auto src_ptr = std::get<0>(iter.second).data_ptr<int64_t>();
    const auto dst_ptr = std::get<1>(iter.second).data_ptr<int64_t>();
    const auto src_size = std::get<0>(iter.second).numel();
    const auto dst_size = std::get<1>(iter.second).numel();

    int64_t* row_prefix = static_cast<int64_t*>(
        CUDAAlloc(sizeof(int64_t) * src_size, stream));
    thrust::exclusive_scan(
      policy, nbrs_num_ptr, nbrs_num_ptr + src_size, row_prefix);
    row_prefix_dict.insert(std::make_pair(iter.first, row_prefix));
    GroupNodesByType(dst_type, dst_ptr, dst_size, input_ptr_dict);
  }
  InsertHostHashTable(
    stream, input_ptr_dict, out_nodes_dict, out_nodes_num_dict);
  CUDACheckError();
  auto tensor_option = std::get<0>(nbrs.begin()->second).options();
  std::unordered_map<std::string, torch::Tensor> nodes_dict;
  std::unordered_map<EdgeType, torch::Tensor, EdgeTypeHash> rows_dict;
  std::unordered_map<EdgeType, torch::Tensor, EdgeTypeHash> cols_dict;
  BuildNodesDict(out_nodes_dict, out_nodes_num_dict, tensor_option, nodes_dict);
  CUDACheckError();
  BuildEdgeIndexDict(
    stream, nbrs, row_prefix_dict, tensor_option, rows_dict, cols_dict);
  CUDACheckError();
  for (auto& iter :out_nodes_dict) {
    CUDADelete((void*)(iter.second));
  }
  for (auto& iter :row_prefix_dict) {
    CUDADelete((void*)(iter.second));
  }
  CUDACheckError();
  return std::make_tuple(nodes_dict, rows_dict, cols_dict);
}

void CUDAHeteroInducer::Reset() {
  for (auto& i : host_table_) {
    i.second->Clear();
  }
}

} // namespace graphlearn_torch