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

#ifndef GRAPHLEARN_TORCH_CUDA_INDUCER_CUH_
#define GRAPHLEARN_TORCH_CUDA_INDUCER_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <unordered_map>
#include <vector>

#include "graphlearn_torch/include/inducer_base.h"
#include "graphlearn_torch/include/types.h"

namespace graphlearn_torch {


class HostHashTable;

class CUDAInducer : public Inducer {
public:
  explicit CUDAInducer(int32_t num_nodes);
  virtual ~CUDAInducer();
  CUDAInducer(const CUDAInducer& other) = delete;
  CUDAInducer& operator=(const CUDAInducer& other) = delete;
  CUDAInducer(CUDAInducer&&) = default;
  CUDAInducer& operator=(CUDAInducer&&) = default;

  virtual torch::Tensor InitNode(const torch::Tensor& seed);
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  InduceNext(const torch::Tensor& srcs,
             const torch::Tensor& nbrs,
             const torch::Tensor& nbrs_num);
  virtual void Reset();

private:
  HostHashTable* host_table_;
};


class CUDAHeteroInducer : public HeteroInducer{
public:
  explicit CUDAHeteroInducer(std::unordered_map<std::string, int32_t> num_nodes);
  virtual ~CUDAHeteroInducer();
  CUDAHeteroInducer(const CUDAHeteroInducer& other) = delete;
  CUDAHeteroInducer& operator=(const CUDAHeteroInducer& other) = delete;
  CUDAHeteroInducer(CUDAHeteroInducer&&) = default;
  CUDAHeteroInducer& operator=(CUDAHeteroInducer&&) = default;

  virtual TensorMap InitNode(const TensorMap& seed);
  virtual HeteroCOO InduceNext(const HeteroNbr& nbrs);
  virtual void Reset();

private:
  void GroupNodesByType(
      const std::string& type,
      const int64_t* nodes,
      const int64_t nodes_size,
      std::unordered_map<std::string, std::vector<Array<int64_t>>>& nodes_dict);

  void InsertHostHashTable(
      const cudaStream_t stream,
      const std::unordered_map<std::string, std::vector<Array<int64_t>>>& input_ptr_dict,
      std::unordered_map<std::string, int64_t*>& out_nodes_dict,
      std::unordered_map<std::string, int64_t>& out_nodes_num_dict);

  void BuildNodesDict(
      const std::unordered_map<std::string, int64_t*>& out_nodes_dict,
      const std::unordered_map<std::string, int64_t>& out_nodes_num_dict,
      const torch::TensorOptions& option,
      std::unordered_map<std::string, torch::Tensor>& nodes_dict);

  void BuildEdgeIndexDict(
      const cudaStream_t stream,
      const HeteroNbr& nbrs,
      const std::unordered_map<EdgeType, int64_t*, EdgeTypeHash>& row_prefix_dict,
      const torch::TensorOptions& option,
      std::unordered_map<EdgeType, torch::Tensor, EdgeTypeHash>& rows_dict,
      std::unordered_map<EdgeType, torch::Tensor, EdgeTypeHash>& cols_dict);

private:
  std::unordered_map<std::string, HostHashTable*> host_table_;
};

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_CUDA_INDUCER_CUH_
