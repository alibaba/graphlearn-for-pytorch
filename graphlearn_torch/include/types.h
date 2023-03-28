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

#ifndef GRAPHLEARN_TORCH_INCLUDE_TYPES_H_
#define GRAPHLEARN_TORCH_INCLUDE_TYPES_H_

#include <functional>
#include <string>
#include <torch/extension.h>
#include <unordered_map>
#include <vector>

namespace graphlearn_torch
{

using TensorMap = std::unordered_map<std::string, torch::Tensor>;
using IntHashMap = std::unordered_map<int64_t, int32_t>;
using EdgeType = std::tuple<std::string, std::string, std::string>;

struct EdgeTypeHash
{
  std::size_t operator()(const EdgeType& t) const noexcept
  {
    auto edge_type = std::get<0>(t) + "_" + std::get<1>(t) +\
      "_" + std::get<2>(t);
    return std::hash<std::string>{}(edge_type);
  }
};

using TensorEdgeMap = std::unordered_map<EdgeType, torch::Tensor, EdgeTypeHash>;
// node_dict, row_dict, col_dict.
using HeteroCOO = std::tuple<TensorMap, TensorEdgeMap, TensorEdgeMap>;
// dict of src, nbrs, nbrs_num.
using HeteroNbr = std::unordered_map<EdgeType,
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>, EdgeTypeHash>;

template<class T>
struct Array
{
  Array() {}
  Array(const T* ptr, const int64_t ptr_size) {
    data = ptr;
    size = ptr_size;
  }
  const T* data;
  int64_t size;
};

struct SubGraph {
  SubGraph() {}
  SubGraph(torch::Tensor nodes, torch::Tensor rows, torch::Tensor cols):
    nodes(nodes), rows(rows), cols(cols) {}

  torch::Tensor nodes;
  torch::Tensor rows;
  torch::Tensor cols;
  torch::Tensor eids;
};

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_INCLUDE_TYPES_H_