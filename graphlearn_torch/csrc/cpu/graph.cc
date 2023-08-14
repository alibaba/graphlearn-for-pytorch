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

#include "graphlearn_torch/include/graph.h"
#include "graphlearn_torch/include/common.h"

namespace graphlearn_torch {

void Graph::InitCPUGraphFromCSR(
    const torch::Tensor& indptr,
    const torch::Tensor& indices,
    const torch::Tensor& edge_ids,
    const torch::Tensor& edge_weights) {
  CheckEq<int64_t>(indptr.dim(), 1);
  CheckEq<int64_t>(indices.dim(), 1);

  row_ptr_ = indptr.data_ptr<int64_t>();
  col_idx_ = indices.data_ptr<int64_t>();
  row_count_ = indptr.size(0) - 1;
  edge_count_ = indices.size(0);
  col_count_ = std::get<0>(at::_unique(indices)).size(0);

  if (edge_ids.numel()) {
    CheckEq<int64_t>(edge_ids.dim(), 1);
    CheckEq<int64_t>(edge_ids.numel(), indices.numel());
    edge_id_ = edge_ids.data_ptr<int64_t>();
  }

  if (edge_weights.numel()) {
    CheckEq<int64_t>(edge_weights.dim(), 1);
    CheckEq<int64_t>(edge_weights.numel(), indices.numel());
    edge_weight_ = edge_weights.data_ptr<float>();
  }
}

} // namespace graphlearn_torch
