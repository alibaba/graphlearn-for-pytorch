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

#include "graphlearn_torch/csrc/cpu/subgraph_op.h"

namespace graphlearn_torch {

SubGraph CPUSubGraphOp::NodeSubGraph(const torch::Tensor& srcs,
                                     bool with_edge) {
  std::vector<int64_t> out_nodes;
  InitNode(srcs, out_nodes);
  torch::Tensor nodes = torch::empty(out_nodes.size(), srcs.options());
  std::copy(out_nodes.begin(), out_nodes.end(), nodes.data_ptr<int64_t>());
  std::vector<int64_t> out_rows;
  std::vector<int64_t> out_cols;
  std::vector<int64_t> out_eids;
  Induce(out_nodes, with_edge, out_rows, out_cols, out_eids);

  auto edge_size = out_rows.size();
  torch::Tensor rows = torch::empty(edge_size, srcs.options());
  torch::Tensor cols = torch::empty(edge_size, srcs.options());
  std::copy(out_rows.begin(), out_rows.end(), rows.data_ptr<int64_t>());
  std::copy(out_cols.begin(), out_cols.end(), cols.data_ptr<int64_t>());
  auto subgraph = SubGraph(nodes, rows, cols);
  if (with_edge) {
    torch::Tensor eids = torch::empty(edge_size, srcs.options());
    std::copy(out_eids.begin(), out_eids.end(), eids.data_ptr<int64_t>());
    subgraph.eids = eids;
  }

  return subgraph;
}

void CPUSubGraphOp::InitNode(const torch::Tensor& srcs,
                             std::vector<int64_t>& out_nodes) {
  Reset();
  const auto src_size = srcs.size(0);
  const auto src_ptr = srcs.data_ptr<int64_t>();
  int32_t nodes_size = 0;
  out_nodes.reserve(src_size);
  for (int32_t i = 0; i < src_size; ++i) {
    if (glob2local_.insert(std::make_pair(src_ptr[i], nodes_size)).second) {
      out_nodes.push_back(src_ptr[i]);
      ++nodes_size;
    }
  }
}

void CPUSubGraphOp::Induce(const std::vector<int64_t>& nodes,
                           bool with_edge,
                           std::vector<int64_t>& out_rows,
                           std::vector<int64_t>& out_cols,
                           std::vector<int64_t>& out_eids) {
  const auto indptr = graph_->GetRowPtr();
  const auto indices = graph_->GetColIdx();
  const auto edge_ids = graph_->GetEdgeId();
  const auto row_count = graph_->GetRowCount();
  const auto node_size = nodes.size();
  out_rows.reserve(node_size * node_size);
  out_cols.reserve(node_size * node_size);
  if (with_edge) out_eids.reserve(node_size * node_size);

  for (int32_t i = 0; i < node_size; ++i) {
    auto old_row = nodes[i];
    if (old_row < row_count) {
      for (int32_t j = indptr[old_row]; j < indptr[old_row+1]; ++j) {
        auto old_col = indices[j];
        auto col_iter = glob2local_.find(old_col);
        if (col_iter != glob2local_.end()) {
          out_rows.push_back(i);
          out_cols.push_back(col_iter->second);
          if (with_edge) out_eids.push_back(edge_ids[j]);
        }
      }
    }
  }
}

}  // namespace graphlearn_torch