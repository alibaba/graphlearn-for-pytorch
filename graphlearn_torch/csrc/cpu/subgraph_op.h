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

#ifndef GRAPHLEARN_TORCH_CSRC_CPU_SUBGRAPH_OP_H_
#define GRAPHLEARN_TORCH_CSRC_CPU_SUBGRAPH_OP_H_

#include "graphlearn_torch/include/subgraph_op_base.h"

#include <torch/extension.h>
#include <unordered_map>
#include <vector>

#include "graphlearn_torch/include/graph.h"
#include "graphlearn_torch/include/types.h"

namespace graphlearn_torch {

class CPUSubGraphOp : public SubGraphOp {
public:
  CPUSubGraphOp(const Graph* graph): SubGraphOp(graph) {}
  ~CPUSubGraphOp() {}
  SubGraph NodeSubGraph(const torch::Tensor& srcs, bool with_edge=false) override;

private:
  void Reset() { glob2local_.clear();}
  void InitNode(const torch::Tensor& srcs, std::vector<int64_t>& out_nodes);
  void Induce(const std::vector<int64_t>& nodes,
              bool with_edge,
              std::vector<int64_t>& out_rows,
              std::vector<int64_t>& out_cols,
              std::vector<int64_t>& out_eids);
private:
  IntHashMap   glob2local_;
};

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_CSRC_CPU_SUBGRAPH_OP_H_
