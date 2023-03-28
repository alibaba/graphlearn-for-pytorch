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

#ifndef GRAPHLEARN_TORCH_INCLUDE_INDUCER_BASE_H_
#define GRAPHLEARN_TORCH_INCLUDE_INDUCER_BASE_H_

#include <torch/extension.h>
#include <vector>
#include <unordered_map>

#include "graphlearn_torch/include/types.h"

namespace graphlearn_torch {


class Inducer {
public:
  Inducer() {}
  virtual ~Inducer() {}

  // Init inducer with seed nodes and return de-duplicated seed nodes.
  virtual torch::Tensor InitNode(const torch::Tensor& seed) = 0;

  // Induce COO subgraph incrementally.
  // Returns: incremental unique nodes, rows, cols.
  // Note that the current srcs must be a subset of last output nodes.
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  InduceNext(const torch::Tensor& srcs,
             const torch::Tensor& nbrs,
             const torch::Tensor& nbrs_num) = 0;

  virtual void Reset() = 0;
};


class HeteroInducer {
public:
  HeteroInducer() {}
  virtual ~HeteroInducer() {}

  // Init inducer with seed nodes and return de-duplicated seed nodes.
  virtual TensorMap InitNode(const TensorMap& seed) = 0;

  // Induce HeteroCOO subgraph incrementally.
  // Returns: incremental unique nodes_dict, rows_dict, cols_dict.
  // Note that each type of the current srcs must be a subset of last
  // corresponding type output nodes.
  virtual HeteroCOO InduceNext(const HeteroNbr& nbrs) = 0;

  virtual void Reset() = 0;
};

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_INCLUDE_INDUCER_BASE_H_
