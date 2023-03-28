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

#ifndef GRAPHLEARN_TORCH_CPU_INDUCER_H_
#define GRAPHLEARN_TORCH_CPU_INDUCER_H_

#include <torch/extension.h>
#include <unordered_map>
#include <vector>

#include "graphlearn_torch/include/inducer_base.h"
#include "graphlearn_torch/include/types.h"

namespace graphlearn_torch {

class CPUInducer : public Inducer {
public:
  explicit CPUInducer(int32_t num_nodes);
  virtual ~CPUInducer() {}
  CPUInducer(const CPUInducer& other) = delete;
  CPUInducer& operator=(const CPUInducer& other) = delete;
  CPUInducer(CPUInducer&&) = default;
  CPUInducer& operator=(CPUInducer&&) = default;

  virtual torch::Tensor InitNode(const torch::Tensor& seed);
  virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  InduceNext(const torch::Tensor& srcs,
             const torch::Tensor& nbrs,
             const torch::Tensor& nbrs_num);
  virtual void Reset();

private:
  int32_t    nodes_size_;
  IntHashMap glob2local_;
};


class CPUHeteroInducer : public HeteroInducer{
public:
  explicit CPUHeteroInducer(std::unordered_map<std::string, int32_t> num_nodes);
  virtual ~CPUHeteroInducer() {}
  CPUHeteroInducer(const CPUHeteroInducer& other) = delete;
  CPUHeteroInducer& operator=(const CPUHeteroInducer& other) = delete;
  CPUHeteroInducer(CPUHeteroInducer&&) = default;
  CPUHeteroInducer& operator=(CPUHeteroInducer&&) = default;

  virtual TensorMap InitNode(const TensorMap& seed);
  virtual HeteroCOO InduceNext(const HeteroNbr& nbrs);
  virtual void Reset();

private:
  void InsertGlob2Local(const HeteroNbr& nbrs,
      std::unordered_map<std::string, std::vector<int64_t>>& out_nodes);

private:
  std::unordered_map<std::string, int32_t>     nodes_size_;
  std::unordered_map<std::string, IntHashMap>  glob2local_;
};

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_CPU_INDUCER_H_
