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

#ifndef GRAPHLEARN_TORCH_INCLUDE_NEGATIVE_SAMPLER_H_
#define GRAPHLEARN_TORCH_INCLUDE_NEGATIVE_SAMPLER_H_

#include "graphlearn_torch/include/graph.h"

namespace graphlearn_torch {

class NegativeSampler {
public:
  NegativeSampler(const Graph* graph) : graph_(graph) {}
  virtual ~NegativeSampler() {}

  virtual std::tuple<torch::Tensor, torch::Tensor> Sample(
    int32_t req_num, int32_t trials_num, bool padding=false) = 0;

protected:
  const Graph* graph_;
};

} // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_INCLUDE_NEGATIVE_SAMPLER_H_