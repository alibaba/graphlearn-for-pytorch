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

#ifndef GRAPHLEARN_TORCH_CPU_RANDOM_NEGATIVE_SAMPLER_H_
#define GRAPHLEARN_TORCH_CPU_RANDOM_NEGATIVE_SAMPLER_H_

#include "graphlearn_torch/include/negative_sampler.h"

namespace graphlearn_torch {

class CPURandomNegativeSampler : public NegativeSampler {
public:
  CPURandomNegativeSampler(const Graph* graph) : NegativeSampler(graph) {}
  ~CPURandomNegativeSampler() {}
  virtual std::tuple<torch::Tensor, torch::Tensor> Sample(
    int32_t req_num, int32_t trials_num, bool padding=false);

private:
  bool EdgeInCSR(const int64_t* row_ptr,
                 const int64_t* col_idx,
                 int64_t r,
                 int64_t c);
};

} // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_CPU_RANDOM_NEGATIVE_SAMPLER_H_