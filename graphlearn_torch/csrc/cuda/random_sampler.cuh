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

#ifndef GRAPHLEARN_TORCH_CUDA_RANDOM_SAMPLER_CUH_
#define GRAPHLEARN_TORCH_CUDA_RANDOM_SAMPLER_CUH_

#include "graphlearn_torch/include/sampler.h"

namespace graphlearn_torch {

class CUDARandomSampler : public Sampler {
public:
  CUDARandomSampler(const Graph* graph) : Sampler(graph) {}
  ~CUDARandomSampler() {}

  std::tuple<torch::Tensor, torch::Tensor>
  Sample(const torch::Tensor& nodes, int32_t req_num) override;

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  SampleWithEdge(const torch::Tensor& nodes, int32_t req_num) override;

  std::tuple<torch::Tensor, torch::Tensor> WeightedSample(
    const torch::Tensor& nodes, int32_t req_num) override {
    std::cerr << "Not supported yet!" << std::endl;
  }

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  WeightedSampleWithEdge(const torch::Tensor& nodes, int32_t req_num) override {
    std::cerr << "Not supported yet!" << std::endl;
  }

  void CalNbrProb(int k, const torch::Tensor& last_prob,
                  const torch::Tensor& nbr_last_prob,
                  const Graph* nbr_graph_, torch::Tensor cur_prob);
};

} // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_CUDA_RANDOM_SAMPLER_CUH_