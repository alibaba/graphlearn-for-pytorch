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

#ifndef GRAPHLEARN_TORCH_CPU_WEIGHTED_SAMPLER_H_
#define GRAPHLEARN_TORCH_CPU_WEIGHTED_SAMPLER_H_

#include "graphlearn_torch/include/sampler.h"


namespace graphlearn_torch {

class CPUWeightedSampler : public Sampler {
public:
  CPUWeightedSampler(const Graph* graph) : Sampler(graph) {}
  ~CPUWeightedSampler() {}

  std::tuple<torch::Tensor, torch::Tensor> Sample(
    const torch::Tensor& nodes, int32_t req_num) override;

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
  SampleWithEdge(const torch::Tensor& nodes, int32_t req_num) override;

private:
  void FillNbrsNum(const int64_t* nodes, const int32_t bs,
    const int32_t req_num,  const int64_t row_count,
    const int64_t* row_ptr, int64_t* out_nbr_num);
  
  void CSRRowWiseSample(const int64_t* nodes, const int64_t* nbrs_offset,
      const int32_t bs, const int32_t req_num, const int64_t row_count,
      const int64_t* row_ptr, const int64_t* col_idx, const float* prob,
      int64_t* out_nbrs);

  void CSRRowWiseSample(const int64_t* nodes, const int64_t* nbrs_offset,
      const int32_t bs, const int32_t req_num, const int64_t row_count,
      const int64_t* row_ptr, const int64_t* col_idx, const float* prob,
      const int64_t* edge_id, int64_t* out_nbrs, int64_t* out_eid);


  void WeightedSample(const int64_t* col_begin, const int64_t* col_end,
      const int32_t req_num, const float* prob_begin, const float* prob_end,
      int64_t* out_nbrs);

  void WeightedSample(const int64_t* col_begin, const int64_t* col_end,
      const int64_t* eid_begin, const int64_t* eid_end,
      const int32_t req_num, const float* prob_begin, const float* prob_end, 
      int64_t* out_nbrs, int64_t* out_eid);
};

} // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_CPU_WEIGHTED_SAMPLER_H_