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

#include "graphlearn_torch/include/common.h"
#include "graphlearn_torch/csrc/cpu/weighted_sampler.h"

#include <cstdint>
#include <cassert>


namespace graphlearn_torch {

std::tuple<torch::Tensor, torch::Tensor>
CPUWeightedSampler::Sample(const torch::Tensor& nodes, int32_t req_num) {
  if (req_num < 0) req_num = std::numeric_limits<int32_t>::max();
  const int64_t* nodes_ptr = nodes.data_ptr<int64_t>();
  int64_t bs = nodes.size(0);
  const auto row_ptr = graph_->GetRowPtr();
  const auto col_idx = graph_->GetColIdx();
  const auto row_count = graph_->GetRowCount();
  const auto edge_weights = graph_->GetEdgeWeight();

  torch::Tensor nbrs_num = torch::empty(bs, nodes.options());
  auto nbrs_num_ptr = nbrs_num.data_ptr<int64_t>();
  FillNbrsNum(nodes_ptr, bs, req_num, row_count, row_ptr, nbrs_num_ptr);
  int64_t nbrs_offset[bs + 1];
  nbrs_offset[0] = 0;
  for(int64_t i = 1; i <= bs; ++i) {
    nbrs_offset[i] = nbrs_offset[i - 1] + nbrs_num_ptr[i - 1];
  }

  torch::Tensor nbrs = torch::empty(nbrs_offset[bs], nodes.options());
  CSRRowWiseSample(nodes_ptr, nbrs_offset, bs, req_num, row_count,
    row_ptr, col_idx, edge_weights, nbrs.data_ptr<int64_t>());
  return std::make_tuple(nbrs, nbrs_num);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CPUWeightedSampler::SampleWithEdge(
  const torch::Tensor& nodes, int32_t req_num
) {
  if (req_num < 0) req_num = std::numeric_limits<int32_t>::max();
  const int64_t* nodes_ptr = nodes.data_ptr<int64_t>();
  int64_t bs = nodes.size(0);
  const auto row_ptr = graph_->GetRowPtr();
  const auto col_idx = graph_->GetColIdx();
  const auto edge_ids = graph_->GetEdgeId();
  const auto row_count = graph_->GetRowCount();
  const auto edge_weights = graph_->GetEdgeWeight();

  torch::Tensor nbrs_num = torch::empty(bs, nodes.options());
  auto nbrs_num_ptr = nbrs_num.data_ptr<int64_t>();
  FillNbrsNum(nodes_ptr, bs, req_num, row_count, row_ptr, nbrs_num_ptr);
  int64_t nbrs_offset[bs + 1];
  nbrs_offset[0] = 0;
  for(int64_t i = 1; i <= bs; ++i) {
    nbrs_offset[i] = nbrs_offset[i - 1] + nbrs_num_ptr[i - 1];
  }

  torch::Tensor nbrs = torch::empty(nbrs_offset[bs], nodes.options());
  torch::Tensor out_eid = torch::empty(nbrs_offset[bs], nodes.options());
  CSRRowWiseSample(nodes_ptr, nbrs_offset, bs, req_num, row_count,
                   row_ptr, col_idx, edge_weights, edge_ids,
                   nbrs.data_ptr<int64_t>(), out_eid.data_ptr<int64_t>());
  return std::make_tuple(nbrs, nbrs_num, out_eid);
}


void CPUWeightedSampler::FillNbrsNum(const int64_t* nodes,
                                     const int32_t bs,
                                     const int32_t req_num,
                                     const int64_t row_count,
                                     const int64_t* row_ptr,
                                     int64_t* out_nbr_num) {
  at::parallel_for(0, bs, 1, [&](int32_t start, int32_t end){
    for(int32_t i = start; i < end; i++){
      auto v = nodes[i];
      if (v < row_count) {
        out_nbr_num[i] = std::min(static_cast<int64_t>(req_num),
          row_ptr[v+1]-row_ptr[v]);
      } else {
        out_nbr_num[i] = 0;
      }
    }
  });
}

void CPUWeightedSampler::CSRRowWiseSample(
    const int64_t* nodes,
    const int64_t* nbrs_offset,
    const int32_t bs,
    const int32_t req_num,
    const int64_t row_count,
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const float* prob,
    int64_t* out_nbrs) {
  at::parallel_for(0, bs, 1, [&](int32_t start, int32_t end){
    for(int32_t i = start; i < end; ++i) {
      auto v = nodes[i];
      if (v < row_count) {
        WeightedSample(col_idx + row_ptr[v], col_idx + row_ptr[v+1], req_num,
          prob + row_ptr[v], prob + row_ptr[v+1], out_nbrs + nbrs_offset[i]);
      }
    }
  });
}

void CPUWeightedSampler::CSRRowWiseSample(
    const int64_t* nodes,
    const int64_t* nbrs_offset,
    const int32_t bs,
    const int32_t req_num,
    const int64_t row_count,
    const int64_t* row_ptr,
    const int64_t* col_idx,
    const float* prob,
    const int64_t* edge_ids,
    int64_t* out_nbrs,
    int64_t* out_eid) {
  at::parallel_for(0, bs, 1, [&](int32_t start, int32_t end){
    for(int32_t i = start; i < end; ++i) {
      auto v = nodes[i];
      if (v < row_count) {
        WeightedSample(col_idx + row_ptr[v], col_idx + row_ptr[v+1],
            edge_ids + row_ptr[v], edge_ids + row_ptr[v+1],
            req_num, prob + row_ptr[v], prob + row_ptr[v+1],
            out_nbrs + nbrs_offset[i], out_eid + nbrs_offset[i]);
      }
    }
  });
}


void CPUWeightedSampler::WeightedSample(const int64_t* col_begin,
                                        const int64_t* col_end,
                                        const int32_t req_num,
                                        const float* prob_begin,
                                        const float* prob_end,
                                        int64_t* out_nbrs) {
  // with replacement
  const auto cap = col_end - col_begin;
  if (req_num < cap) {
    uint32_t seed = RandomSeedManager::getInstance().getSeed();
    thread_local static std::mt19937 engine(seed);
    std::discrete_distribution<> dist(prob_begin, prob_end);
    for (int32_t i = 0; i < req_num; ++i) {
      out_nbrs[i] = col_begin[dist(engine)];
    }
  } else {
    std::copy(col_begin, col_end, out_nbrs);
  }
}

void CPUWeightedSampler::WeightedSample(const int64_t* col_begin,
                                        const int64_t* col_end,
                                        const int64_t* eid_begin,
                                        const int64_t* eid_end,
                                        const int32_t req_num,
                                        const float* prob_begin,
                                        const float* prob_end, 
                                        int64_t* out_nbrs,
                                        int64_t* out_eid) {
  // with replacement
  const auto cap = col_end - col_begin;
  if (req_num < cap) {
    uint32_t seed = RandomSeedManager::getInstance().getSeed();
    thread_local static std::mt19937 engine(seed);
    std::discrete_distribution<> dist(prob_begin, prob_end);
    for (int32_t i = 0; i < req_num; ++i) {
      auto idx = dist(engine);
      out_nbrs[i] = col_begin[idx];
      out_eid[i] = eid_begin[idx];
    }
  } else {
    std::copy(col_begin, col_end, out_nbrs);
    std::copy(eid_begin, eid_end, out_eid);
  }
}


}  // namespace graphlearn_torch