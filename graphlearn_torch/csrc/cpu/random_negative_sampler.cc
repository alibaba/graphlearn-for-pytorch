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
#include "graphlearn_torch/csrc/cpu/random_negative_sampler.h"

#include <algorithm>


namespace graphlearn_torch {

std::tuple<torch::Tensor, torch::Tensor> CPURandomNegativeSampler::Sample(
    int32_t req_num, int32_t trials_num, bool padding) {
  const int64_t* row_ptr = graph_->GetRowPtr();
  const int64_t* col_idx = graph_->GetColIdx();
  int64_t row_num = graph_->GetRowCount();
  int64_t col_num = graph_->GetColCount();
  uint32_t seed = RandomSeedManager::getInstance().getSeed();
  thread_local static std::mt19937 engine(seed);
  std::uniform_int_distribution<int64_t> row_dist(0, row_num - 1);
  std::uniform_int_distribution<int64_t> col_dist(0, col_num - 1);
  int64_t row_data[req_num];
  int64_t col_data[req_num];
  int32_t out_prefix[req_num];
  std::fill(out_prefix, out_prefix + req_num, 0);

  at::parallel_for(0, req_num, 1, [&](int32_t start, int32_t end) {
    for(int32_t i = start; i < end; ++i) {
      for(int32_t j = 0; j < trials_num; ++j) {
        int64_t r = row_dist(engine);
        int64_t c = col_dist(engine);
        if (!EdgeInCSR(row_ptr, col_idx, r, c)) {
          row_data[i] = r;
          col_data[i] = c;
          out_prefix[i] = 1;
          break;
        }
      }
    }
  });
  // sort sampled results.
  int32_t cursor = 0;
  for (int32_t i = 0; i < req_num; ++i) {
    if (out_prefix[i] == 1) {
      row_data[cursor] = row_data[i];
      col_data[cursor] = col_data[i];
      ++cursor;
    }
  }
  int32_t sampled_num = std::accumulate(out_prefix, out_prefix + req_num, 0);
  while ((sampled_num < req_num) && padding) { // non-strict negative sampling.
    row_data[sampled_num] = row_dist(engine);
    col_data[sampled_num] = col_dist(engine);
    ++sampled_num;
  }

  torch::Tensor rows = torch::empty(sampled_num, torch::kInt64);
  torch::Tensor cols = torch::empty(sampled_num, torch::kInt64);
  std::copy(row_data, row_data + sampled_num, rows.data_ptr<int64_t>());
  std::copy(col_data, col_data + sampled_num, cols.data_ptr<int64_t>());
  return std::make_tuple(rows, cols);
}

bool CPURandomNegativeSampler::EdgeInCSR(const int64_t* row_ptr,
                                         const int64_t* col_idx,
                                         int64_t r,
                                         int64_t c) {
  const int64_t* start = col_idx + row_ptr[r];
  const int64_t* end = col_idx + row_ptr[r + 1];
  return std::binary_search(start, end, c);
}

} // namespace graphlearn_torch
