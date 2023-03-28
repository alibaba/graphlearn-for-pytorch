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

#include "graphlearn_torch/include/stitch_sample_results.h"

namespace graphlearn_torch {

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
CPUStitchSampleResults(const torch::Tensor& ids,
                       const std::vector<torch::Tensor>& idx_list,
                       const std::vector<torch::Tensor>& nbrs_list,
                       const std::vector<torch::Tensor>& nbrs_num_list,
                       const std::vector<torch::Tensor>& eids_list) {
  int64_t ids_count = ids.size(0);
  int64_t partitions = idx_list.size();
  bool with_edge = !eids_list.empty();
  auto options = torch::TensorOptions()
    .dtype(torch::kInt64)
    .device(ids.device());

  auto nbrs_num = torch::zeros(ids_count, options);
  for (int64_t i = 0; i < partitions; ++i) {
    nbrs_num = nbrs_num.index_copy(0, idx_list[i], nbrs_num_list[i]);
  }
  auto nbrs_offsets = torch::cumsum(nbrs_num, 0, torch::kInt64);
  const int64_t* nbrs_offset_data = nbrs_offsets.data_ptr<int64_t>();
  auto nbrs_count = nbrs_offset_data[ids_count - 1];
  auto nbrs = torch::zeros(nbrs_count, options);
  int64_t* nbrs_data = nbrs.data_ptr<int64_t>();
  auto eids = with_edge ?
    torch::zeros(nbrs_count, options) : torch::empty(0, options);
  int64_t* eids_data = with_edge ? eids.data_ptr<int64_t>() : nullptr;

  for (int64_t i = 0; i < partitions; ++i) {
    int64_t partital_count = idx_list[i].size(0);
    auto p_nbrs_offsets = torch::cumsum(nbrs_num_list[i], 0);
    const int64_t* p_nbrs_offsets_data = p_nbrs_offsets.data_ptr<int64_t>();
    const int64_t* p_idx_data = idx_list[i].data_ptr<int64_t>();
    const int64_t* p_nbrs_data = nbrs_list[i].data_ptr<int64_t>();
    const int64_t* p_nbrs_num_data = nbrs_num_list[i].data_ptr<int64_t>();
    const int64_t* p_eids_data = with_edge ?
      eids_list[i].data_ptr<int64_t>() : nullptr;
    at::parallel_for(0, partital_count, 1,
        [p_idx_data, p_nbrs_data, p_nbrs_num_data, p_eids_data,
         p_nbrs_offsets_data, nbrs_offset_data, with_edge, nbrs_data, eids_data]
        (int32_t start, int32_t end) {
      for (int32_t i = start; i < end; i++) {
        auto cur_id_idx = p_idx_data[i];
        auto cur_nbr_num = p_nbrs_num_data[i];
        if (cur_nbr_num > 0) {
          auto local_offset = p_nbrs_offsets_data[i] - cur_nbr_num;
          auto global_offset = nbrs_offset_data[cur_id_idx] - cur_nbr_num;
          memcpy(nbrs_data + global_offset,
                 p_nbrs_data + local_offset,
                 sizeof(int64_t) * cur_nbr_num);
          if (with_edge) {
            memcpy(eids_data + global_offset,
                   p_eids_data + local_offset,
                   sizeof(int64_t) * cur_nbr_num);
          }
        }
      }
    });
  }

  return std::make_tuple(
    nbrs,
    nbrs_num,
    with_edge ? torch::optional<torch::Tensor>{eids} :
                torch::optional<torch::Tensor>{torch::nullopt});
}

}  // namespace graphlearn_torch
