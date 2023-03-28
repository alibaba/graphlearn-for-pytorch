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

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "graphlearn_torch/include/common.cuh"

namespace graphlearn_torch {

__global__ void FillPartialNbrsKernel(const int64_t* partital_idxs,
                                      const int64_t* partital_nbrs,
                                      const int64_t* partital_nbrs_nums,
                                      const int64_t* partital_nbrs_offsets,
                                      const int64_t* partital_eids,
                                      const int64_t* nbrs_offsets,
                                      int64_t partital_count,
                                      bool with_edge,
                                      int64_t* nbrs,
                                      int64_t* eids) {
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < partital_count) {
    auto cur_id_idx = partital_idxs[tid];
    auto cur_nbr_num = partital_nbrs_nums[tid];
    if (cur_nbr_num > 0) {
      auto local_offset = partital_nbrs_offsets[tid] - cur_nbr_num;
      auto global_offset = nbrs_offsets[cur_id_idx] - cur_nbr_num;
      for (int64_t i = 0; i < cur_nbr_num; ++i) {
        nbrs[global_offset + i] = partital_nbrs[local_offset + i];
      }
      if (with_edge) {
        for (int64_t i = 0; i < cur_nbr_num; ++i) {
          eids[global_offset + i] = partital_eids[local_offset + i];
        }
      }
    }
  }
}

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
CUDAStitchSampleResults(const torch::Tensor& ids,
                        const std::vector<torch::Tensor>& idx_list,
                        const std::vector<torch::Tensor>& nbrs_list,
                        const std::vector<torch::Tensor>& nbrs_num_list,
                        const std::vector<torch::Tensor>& eids_list) {
  auto stream = ::at::cuda::getDefaultCUDAStream();
  ::at::cuda::CUDAStreamGuard guard(stream);

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
  auto nbrs_offsets = torch::cumsum(nbrs_num, 0);
  auto nbrs_count = nbrs_offsets[ids_count - 1].item<int64_t>();
  auto nbrs = torch::zeros(nbrs_count, options);
  auto eids = with_edge ?
    torch::zeros(nbrs_count, options) : torch::empty(0, options);

  for (int64_t i = 0; i < partitions; ++i) {
    auto partital_nbrs_offsets = torch::cumsum(nbrs_num_list[i], 0);
    int64_t partital_count = idx_list[i].size(0);
    const dim3 block(512);
    const dim3 grid((partital_count + block.x - 1) / block.x);
    FillPartialNbrsKernel<<<grid, block, 0, stream>>>(
      idx_list[i].data_ptr<int64_t>(),
      nbrs_list[i].data_ptr<int64_t>(),
      nbrs_num_list[i].data_ptr<int64_t>(),
      partital_nbrs_offsets.data_ptr<int64_t>(),
      with_edge ? eids_list[i].data_ptr<int64_t>() : nullptr,
      nbrs_offsets.data_ptr<int64_t>(),
      partital_count,
      with_edge,
      nbrs.data_ptr<int64_t>(),
      with_edge ? eids.data_ptr<int64_t>() : nullptr);
    CUDACheckError();
  }

  return std::make_tuple(
    nbrs,
    nbrs_num,
    with_edge ? torch::optional<torch::Tensor>{eids} :
                torch::optional<torch::Tensor>{torch::nullopt});
}

}  // namespace graphlearn_torch
