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

#ifndef GRAPHLEARN_TORCH_INCLUDE_STITCH_SAMPLE_RESULTS_H_
#define GRAPHLEARN_TORCH_INCLUDE_STITCH_SAMPLE_RESULTS_H_

#include <torch/extension.h>

namespace graphlearn_torch {

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
CPUStitchSampleResults(const torch::Tensor& ids,
                       const std::vector<torch::Tensor>& idx_list,
                       const std::vector<torch::Tensor>& nbrs_list,
                       const std::vector<torch::Tensor>& nbrs_num_list,
                       const std::vector<torch::Tensor>& eids_list);

std::tuple<torch::Tensor, torch::Tensor, torch::optional<torch::Tensor>>
CUDAStitchSampleResults(const torch::Tensor& ids,
                        const std::vector<torch::Tensor>& idx_list,
                        const std::vector<torch::Tensor>& nbrs_list,
                        const std::vector<torch::Tensor>& nbrs_num_list,
                        const std::vector<torch::Tensor>& eids_list);

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_INCLUDE_STITCH_SAMPLE_RESULTS_H_
