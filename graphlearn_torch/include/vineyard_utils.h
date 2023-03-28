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

#ifdef WITH_VINEYARD

#include <torch/extension.h>

#include "graphlearn_torch/include/graph.h"
#include "graphlearn_torch/include/common.cuh"


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ToCSR(
  const std::string& ipc_socket, const std::string& object_id_str,
  int v_label, int e_label,
  bool has_eid);


torch::Tensor LoadVertexFeatures(
  const std::string& ipc_socket, const std::string& object_id_str,
  int v_label, std::vector<std::string>& vcols, graphlearn_torch::DataType dtype
);

torch::Tensor LoadEdgeFeatures(
  const std::string& ipc_socket, const std::string& object_id_str,
  int e_label, std::vector<std::string>& ecols, graphlearn_torch::DataType dtype
);

#endif // WITH_VINEYARD
