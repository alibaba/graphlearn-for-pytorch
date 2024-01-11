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

#include <torch/extension.h>

#include <vineyard/graph/fragment/arrow_fragment.h>
#include <vineyard/graph/loader/arrow_fragment_loader.h>
#include <vineyard/graph/fragment/graph_schema.h>

namespace graphlearn_torch {
namespace vineyard_utils {
  
using GraphType = vineyard::ArrowFragment<vineyard::property_graph_types::OID_TYPE,
                                          vineyard::property_graph_types::VID_TYPE>;
using vertex_t = GraphType::vertex_t;
using vid_t = vineyard::property_graph_types::VID_TYPE;
using fid_t = int64_t;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ToCSR(
  const std::string& ipc_socket, const std::string& object_id_str,
  const std::string& v_label_name, const std::string& e_label_id_name,
  const std::string& edge_dir, bool has_eid);

torch::Tensor LoadVertexFeatures(
  const std::string& ipc_socket, const std::string& object_id_str,
  const std::string& v_label_name, std::vector<std::string>& vcols
);

torch::Tensor LoadEdgeFeatures(
  const std::string& ipc_socket, const std::string& object_id_str,
  const std::string& e_label_name, std::vector<std::string>& ecols
);

int64_t GetFragVertexOffset(
  const std::string& ipc_socket, const std::string& object_id_str,
  const std::string& v_label_name
);

uint32_t GetFragVertexNum(
  const std::string& ipc_socket, const std::string& object_id_str,
  const std::string& v_label_name
);


class VineyardFragHandle {
public:
  explicit VineyardFragHandle(
    const std::string& ipc_socket, const std::string& object_id_str);
  torch::Tensor GetFidFromGid(const std::vector<int64_t>& gids);
  torch::Tensor GetInnerVertices(const std::string& v_label_name);
private:
  std::shared_ptr<GraphType> frag_;
};

} // namespace vineyard_utils
}  // namespace graphlearn_torch
