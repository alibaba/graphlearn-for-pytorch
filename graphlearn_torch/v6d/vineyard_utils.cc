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

#include "graphlearn_torch/v6d/vineyard_utils.h"

#include <vineyard/client/client.h>

#include "glog/logging.h"

namespace graphlearn_torch {
namespace vineyard_utils {

vineyard::Client vyclient;

template<typename T>
void customDeleter(void* ptr) {
  delete[] static_cast<T*>(ptr);
}

std::shared_ptr<GraphType> GetGraphFromVineyard(
  const std::string& ipc_socket, const std::string& object_id_str) {
  // Get the graph via vineyard fragment id from vineyard server.
  VINEYARD_CHECK_OK(vyclient.Connect(ipc_socket));

  vineyard::ObjectID object_id = vineyard::ObjectIDFromString(object_id_str);
  bool exists = false;
  if (!vyclient.Exists(object_id, exists).ok() || !exists) {
    object_id = std::strtoll(object_id_str.c_str(), nullptr, 10);
  }
  if (!vyclient.Exists(object_id, exists).ok() || !exists) {
    throw std::runtime_error("ERROR: Object not exists!\n");
  }

  // check if the object is a fragment group
  std::shared_ptr<vineyard::ArrowFragmentGroup> fragment_group;
  if (vyclient.GetObject(object_id, fragment_group).ok()) {
    for (auto const &kv: fragment_group->FragmentLocations()) {
      if (kv.second == vyclient.instance_id()) {
        object_id = fragment_group->Fragments().at(kv.first);
        break;
      }
    }
  }

  // try get the fragment
  std::shared_ptr<GraphType> frag;
  VINEYARD_CHECK_OK(vyclient.GetObject(object_id, frag));
  return frag;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ToCSR(
  const std::string& ipc_socket, const std::string& object_id_str,
  const std::string& v_label_name, const std::string& e_label_name,
  const std::string& edge_dir, bool has_eid) {

  if (edge_dir != "in" && edge_dir != "out") {
    throw std::runtime_error("Invalid edge_dir value. edge_dir must be 'in' or 'out'.");
  }

  auto vineyard_graph = GetGraphFromVineyard(ipc_socket, object_id_str);

  auto v_label_id =  vineyard_graph->schema().GetVertexLabelId(v_label_name);
  if (v_label_id < 0) {
    throw std::runtime_error("v_label_name not exist");
  }
  auto e_label_id =  vineyard_graph->schema().GetEdgeLabelId(e_label_name);
  if (e_label_id < 0) {
    throw std::runtime_error("e_label_name not exist");
  }

  int64_t* offsets;
  int64_t offset_len;

  if (edge_dir == "out") {
    offsets = const_cast<int64_t*>(
      vineyard_graph->GetOutgoingOffsetArray(v_label_id, e_label_id));
    offset_len = vineyard_graph->GetOutgoingOffsetLength(v_label_id, e_label_id);
  } else {
    offsets = const_cast<int64_t*>(
      vineyard_graph->GetIncomingOffsetArray(v_label_id, e_label_id));
    offset_len = vineyard_graph->GetIncomingOffsetLength(v_label_id, e_label_id);
  }

  auto iv = vineyard_graph->InnerVertices(v_label_id);
  int64_t indice_len = 0;

  for (auto v: iv) {
    if (edge_dir == "out") {
      auto oe = vineyard_graph->GetOutgoingRawAdjList(v, e_label_id);
      indice_len += oe.Size();
    } else {
      auto oe = vineyard_graph->GetIncomingRawAdjList(v, e_label_id);
      indice_len += oe.Size();
    }
  }

  int64_t* cols = new int64_t[indice_len];
  int64_t* eids = new int64_t[indice_len];

  int64_t i = 0;

  for (auto v : iv) {
    if (edge_dir == "out") {
      auto oe = vineyard_graph->GetOutgoingAdjList(v, e_label_id);
      for (auto& e : oe) {
        cols[i] = vineyard_graph->Vertex2Gid(e.get_neighbor());
        if (has_eid) {
          eids[i++] = e.edge_id();
        } else {
          ++i;
        }
      }
    } else {
      auto oe = vineyard_graph->GetIncomingAdjList(v, e_label_id);
      for (auto& e : oe) {
        cols[i] = vineyard_graph->Vertex2Gid(e.get_neighbor());
        if (has_eid) {
          eids[i++] = e.edge_id();
        } else {
          ++i;
        }
      }
    }
  }

  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  torch::Tensor indptr = torch::from_blob(offsets, offset_len, options);
  torch::Tensor indices = torch::from_blob(cols, indice_len, customDeleter<int64_t>, options);
  torch::Tensor edge_ids = torch::from_blob(eids, indice_len, customDeleter<int64_t>, options);
  return {indptr, indices, edge_ids};
}


torch::Tensor ArrowArray2Tensor(
  std::shared_ptr<arrow::Array> fscol, uint64_t col_num) {
  if (fscol->type()->Equals(arrow::int32())) {
    auto mcol = std::dynamic_pointer_cast<arrow::Int32Array>(fscol);
    auto options = torch::TensorOptions().dtype(torch::kI32).device(torch::kCPU);
    return torch::from_blob(const_cast<int32_t*>(mcol->raw_values()),
      {static_cast<int64_t>(mcol->length() / col_num), static_cast<int64_t>(col_num)}, options);
  } else if (fscol->type()->Equals(arrow::int64())) {
    auto mcol = std::dynamic_pointer_cast<arrow::Int64Array>(fscol);
    auto options = torch::TensorOptions().dtype(torch::kI64).device(torch::kCPU);
    return torch::from_blob(const_cast<int64_t*>(mcol->raw_values()),
      {static_cast<int64_t>(mcol->length() / col_num), static_cast<int64_t>(col_num)}, options);
  } else if (fscol->type()->Equals(arrow::float32())) { //dtype: float
    auto mcol = std::dynamic_pointer_cast<arrow::FloatArray>(fscol);
    auto options = torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU);
    return torch::from_blob(const_cast<float*>(mcol->raw_values()),
      {static_cast<int64_t>(mcol->length() / col_num), static_cast<int64_t>(col_num)}, options);
  } else if (fscol->type()->Equals(arrow::float64())){ //dtype: double
    auto mcol = std::dynamic_pointer_cast<arrow::DoubleArray>(fscol);
    auto options = torch::TensorOptions().dtype(torch::kF64).device(torch::kCPU);
    return torch::from_blob(const_cast<double*>(mcol->raw_values()),
      {static_cast<int64_t>(mcol->length() / col_num), static_cast<int64_t>(col_num)}, options);
  } else {
    throw std::runtime_error("Unsupported column type: " + fscol->type()->ToString());
  }
}


torch::Tensor LoadVertexFeatures(
  const std::string& ipc_socket, const std::string& object_id_str,
  const std::string& v_label_name, std::vector<std::string>& vcols) {

  auto frag = GetGraphFromVineyard(ipc_socket, object_id_str);
  auto v_label_id =  frag->schema().GetVertexLabelId(v_label_name);

  if (v_label_id < 0) {
    throw std::runtime_error("v_label_name not exist");
  }
  std::shared_ptr<arrow::Array> fscol;
  torch::Tensor feat;

  // By default merge all cols when `vcols` is empty.
  if (vcols.size() == 0) {
    vcols = frag->vertex_data_table(v_label_id)->ColumnNames();
  }

  // Consolidate given columns
  if (vcols.size() >= 2) {
    try {
      auto vfrag_id =
        frag->ConsolidateVertexColumns(vyclient, v_label_id, vcols, "vmerged").value();
      auto vfrag =
        std::dynamic_pointer_cast<GraphType>(vyclient.GetObject(vfrag_id));
      fscol = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(
        vfrag->vertex_data_table(v_label_id)->GetColumnByName("vmerged")->chunk(0)
      )->values();
    } catch(...) {
      LOG(ERROR) << "Possibly different column types OR wrong column names.\n";
      throw std::runtime_error("ERROR: Unable to merge!");
    }
  } else if (vcols.size() == 1) {
    try {
      fscol = frag->vertex_data_table(v_label_id)->GetColumnByName(vcols[0])->chunk(0);
    } catch(...) {
      throw std::runtime_error("ERROR: Column name not exists!");
    }
  }
  feat = ArrowArray2Tensor(fscol, vcols.size());
  return feat;
}


torch::Tensor LoadEdgeFeatures(
  const std::string& ipc_socket, const std::string& object_id_str,
  const std::string& e_label_name, std::vector<std::string>& ecols) {
  
  auto frag = GetGraphFromVineyard(ipc_socket, object_id_str);
  auto e_label_id = frag->schema().GetEdgeLabelId(e_label_name);
  if (e_label_id < 0) {
      throw std::runtime_error("e_label_name not exist");
  }

  std::shared_ptr<arrow::Array> fscol;
  torch::Tensor feat;

  // By default merge all cols when `vcols` is empty.
  if (ecols.size() == 0) {
    ecols = frag->edge_data_table(e_label_id)->ColumnNames();
  }

  // Consolidate given columns
  if (ecols.size() >= 2) {
    try {
      auto efrag_id =
        frag->ConsolidateEdgeColumns(vyclient, e_label_id, ecols, "emerged").value();
      auto efrag =
        std::dynamic_pointer_cast<GraphType>(vyclient.GetObject(efrag_id));
      fscol = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(
        efrag->edge_data_table(e_label_id)->GetColumnByName("emerged")->chunk(0)
      )->values();
    } catch(...) {
      LOG(ERROR) << "Possibly different column types OR wrong column names.\n";
      throw std::runtime_error(
        "ERROR: Unable to merge!");
    }
  } else if (ecols.size() == 1) {
    try {
      fscol = frag->edge_data_table(e_label_id)->GetColumnByName(ecols[0])->chunk(0);
    } catch(...) {
      throw std::runtime_error("ERROR: Column name not exists!");
    }
  }
  feat = ArrowArray2Tensor(fscol, ecols.size());
  return feat;
}


int64_t GetFragVertexOffset(
  const std::string& ipc_socket, const std::string& object_id_str,
  const std::string& v_label_name) {

  auto vineyard_graph = GetGraphFromVineyard(ipc_socket, object_id_str);
  auto v_label_id =  vineyard_graph->schema().GetVertexLabelId(v_label_name);
  auto ivbase = vineyard_graph->InnerVertices(v_label_id).begin();
  return vineyard_graph->Vertex2Gid(*ivbase);
}


uint32_t GetFragVertexNum(
  const std::string& ipc_socket, const std::string& object_id_str,
  const std::string& v_label_name) {
  
  auto vineyard_graph = GetGraphFromVineyard(ipc_socket, object_id_str);
  auto v_label_id =  vineyard_graph->schema().GetVertexLabelId(v_label_name);
  return vineyard_graph->InnerVertices(v_label_id).size();
}


VineyardFragHandle::VineyardFragHandle(
  const std::string& ipc_socket, const std::string& object_id_str) {
    frag_ = GetGraphFromVineyard(ipc_socket, object_id_str);
}

torch::Tensor VineyardFragHandle::GetFidFromGid(const std::vector<int64_t>& gids) {
  int64_t num_gids = gids.size();
  int64_t* fids_ptr = new int64_t[num_gids];

  at::parallel_for(0, num_gids, 1, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      fids_ptr[i] = (int64_t)frag_->GetFragId(gids[i]);
    }
  });

  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  torch::Tensor fids = torch::from_blob(fids_ptr, {num_gids}, customDeleter<int64_t>, options);
  return fids;
}

torch::Tensor VineyardFragHandle::GetInnerVertices(const std::string& v_label_name) {
  auto v_label_id = frag_->schema().GetVertexLabelId(v_label_name);
  auto iv = frag_->InnerVertices(v_label_id);
  int64_t* iv_ = new int64_t[iv.size()];
  int64_t i = 0;
  for (auto& v : iv) {
    iv_[i++] = frag_->Vertex2Gid(v);
  }
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  torch::Tensor vertices = torch::from_blob(iv_, {static_cast<int64_t>(iv.size())}, customDeleter<int64_t>, options);
  return vertices;
}

} // namespace vineyard_utils
}  // namespace graphlearn_torch
