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

#include "graphlearn_torch/include/vineyard_utils.h"

#include <vineyard/client/client.h>
#include <vineyard/graph/fragment/arrow_fragment.h>
#include <vineyard/graph/fragment/graph_schema.h>
#include <vineyard/graph/loader/arrow_fragment_loader.h>

#include "glog/logging.h"

using GraphType = vineyard::ArrowFragment<vineyard::property_graph_types::OID_TYPE,
                                          vineyard::property_graph_types::VID_TYPE>;

vineyard::Client vyclient;

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ToCSR(
  const std::string& ipc_socket, const std::string& object_id_str,
  int v_label, int e_label,
  bool has_eid) {

  // Get the graph via vineyard fragment id from vineyard server.
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  VINEYARD_CHECK_OK(vyclient.Connect(ipc_socket));

  vineyard::ObjectID object_id = vineyard::ObjectIDFromString(object_id_str);
  bool exists = false;
  if (!vyclient.Exists(object_id, exists).ok() || !exists) {
    object_id = std::strtoll(object_id_str.c_str(), nullptr, 10);
  }
  if (!vyclient.Exists(object_id, exists).ok() || !exists) {
    throw std::runtime_error("ERROR: Object not exists!\n");
  }

  std::shared_ptr<GraphType> vineyard_graph =
    std::dynamic_pointer_cast<GraphType>(vyclient.GetObject(object_id));
  LOG(INFO) << "loaded graph ...";

  // Get indptr from vineyard
  int64_t* offsets = const_cast<int64_t*>(
    vineyard_graph->GetOutgoingOffsetArray(v_label, e_label));
  int64_t offset_len = vineyard_graph->GetOutgoingOffsetLength(v_label, e_label);

  // pre-calculate length of indices
  int64_t indice_len = 0;
  auto iv = vineyard_graph->InnerVertices(v_label);
  for (auto v: iv) {
    auto oe = vineyard_graph->GetOutgoingAdjList(v, e_label);
    indice_len += oe.Size();
  }

  int64_t* cols = new int64_t[indice_len];
  int64_t* eids = new int64_t[indice_len];

  // Get indices & edge id
  int64_t i = 0;
  if (has_eid) {
    for (auto v : iv) {
      auto oe = vineyard_graph->GetOutgoingAdjList(v, e_label);
      for (auto& e : oe) {
        // LOG(INFO) << "nb: " << e.get_neighbor().GetValue() << "\n";
        // LOG(INFO) << "eid: " << e.edge_id() << "\n";
        cols[i] = e.get_neighbor().GetValue();
        eids[i++] = e.edge_id();
      }
    }
  } else {
    for (auto v : iv) {
      auto oe = vineyard_graph->GetOutgoingRawAdjList(v, e_label);
      for (auto& e : oe) {
        // LOG(INFO) << e.get_neighbor().GetValue() << "\n";
        cols[i++] = e.get_neighbor().GetValue();
      }
    }
  }

  torch::Tensor indptr = torch::from_blob(offsets, offset_len, options);
  torch::Tensor indices = torch::from_blob(cols, indice_len, options);
  torch::Tensor edge_ids = torch::from_blob(eids, indice_len, options);
  return {indptr, indices, edge_ids};
}


torch::Tensor ArrowArray2Tensor(
  std::shared_ptr<arrow::Array> fscol, uint64_t col_num, graphlearn_torch::DataType dtype) {
  switch(dtype) {
    case graphlearn_torch::DataType::Int32: {
      auto mcol = std::dynamic_pointer_cast<arrow::Int32Array>(fscol);
      auto options = torch::TensorOptions().dtype(torch::kI32).device(torch::kCPU);
      return torch::from_blob(const_cast<int32_t*>(mcol->raw_values()),
        {mcol->length() / col_num, col_num}, options);
    }
    case graphlearn_torch::DataType::Int64: {
      auto mcol = std::dynamic_pointer_cast<arrow::Int64Array>(fscol);
      auto options = torch::TensorOptions().dtype(torch::kI64).device(torch::kCPU);
      return torch::from_blob(const_cast<int64_t*>(mcol->raw_values()),
        {mcol->length() / col_num, col_num}, options);
    }
    case graphlearn_torch::DataType::Float32: { //dtype: float
      auto mcol = std::dynamic_pointer_cast<arrow::FloatArray>(fscol);
      auto options = torch::TensorOptions().dtype(torch::kF32).device(torch::kCPU);
      return torch::from_blob(const_cast<float*>(mcol->raw_values()),
        {mcol->length() / col_num, col_num}, options);
    }
    case graphlearn_torch::DataType::Float64: { //dtype: double
      auto mcol = std::dynamic_pointer_cast<arrow::DoubleArray>(fscol);
      auto options = torch::TensorOptions().dtype(torch::kF64).device(torch::kCPU);
      return torch::from_blob(const_cast<double*>(mcol->raw_values()),
        {mcol->length() / col_num, col_num}, options);
    }
  }
}


torch::Tensor LoadVertexFeatures(
  const std::string& ipc_socket, const std::string& object_id_str,
  int v_label, std::vector<std::string>& vcols, graphlearn_torch::DataType dtype) {

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

  std::shared_ptr<GraphType> frag =
    std::dynamic_pointer_cast<GraphType>(vyclient.GetObject(object_id));

  std::shared_ptr<arrow::Array> fscol;
  torch::Tensor feat;

  // By default merge all cols when `vcols` is empty.
  if (vcols.size() == 0) {
    vcols = frag->vertex_data_table(v_label)->ColumnNames();
  }

  // Consolidate given columns
  if (vcols.size() >= 2) {
    try {
      auto vfrag_id =
        frag->ConsolidateVertexColumns(vyclient, 0, vcols, "vmerged").value();
      auto vfrag =
        std::dynamic_pointer_cast<GraphType>(vyclient.GetObject(vfrag_id));

      fscol = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(
        vfrag->vertex_data_table(v_label)->GetColumnByName("vmerged")->chunk(0)
      )->values();
    } catch(...) {
      LOG(ERROR) << "Possibly different column types OR wrong column names.\n";
      throw std::runtime_error(
        "ERROR: Unable to merge!");
    }
  } else if (vcols.size() == 1) {
    try {
      fscol = frag->vertex_data_table(v_label)->GetColumnByName(vcols[0])->chunk(0);
    } catch(...) {
      throw std::runtime_error("ERROR: Column name not exists!");
    }
  }

  LOG(INFO) << "loaded vertex features ...";

  feat = ArrowArray2Tensor(fscol, vcols.size(), dtype);
  return feat;

}


torch::Tensor LoadEdgeFeatures(
  const std::string& ipc_socket, const std::string& object_id_str,
  int e_label, std::vector<std::string>& ecols, graphlearn_torch::DataType dtype) {

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

  std::shared_ptr<GraphType> frag =
    std::dynamic_pointer_cast<GraphType>(vyclient.GetObject(object_id));

    std::shared_ptr<arrow::Array> fscol;
  torch::Tensor feat;

  // By default merge all cols when `vcols` is empty.
  if (ecols.size() == 0) {
    ecols = frag->edge_data_table(e_label)->ColumnNames();
  }

  // Consolidate given columns
  if (ecols.size() >= 2) {
    try {
      auto efrag_id =
        frag->ConsolidateEdgeColumns(vyclient, 0, ecols, "emerged").value();
      auto efrag =
        std::dynamic_pointer_cast<GraphType>(vyclient.GetObject(efrag_id));

      fscol = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(
        efrag->edge_data_table(e_label)->GetColumnByName("emerged")->chunk(0)
      )->values();
    } catch(...) {
      LOG(ERROR) << "Possibly different column types OR wrong column names.\n";
      throw std::runtime_error(
        "ERROR: Unable to merge!");
    }
  } else if (ecols.size() == 1) {
    try {
      fscol = frag->edge_data_table(e_label)->GetColumnByName(ecols[0])->chunk(0);
    } catch(...) {
      throw std::runtime_error("ERROR: Column name not exists!");
    }
  }

  LOG(INFO) << "loaded edge features ...";

  feat = ArrowArray2Tensor(fscol, ecols.size(), dtype);
  return feat;

}

#endif // WITH_VINEYARD
