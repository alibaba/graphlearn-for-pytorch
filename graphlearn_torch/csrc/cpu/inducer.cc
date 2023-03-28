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

#include "graphlearn_torch/csrc/cpu/inducer.h"


namespace graphlearn_torch {

CPUInducer::CPUInducer(int32_t num_nodes): Inducer(), nodes_size_(0) {
  glob2local_.reserve(num_nodes);
}

torch::Tensor CPUInducer::InitNode(const torch::Tensor& seed) {
  Reset();
  const auto seed_size = seed.size(0);
  const auto seed_ptr = seed.data_ptr<int64_t>();
  std::vector<int64_t> out_nodes;
  out_nodes.reserve(seed_size);
  for (int32_t i = 0; i < seed_size; ++i) {
    if (glob2local_.insert(std::make_pair(seed_ptr[i], nodes_size_)).second) {
      out_nodes.push_back(seed_ptr[i]);
      ++nodes_size_;
    }
  }
  torch::Tensor nodes = torch::empty(out_nodes.size(), seed.options());
  std::copy(out_nodes.begin(), out_nodes.end(), nodes.data_ptr<int64_t>());
  return nodes;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
CPUInducer::InduceNext(const torch::Tensor& srcs, const torch::Tensor& nbrs,
    const torch::Tensor& nbrs_num) {
  const auto src_size = srcs.size(0);
  const auto edge_size = nbrs.size(0);
  const auto src_ptr = srcs.data_ptr<int64_t>();
  const auto nbrs_ptr = nbrs.data_ptr<int64_t>();
  const auto nbrs_num_ptr = nbrs_num.data_ptr<int64_t>();

  std::vector<int64_t> out_nodes;
  out_nodes.reserve(edge_size);
  for (int32_t i = 0; i < edge_size; ++i) {
    if (glob2local_.insert(std::make_pair(nbrs_ptr[i], nodes_size_)).second) {
      out_nodes.push_back(nbrs_ptr[i]);
      ++nodes_size_;
    }
  }

  torch::Tensor rows = torch::empty(edge_size, srcs.options());
  torch::Tensor cols = torch::empty(edge_size, srcs.options());
  torch::Tensor nodes = torch::empty(out_nodes.size(), srcs.options());
  auto rows_ptr = rows.data_ptr<int64_t>();
  auto cols_ptr = cols.data_ptr<int64_t>();
  int32_t cnt = 0;
  for (int32_t i = 0; i < src_size; ++i) {
    for (int32_t j = 0; j < nbrs_num_ptr[i]; ++j) {
      rows_ptr[cnt] = glob2local_[src_ptr[i]];
      cols_ptr[cnt] = glob2local_[nbrs_ptr[cnt]];
      ++cnt;
    }
  }
  std::copy(out_nodes.begin(), out_nodes.end(), nodes.data_ptr<int64_t>());
  return std::make_tuple(nodes, rows, cols);
}

void CPUInducer::Reset() {
  glob2local_.clear();
  nodes_size_ = 0;
}


// heterogeneous graph
CPUHeteroInducer::CPUHeteroInducer(
    std::unordered_map<std::string, int32_t> num_nodes)
  : HeteroInducer() {
  for (auto i : num_nodes) {
    nodes_size_[i.first] = 0;
    (glob2local_[i.first]).reserve(i.second);
  }
}

TensorMap CPUHeteroInducer::InitNode(const TensorMap& seed) {
  Reset();
  TensorMap nodes_dict;
  for (const auto& iter :seed) {
    const auto seed_size = iter.second.size(0);
    const auto seed_ptr = iter.second.data_ptr<int64_t>();
    std::vector<int64_t> out_nodes;
    out_nodes.reserve(seed_size);
    int64_t n_id = nodes_size_[iter.first];
    for (int32_t i = 0; i < seed_size; ++i) {
      if (((glob2local_[iter.first]).insert({seed_ptr[i], n_id})).second) {
        out_nodes.push_back(seed_ptr[i]);
        ++n_id;
      }
    }
    nodes_size_[iter.first] = n_id;
    torch::Tensor nodes = torch::empty(out_nodes.size(), iter.second.options());
    std::copy(out_nodes.begin(), out_nodes.end(), nodes.data_ptr<int64_t>());
    nodes_dict.emplace(iter.first, std::move(nodes));
  }
  return nodes_dict;
}

void CPUHeteroInducer::InsertGlob2Local(
    const HeteroNbr& nbrs,
    std::unordered_map<std::string, std::vector<int64_t>>& out_nodes) {
  for (const auto& iter : nbrs) {
    const auto& dst_type = std::get<2>(iter.first);
    const auto& nbrs = std::get<1>(iter.second);
    const auto dst_size = nbrs.size(0);
    const auto dst_ptr = nbrs.data_ptr<int64_t>();
    auto& dst_glob2local = glob2local_[dst_type];
    auto& dst_out_nodes = out_nodes[dst_type];
    dst_out_nodes.reserve(dst_size);
    int64_t n_id = nodes_size_[dst_type];
    for (int32_t i = 0; i < dst_size; ++i) {
      if (dst_glob2local.insert(std::make_pair(dst_ptr[i], n_id)).second) {
        dst_out_nodes.push_back(dst_ptr[i]);
        ++n_id;
      }
    }
    nodes_size_[dst_type] = n_id;
  }
}

HeteroCOO CPUHeteroInducer::InduceNext(const HeteroNbr& nbrs) {
  std::unordered_map<std::string, std::vector<int64_t>> out_nodes;
  InsertGlob2Local(nbrs, out_nodes);
  auto tensor_option = std::get<0>(nbrs.begin()->second).options();
  TensorEdgeMap rows_dict;
  TensorEdgeMap cols_dict;
  TensorMap nodes_dict;

  for (auto& iter : out_nodes) {
    const auto& node_type = iter.first;
    auto& out_nodes = iter.second;
    torch::Tensor nodes = torch::empty(out_nodes.size(), tensor_option);
    std::copy(out_nodes.begin(), out_nodes.end(), nodes.data_ptr<int64_t>());
    nodes_dict.emplace(node_type, std::move(nodes));
  }

  for (const auto& iter : nbrs) {
    const auto src_ptr = std::get<0>(iter.second).data_ptr<int64_t>();
    const auto nbrs_ptr = std::get<1>(iter.second).data_ptr<int64_t>();
    const auto nbrs_num_ptr = std::get<2>(iter.second).data_ptr<int64_t>();
    const auto src_size = std::get<0>(iter.second).size(0);
    const auto edge_size = std::get<1>(iter.second).size(0);
    const auto& src_type = std::get<0>(iter.first);
    const auto& dst_type = std::get<2>(iter.first);
    const auto& src_glob2local = glob2local_[src_type];
    const auto& dst_glob2local = glob2local_[dst_type];

    torch::Tensor rows = torch::empty(edge_size, tensor_option);
    torch::Tensor cols = torch::empty(edge_size, tensor_option);
    auto rows_ptr = rows.data_ptr<int64_t>();
    auto cols_ptr = cols.data_ptr<int64_t>();
    int32_t cnt = 0;
    for (int32_t i = 0; i < src_size; ++i) {
      for (int32_t j = 0; j < nbrs_num_ptr[i]; ++j) {
        rows_ptr[cnt] = src_glob2local.at(src_ptr[i]);
        cols_ptr[cnt] = dst_glob2local.at(nbrs_ptr[cnt]);
        cnt++;
      }
    }
    rows_dict.emplace(iter.first, std::move(rows));
    cols_dict.emplace(iter.first, std::move(cols));
  }
  return std::make_tuple(nodes_dict, rows_dict, cols_dict);
}

void CPUHeteroInducer::Reset() {
  for (auto& i : glob2local_) {
    i.second.clear();
  }
  for (auto& i : nodes_size_) {
    i.second = 0;
  }
}

} // namespace graphlearn_torch