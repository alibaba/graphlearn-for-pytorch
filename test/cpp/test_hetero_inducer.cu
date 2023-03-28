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

#include <iostream>
#include <cuda.h>
#include <thrust/device_vector.h>

#include "gtest/gtest.h"

#include "graphlearn_torch/csrc/cpu/inducer.h"
#include "graphlearn_torch/csrc/cuda/inducer.cuh"
#include "graphlearn_torch/include/types.h"

using namespace graphlearn_torch;

class InduceHeteroTest : public ::testing::Test {
protected:
  void InitData(const torch::Device& device) {
    /* The example meta-path of neighbor sampling:
    a
    |  \
    b   c
    */
    // list 1.
    node_types.push_back("a");
    node_types.push_back("b");
    node_types.push_back("c");
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    seed.emplace("a", torch::tensor({0, 1, 2, 2, 3}, options));
    auto srcs_1 = torch::tensor({0, 1, 2, 2, 3}, options);
    auto nbrs_1 = torch::tensor({1, 2, 2, 3, 4, 5, 4, 5, 1}, options);
    auto nbrs_num_1 = torch::tensor({2, 2, 2, 2, 1}, options);
    hetero_nbrs_1.emplace(std::make_tuple("a", "a2b", "b"),
        std::make_tuple(srcs_1, nbrs_1, nbrs_num_1));

    auto srcs_2 = torch::tensor({2, 1, 3, 2, 3}, options);
    auto nbrs_2 = torch::tensor({3, 5, 2, 3, 4, 3, 1, 2, 1}, options);
    auto nbrs_num_2 = torch::tensor({1, 2, 2, 3, 1}, options);
    hetero_nbrs_1.emplace(std::make_tuple("a", "a2c", "c"),
        std::make_tuple(srcs_2, nbrs_2, nbrs_num_2));

    std::vector<int64_t> true_seed_a = {0, 1, 2, 3};
    std::vector<int64_t> true_b = {1, 2, 3, 4, 5};
    std::vector<int64_t> true_c = {3, 5, 2, 4, 1};
    std::vector<int64_t> true_a2b_row = {0, 0, 1, 1, 2, 2, 2, 2, 3};
    std::vector<int64_t> true_a2b_col = {0, 1, 1, 2, 3, 4, 3, 4, 0};
    std::vector<int64_t> true_a2c_row = {2, 1, 1, 3, 3, 2, 2, 2, 3};
    std::vector<int64_t> true_a2c_col = {0, 1, 2, 0, 3, 0, 4, 2, 4};
    true_seed =
    {
      {"a", true_seed_a}
    };
    true_nodes_dict1 =
    {
      {"b", true_b}, {"c", true_c}
    };
    true_rows_dict1 =
    {
      {std::make_tuple("a", "a2b", "b"), true_a2b_row},
      {std::make_tuple("a", "a2c", "c"), true_a2c_row}
    };
    true_cols_dict1 =
    {
      {std::make_tuple("a", "a2b", "b"), true_a2b_col},
      {std::make_tuple("a", "a2c", "c"), true_a2c_col}
    };
  }

protected:
  void CheckCOODict(
      const std::unordered_map<std::string, torch::Tensor>& nodes_dict,
      const std::unordered_map<EdgeType, torch::Tensor, EdgeTypeHash>& rows_dict,
      const std::unordered_map<EdgeType, torch::Tensor, EdgeTypeHash>& cols_dict,
      const std::unordered_map<std::string, std::vector<int64_t>>& true_nodes_dict,
      const std::unordered_map<EdgeType, std::vector<int64_t>, EdgeTypeHash>& true_rows_dict,
      const std::unordered_map<EdgeType, std::vector<int64_t>, EdgeTypeHash>& true_cols_dict) {
  for (const auto& iter : true_nodes_dict) {
    const auto& nodes = nodes_dict.at(iter.first);
    const auto& true_nodes = iter.second;
    auto cpu_nodes = nodes.to(device(torch::kCPU));
    auto nodes_ptr = cpu_nodes.data_ptr<int64_t>();
    EXPECT_EQ(nodes.size(0), true_nodes.size());
    for (int32_t i = 0; i < true_nodes.size(); ++i) {
      EXPECT_EQ(nodes_ptr[i], true_nodes[i]);
    }
  }

  for (const auto& iter : true_rows_dict) {
    const auto& rows = rows_dict.at(iter.first);
    const auto& true_rows = iter.second;
    auto cpu_rows = rows.to(device(torch::kCPU));
    auto rows_ptr = cpu_rows.data_ptr<int64_t>();
    EXPECT_EQ(rows.size(0), true_rows.size());
    for (int32_t i = 0; i < true_rows.size(); ++i) {
      EXPECT_EQ(rows_ptr[i], true_rows[i]);
    }
  }

  for (const auto& iter : true_cols_dict) {
    const auto& cols = cols_dict.at(iter.first);
    const auto& true_cols = iter.second;
    auto cpu_cols = cols.to(device(torch::kCPU));
    auto cols_ptr = cpu_cols.data_ptr<int64_t>();
    EXPECT_EQ(cols.size(0), true_cols.size());
    for (int32_t i = 0; i < true_cols.size(); ++i) {
      EXPECT_EQ(cols_ptr[i], true_cols[i]);
    }
  }
}

protected:
  std::vector<std::string> node_types;
  TensorMap seed;
  HeteroNbr hetero_nbrs_1;
  std::unordered_map<std::string, std::vector<int64_t>> true_seed;
  std::unordered_map<std::string, std::vector<int64_t>> true_nodes_dict1;
  std::unordered_map<EdgeType, std::vector<int64_t>, EdgeTypeHash> true_rows_dict1;
  std::unordered_map<EdgeType, std::vector<int64_t>, EdgeTypeHash> true_cols_dict1;
};


TEST_F(InduceHeteroTest, CPUInducer) {
  InitData(torch::Device(torch::kCPU));
  std::unordered_map<std::string, int32_t> num_nodes;
  for (auto i : node_types) {
    num_nodes[i] = 10;
  }
  CPUHeteroInducer inducer(num_nodes);
  // seeds.
  auto nodes_dict = inducer.InitNode(seed);
  for (const auto& iter : true_seed) {
    const auto& nodes = nodes_dict.at(iter.first);
    const auto& true_nodes = iter.second;
    auto cpu_nodes = nodes.to(device(torch::kCPU));
    auto nodes_ptr = cpu_nodes.data_ptr<int64_t>();
    EXPECT_EQ(nodes.size(0), true_nodes.size());
    for (int32_t i = 0; i < true_nodes.size(); ++i) {
      EXPECT_EQ(nodes_ptr[i], true_nodes[i]);
    }
  }

  // nbrs.
  auto res = inducer.InduceNext(hetero_nbrs_1);
  nodes_dict = std::get<0>(res);
  auto rows_dict = std::get<1>(res);
  auto cols_dict = std::get<2>(res);
  CheckCOODict(nodes_dict, rows_dict, cols_dict,
               true_nodes_dict1, true_rows_dict1, true_cols_dict1);
}

TEST_F(InduceHeteroTest, CUDAInducer) {
  InitData(torch::Device(torch::kCUDA, 0));
  std::unordered_map<std::string, int32_t> num_nodes;
  for (auto i : node_types) {
    num_nodes[i] = 10;
  }
  CUDAHeteroInducer inducer(num_nodes);
  // seeds.
  auto nodes_dict = inducer.InitNode(seed);
  for (const auto& iter : true_seed) {
    const auto& nodes = nodes_dict.at(iter.first);
    const auto& true_nodes = iter.second;
    auto cpu_nodes = nodes.to(device(torch::kCPU));
    auto nodes_ptr = cpu_nodes.data_ptr<int64_t>();
    EXPECT_EQ(nodes.size(0), true_nodes.size());
    for (int32_t i = 0; i < true_nodes.size(); ++i) {
      EXPECT_EQ(nodes_ptr[i], true_nodes[i]);
    }
  }
  // nbrs.
  auto res = inducer.InduceNext(hetero_nbrs_1);
  nodes_dict = std::get<0>(res);
  auto rows_dict = std::get<1>(res);
  auto cols_dict = std::get<2>(res);
  for (auto iter : nodes_dict) {
    const auto& nodes = iter.second;
    EXPECT_EQ(nodes.device().type(), torch::kCUDA);
    EXPECT_EQ(nodes.device().index(), 0);
  }
  for (auto iter : rows_dict) {
    const auto& rows = iter.second;
    EXPECT_EQ(rows.device().type(), torch::kCUDA);
    EXPECT_EQ(rows.device().index(), 0);
  }
  for (auto iter : cols_dict) {
    const auto& cols = iter.second;
    EXPECT_EQ(cols.device().type(), torch::kCUDA);
    EXPECT_EQ(cols.device().index(), 0);
  }
  CheckCOODict(nodes_dict, rows_dict, cols_dict,
               true_nodes_dict1, true_rows_dict1, true_cols_dict1);
}