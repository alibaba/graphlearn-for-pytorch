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

using namespace graphlearn_torch;

class InduceGraphTest : public ::testing::Test {
protected:
  void CheckCOO(const torch::Tensor& nodes,
                const torch::Tensor& rows,
                const torch::Tensor& cols,
                const std::vector<int64_t>& true_nodes,
                const std::vector<int64_t>& true_rows,
                const std::vector<int64_t>& true_cols) {
    auto cpu_nodes = nodes.to(device(torch::kCPU));
    auto cpu_rows = rows.to(device(torch::kCPU));
    auto cpu_cols = cols.to(device(torch::kCPU));
    auto nodes_ptr = cpu_nodes.data_ptr<int64_t>();
    auto rows_ptr = cpu_rows.data_ptr<int64_t>();
    auto cols_ptr = cpu_cols.data_ptr<int64_t>();

    EXPECT_EQ(nodes.size(0), true_nodes.size());
    EXPECT_EQ(rows.size(0), true_rows.size());
    EXPECT_EQ(cols.size(0), true_cols.size());

    for (int32_t i = 0; i < true_nodes.size(); ++i) {
      EXPECT_EQ(nodes_ptr[i], true_nodes[i]);
    }
    for (int32_t i = 0; i < true_rows.size(); ++i) {
      EXPECT_EQ(rows_ptr[i], true_rows[i]);
    }
    for (int32_t i = 0; i < true_cols.size(); ++i) {
      EXPECT_EQ(cols_ptr[i], true_cols[i]);
    }
  }
};

TEST_F(InduceGraphTest, CPUInducer) {
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto srcs1 = torch::tensor({0, 1, 2, 2, 3}, options);
  auto nbrs1 = torch::tensor({1, 2, 2, 3, 4, 5, 4, 5, 1}, options);
  auto nbrs_num1 = torch::tensor({2, 2, 2, 2, 1}, options);
  auto srcs2 = torch::tensor({0, 1, 2, 3, 4, 5}, options);
  auto nbrs2 = torch::tensor({1, 7, 2, 3, 4, 6, 7}, options);
  auto nbrs_num2 = torch::tensor({1, 1, 1, 2, 1, 1}, options);
  std::vector<int64_t>true_seed = {0, 1, 2, 3};
  std::vector<int64_t>true_nodes1 = {4, 5};
  std::vector<int64_t>true_rows1 = {0, 0, 1, 1, 2, 2, 2, 2, 3};
  std::vector<int64_t>true_cols1 = {1, 2, 2, 3, 4, 5, 4, 5, 1};
  std::vector<int64_t>true_nodes2 = {7, 6};
  std::vector<int64_t>true_rows2 = {0, 1, 2, 3, 3, 4, 5};
  std::vector<int64_t>true_cols2 = {1, 6, 2, 3, 4, 7, 6};

  CPUInducer inducer(srcs1.size(0) + nbrs1.size(0) + srcs2.size(0) + nbrs2.size(0));
  // seed
  auto nodes = inducer.InitNode(srcs1);
  auto nodes_ptr = nodes.data_ptr<int64_t>();
  EXPECT_EQ(nodes.size(0), true_seed.size());
  for (int32_t i = 0; i < true_seed.size(); ++i) {
    EXPECT_EQ(nodes_ptr[i], true_seed[i]);
  }
  // nbrs.
  auto res = inducer.InduceNext(srcs1, nbrs1, nbrs_num1);
  nodes = std::get<0>(res);
  auto rows = std::get<1>(res);
  auto cols = std::get<2>(res);
  CheckCOO(nodes, rows, cols, true_nodes1, true_rows1, true_cols1);

  res = inducer.InduceNext(srcs2, nbrs2, nbrs_num2);
  nodes = std::get<0>(res);
  rows = std::get<1>(res);
  cols = std::get<2>(res);
  CheckCOO(nodes, rows, cols, true_nodes2, true_rows2, true_cols2);

  nodes = inducer.InitNode(srcs1);
  nodes_ptr = nodes.data_ptr<int64_t>();
  EXPECT_EQ(nodes.size(0), true_seed.size());
  for (int32_t i = 0; i < true_seed.size(); ++i) {
    EXPECT_EQ(nodes_ptr[i], true_seed[i]);
  }
  res = inducer.InduceNext(srcs1, nbrs1, nbrs_num1);
  nodes = std::get<0>(res);
  rows = std::get<1>(res);
  cols = std::get<2>(res);
  CheckCOO(nodes, rows, cols, true_nodes1, true_rows1, true_cols1);
}

TEST_F(InduceGraphTest, CUDAInducer) {
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, 0);
  auto srcs1 = torch::tensor({0, 1, 2, 2, 3}, options);
  auto nbrs1 = torch::tensor({1, 2, 2, 3, 4, 5, 4, 5, 1}, options);
  auto nbrs_num1 = torch::tensor({2, 2, 2, 2, 1}, options);
  auto srcs2 = torch::tensor({0, 1, 2, 3, 4, 5}, options);
  auto nbrs2 = torch::tensor({1, 7, 2, 3, 4, 6, 7}, options);
  auto nbrs_num2 = torch::tensor({1, 1, 1, 2, 1, 1}, options);
  std::vector<int64_t>true_seed = {0, 1, 2, 3};
  std::vector<int64_t>true_nodes1 = {4, 5};
  std::vector<int64_t>true_rows1 = {0, 0, 1, 1, 2, 2, 2, 2, 3};
  std::vector<int64_t>true_cols1 = {1, 2, 2, 3, 4, 5, 4, 5, 1};
  std::vector<int64_t>true_nodes2 = {7, 6};
  std::vector<int64_t>true_rows2 = {0, 1, 2, 3, 3, 4, 5};
  std::vector<int64_t>true_cols2 = {1, 6, 2, 3, 4, 7, 6};

  CUDAInducer inducer(srcs1.size(0) + nbrs1.size(0) + srcs2.size(0) + nbrs2.size(0));
  auto nodes = inducer.InitNode(srcs1);
  // de-duplicate seed.
  EXPECT_EQ(nodes.device().type(), torch::kCUDA);
  EXPECT_EQ(nodes.device().index(), 0);
  auto cpu_nodes = nodes.to(device(torch::kCPU));
  auto nodes_ptr = cpu_nodes.data_ptr<int64_t>();
  EXPECT_EQ(nodes.size(0), true_seed.size());
  for (int32_t i = 0; i < true_seed.size(); ++i) {
    EXPECT_EQ(nodes_ptr[i], true_seed[i]);
  }
  // nbrs.
  auto res = inducer.InduceNext(srcs1, nbrs1, nbrs_num1);
  nodes = std::get<0>(res);
  auto rows = std::get<1>(res);
  auto cols = std::get<2>(res);
  EXPECT_EQ(nodes.device().type(), torch::kCUDA);
  EXPECT_EQ(nodes.device().index(), 0);
  EXPECT_EQ(rows.device().type(), torch::kCUDA);
  EXPECT_EQ(rows.device().index(), 0);
  EXPECT_EQ(cols.device().type(), torch::kCUDA);
  EXPECT_EQ(cols.device().index(), 0);
  CheckCOO(nodes, rows, cols, true_nodes1, true_rows1, true_cols1);

  res = inducer.InduceNext(srcs2, nbrs2, nbrs_num2);
  nodes = std::get<0>(res);
  rows = std::get<1>(res);
  cols = std::get<2>(res);
  EXPECT_EQ(nodes.device().type(), torch::kCUDA);
  EXPECT_EQ(nodes.device().index(), 0);
  EXPECT_EQ(rows.device().type(), torch::kCUDA);
  EXPECT_EQ(rows.device().index(), 0);
  EXPECT_EQ(cols.device().type(), torch::kCUDA);
  EXPECT_EQ(cols.device().index(), 0);
  CheckCOO(nodes, rows, cols, true_nodes2, true_rows2, true_cols2);

  nodes = inducer.InitNode(srcs1);
  // de-duplicate seed.
  EXPECT_EQ(nodes.device().type(), torch::kCUDA);
  EXPECT_EQ(nodes.device().index(), 0);
  cpu_nodes = nodes.to(device(torch::kCPU));
  nodes_ptr = cpu_nodes.data_ptr<int64_t>();
  EXPECT_EQ(nodes.size(0), true_seed.size());
  for (int32_t i = 0; i < true_seed.size(); ++i) {
    EXPECT_EQ(nodes_ptr[i], true_seed[i]);
  }
  // nbrs
  res = inducer.InduceNext(srcs1, nbrs1, nbrs_num1);
  nodes = std::get<0>(res);
  rows = std::get<1>(res);
  cols = std::get<2>(res);
  EXPECT_EQ(nodes.device().type(), torch::kCUDA);
  EXPECT_EQ(nodes.device().index(), 0);
  EXPECT_EQ(rows.device().type(), torch::kCUDA);
  EXPECT_EQ(rows.device().index(), 0);
  EXPECT_EQ(cols.device().type(), torch::kCUDA);
  EXPECT_EQ(cols.device().index(), 0);
  CheckCOO(nodes, rows, cols, true_nodes1, true_rows1, true_cols1);
}