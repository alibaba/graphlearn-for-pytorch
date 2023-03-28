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

#include "gtest/gtest.h"

#include "graphlearn_torch/csrc/cpu/subgraph_op.h"
#include "graphlearn_torch/csrc/cuda/subgraph_op.cuh"

using namespace graphlearn_torch;

class SubGraphOpTest : public ::testing::Test {
protected:
  void SetUp() override {
    /** graph adj matrix
    * 1 1 0 0 0 0
    * 0 1 0 1 0 0
    * 0 0 1 1 1 0
    * 0 0 0 0 0 1
    */
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    indptr = torch::tensor({0, 2, 4, 7, 8}, options);
    indices = torch::tensor({0, 1, 1, 3, 2, 3, 4, 5}, options);
    edge_ids = torch::tensor({0, 1, 2, 3, 4, 5, 6, 7}, options);
  }

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

protected:
  torch::Tensor indptr;
  torch::Tensor indices;
  torch::Tensor edge_ids;
};

TEST_F(SubGraphOpTest, CPU) {
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto input = torch::tensor({0, 2, 1, 2, 4}, options);
  std::vector<int64_t>true_nodes = {0, 2, 1, 4};
  std::vector<int64_t>true_rows = {0, 0, 1, 1, 2};
  std::vector<int64_t>true_cols = {0, 2, 1, 3, 2};
  std::vector<int64_t>true_eids = {0, 1, 4, 6, 2};

  Graph* graph = new Graph();
  graph->InitCPUGraphFromCSR(indptr, indices, edge_ids);
  CPUSubGraphOp subgraph(graph);

  auto res = subgraph.NodeSubGraph(input, false);
  CheckCOO(res.nodes, res.rows, res.cols,
           true_nodes, true_rows, true_cols);

  res = subgraph.NodeSubGraph(input, true);
  CheckCOO(res.nodes, res.rows, res.cols,
           true_nodes, true_rows, true_cols);
  auto eids_ptr = res.eids.data_ptr<int64_t>();
  for (int32_t i = 0; i < true_eids.size(); ++i) {
    EXPECT_EQ(eids_ptr[i], true_eids[i]);
  }
  delete graph;
}

TEST_F(SubGraphOpTest, CUDA) {
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, 0);
  auto input = torch::tensor({0, 2, 1, 2, 4}, options);
  std::vector<int64_t>true_nodes = {0, 2, 1, 4};
  std::vector<int64_t>true_rows = {0, 0, 1, 1, 2};
  std::vector<int64_t>true_cols = {0, 2, 1, 3, 2};
  std::vector<int64_t>true_eids = {0, 1, 4, 6, 2};

  Graph* graph = new Graph();
  graph->InitCUDAGraphFromCSR(indptr, indices, 0, GraphMode::DMA, edge_ids);
  CUDASubGraphOp subgraph(graph);

  auto res = subgraph.NodeSubGraph(input, false);
  EXPECT_EQ(res.nodes.device().type(), torch::kCUDA);
  EXPECT_EQ(res.rows.device().type(), torch::kCUDA);
  EXPECT_EQ(res.cols.device().type(), torch::kCUDA);
  CheckCOO(res.nodes, res.rows, res.cols,
           true_nodes, true_rows, true_cols);

  res = subgraph.NodeSubGraph(input, true);
  EXPECT_EQ(res.nodes.device().type(), torch::kCUDA);
  EXPECT_EQ(res.rows.device().type(), torch::kCUDA);
  EXPECT_EQ(res.cols.device().type(), torch::kCUDA);
  EXPECT_EQ(res.eids.device().type(), torch::kCUDA);
  CheckCOO(res.nodes, res.rows, res.cols,
           true_nodes, true_rows, true_cols);
  auto cpu_eids = res.eids.to(device(torch::kCPU));
  auto eids_ptr = cpu_eids.data_ptr<int64_t>();
  for (int32_t i = 0; i < true_eids.size(); ++i) {
    EXPECT_EQ(eids_ptr[i], true_eids[i]);
  }
  delete graph;
}