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

#include <cuda.h>
#include <iostream>
#include <unordered_set>

#include "gtest/gtest.h"

#include "graphlearn_torch/include/common.cuh"
#include "graphlearn_torch/include/graph.h"
#include "graphlearn_torch/csrc/cpu/random_negative_sampler.h"
#include "graphlearn_torch/csrc/cuda/random_negative_sampler.cuh"

using namespace graphlearn_torch;

class NegativeSamplerTest : public ::testing::Test {
protected:
  void SetUp() override {
    row_count = 5;
    edge_count = 11;
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    /*
    rows: 0 0 1 1 2 2 3 3 3 3 4
    cols: 0 1 1 3 2 3 0 1 2 3 4
    */
    indptr = torch::tensor({0, 2, 4, 6, 10, 11}, options);
    indices = torch::tensor({0, 1, 1, 3, 2, 3, 0, 1, 2, 3, 4}, options);
  }

protected:
  torch::Tensor indptr;
  torch::Tensor indices;
  int64_t row_count;
  int64_t edge_count;
};


void CheckNegativeEdges(const torch::Tensor& rows, const torch::Tensor& cols) {
  auto row_ptr = rows.to(device(torch::kCPU)).data_ptr<int64_t>();
  auto col_ptr = cols.to(device(torch::kCPU)).data_ptr<int64_t>();
  std::unordered_set<int64_t> nbr_set_0({0, 1});
  std::unordered_set<int64_t> nbr_set_1({1, 3});
  std::unordered_set<int64_t> nbr_set_2({2, 3});
  std::unordered_set<int64_t> nbr_set_3({0, 1, 2, 3});
  std::unordered_set<int64_t> nbr_set_4({4});

  for (int32_t i = 0; i < rows.size(0); ++i) {
    switch (row_ptr[i]) {
      case 0:
        EXPECT_TRUE(nbr_set_0.find(col_ptr[i]) == nbr_set_0.end());
        break;
      case 1:
        EXPECT_TRUE(nbr_set_1.find(col_ptr[i]) == nbr_set_1.end());
        break;
      case 2:
        EXPECT_TRUE(nbr_set_2.find(col_ptr[i]) == nbr_set_2.end());
        break;
      case 3:
        EXPECT_TRUE(nbr_set_3.find(col_ptr[i]) == nbr_set_3.end());
        break;
      case 4:
        EXPECT_TRUE(nbr_set_4.find(col_ptr[i]) == nbr_set_4.end());
        break;
    }
  }
  std::cout << "rows: " << rows << std::endl;
  std::cout << "cols: " << cols << std::endl;
}

TEST_F(NegativeSamplerTest, CPURandom) {
  Graph* graph = new Graph();
  int32_t req_num = 10;
  int32_t trials_num = 5;
  bool padding = false;
  graph->InitCPUGraphFromCSR(indptr, indices);
  auto sampler = CPURandomNegativeSampler(graph);
  auto res = sampler.Sample(req_num, trials_num, padding);
  const auto& rows = std::get<0>(res);
  const auto& cols = std::get<1>(res);
  EXPECT_LE(rows.size(0), req_num);
  EXPECT_LE(cols.size(0), req_num);
  CheckNegativeEdges(rows, cols);
  delete graph;
}

TEST_F(NegativeSamplerTest, CUDADMARandom) {
  Graph* graph = new Graph();
  int32_t req_num = 10;
  int32_t trials_num = 5;
  bool padding = false;
  graph->InitCUDAGraphFromCSR(indptr, indices, 0, GraphMode::DMA);
  auto sampler = CUDARandomNegativeSampler(graph);
  auto res = sampler.Sample(req_num, trials_num, padding);
  CUDACheckError();
  cudaDeviceSynchronize();
  const auto& rows = std::get<0>(res);
  const auto& cols = std::get<1>(res);
  EXPECT_EQ(rows.device().type(), torch::kCUDA);
  EXPECT_EQ(rows.device().index(), 0);
  EXPECT_EQ(cols.device().type(), torch::kCUDA);
  EXPECT_EQ(cols.device().index(), 0);
  EXPECT_LE(rows.size(0), req_num);
  EXPECT_LE(cols.size(0), req_num);
  CheckNegativeEdges(rows, cols);
  delete graph;
}

TEST_F(NegativeSamplerTest, CUDAPinRandom) {
  Graph* graph = new Graph();
  int32_t req_num = 10;
  int32_t trials_num = 5;
  bool padding = false;
  graph->InitCUDAGraphFromCSR(indptr, indices, 0, GraphMode::ZERO_COPY);
  auto sampler = CUDARandomNegativeSampler(graph);
  auto res = sampler.Sample(req_num, trials_num, padding);
  CUDACheckError();
  cudaDeviceSynchronize();
  const auto& rows = std::get<0>(res);
  const auto& cols = std::get<1>(res);
  EXPECT_EQ(rows.device().type(), torch::kCUDA);
  EXPECT_EQ(rows.device().index(), 0);
  EXPECT_EQ(cols.device().type(), torch::kCUDA);
  EXPECT_EQ(cols.device().index(), 0);
  EXPECT_LE(rows.size(0), req_num);
  EXPECT_LE(cols.size(0), req_num);
  CheckNegativeEdges(rows, cols);
  delete graph;
}

TEST_F(NegativeSamplerTest, CPURandomPadding) {
  Graph* graph = new Graph();
  int32_t req_num = 10;
  int32_t trials_num = 1;
  bool padding = true;
  graph->InitCPUGraphFromCSR(indptr, indices);
  auto sampler = CPURandomNegativeSampler(graph);
  auto res = sampler.Sample(req_num, trials_num, padding);
  const auto& rows = std::get<0>(res);
  const auto& cols = std::get<1>(res);
  EXPECT_EQ(rows.size(0), req_num);
  EXPECT_EQ(cols.size(0), req_num);
  std::cout << "rows: " << rows << std::endl;
  std::cout << "cols: " << cols << std::endl;
  delete graph;
}

TEST_F(NegativeSamplerTest, CUDADMARandomPadding) {
  Graph* graph = new Graph();
  int32_t req_num = 10;
  int32_t trials_num = 1;
  bool padding = true;
  graph->InitCUDAGraphFromCSR(indptr, indices, 0, GraphMode::ZERO_COPY);
  auto sampler = CUDARandomNegativeSampler(graph);
  auto res = sampler.Sample(req_num, trials_num, padding);
  CUDACheckError();
  cudaDeviceSynchronize();
  const auto& rows = std::get<0>(res);
  const auto& cols = std::get<1>(res);
  EXPECT_EQ(rows.device().type(), torch::kCUDA);
  EXPECT_EQ(rows.device().index(), 0);
  EXPECT_EQ(cols.device().type(), torch::kCUDA);
  EXPECT_EQ(cols.device().index(), 0);
  EXPECT_EQ(rows.size(0), req_num);
  EXPECT_EQ(cols.size(0), req_num);
  std::cout << "rows: " << rows << std::endl;
  std::cout << "cols: " << cols << std::endl;
  delete graph;
}