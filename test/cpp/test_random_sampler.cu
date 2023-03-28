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

#include "gtest/gtest.h"

#include "graphlearn_torch/include/common.cuh"
#include "graphlearn_torch/include/graph.h"
#include "graphlearn_torch/csrc/cpu/random_sampler.h"
#include "graphlearn_torch/csrc/cuda/random_sampler.cuh"

using namespace graphlearn_torch;

class SamplerTest : public ::testing::Test {
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
  torch::Tensor indptr;
  torch::Tensor indices;
  torch::Tensor edge_ids;
};


void CheckRandomUniqueNbrs(const torch::Tensor& nbrs, const torch::Tensor& nbrs_num) {
  auto cpu_nbrs = nbrs.to(device(torch::kCPU));
  auto cpu_nbrs_num = nbrs_num.to(device(torch::kCPU));
  auto nbr_ptr = cpu_nbrs.data_ptr<int64_t>();
  auto nbr_num_ptr = cpu_nbrs_num.data_ptr<int64_t>();
  EXPECT_EQ(nbr_num_ptr[0], 2);
  EXPECT_EQ(nbr_num_ptr[1], 2);
  EXPECT_EQ(nbr_num_ptr[2], 2);
  EXPECT_EQ(nbr_num_ptr[3], 2);
  EXPECT_EQ(nbr_num_ptr[4], 2);
  EXPECT_EQ(nbr_num_ptr[5], 1);
  int64_t offset = 0;

  std::unordered_set<int64_t> nbr_set_0({0, 1});
  for (int32_t i = offset; i < offset + nbr_num_ptr[0]; ++i) {
    EXPECT_TRUE(nbr_set_0.find(nbr_ptr[i]) != nbr_set_0.end());
  }
  offset += nbr_num_ptr[0];

  std::unordered_set<int64_t> nbr_set_1({1, 3});
  for (int32_t i = offset; i < offset + nbr_num_ptr[1]; ++i) {
    EXPECT_TRUE(nbr_set_1.find(nbr_ptr[i]) != nbr_set_1.end());
  }
  offset += nbr_num_ptr[1];

  std::unordered_set<int64_t> nbr_set_2({2, 3, 4});
  for (int32_t k = 2; k < 5; ++k) {
    for (int32_t i = offset; i < offset + nbr_num_ptr[k]; ++i) {
      EXPECT_TRUE(nbr_set_2.find(nbr_ptr[i]) != nbr_set_2.end());
    }
    offset += nbr_num_ptr[k];
  }

  std::unordered_set<int64_t> nbr_set_3({5});
  for (int32_t i = offset; i < offset + nbr_num_ptr[5]; ++i) {
    EXPECT_TRUE(nbr_set_3.find(nbr_ptr[i]) != nbr_set_3.end());
  }
  std::cout << nbrs << std::endl;
  std::cout << nbrs_num << std::endl;
}

TEST_F(SamplerTest, CPURandom) {
  Graph* graph = new Graph();
  int32_t req_num = 2;
  graph->InitCPUGraphFromCSR(indptr, indices);
  auto sampler = CPURandomSampler(graph);
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto inputs = torch::tensor({0, 1, 2, 2, 2, 3}, options);
  auto res = sampler.Sample(inputs, req_num);
  CheckRandomUniqueNbrs(std::get<0>(res), std::get<1>(res));
  delete graph;
}

TEST_F(SamplerTest, CPURandomWithEdge) {
  Graph* graph = new Graph();
  int32_t req_num = 2;
  graph->InitCPUGraphFromCSR(indptr, indices, edge_ids);
  auto sampler = CPURandomSampler(graph);
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto inputs = torch::tensor({0, 1, 2, 2, 2, 3}, options);
  auto res = sampler.SampleWithEdge(inputs, req_num);
  CheckRandomUniqueNbrs(std::get<0>(res), std::get<1>(res));
  auto eids = std::get<2>(res).to(device(torch::kCPU));
  auto eids_ptr = eids.data_ptr<int64_t>();
  for (int32_t i = 0; i < eids.size(0); ++i) {
    std::cout << "eid: " << eids_ptr[i] << std::endl;
  }
  delete graph;
}

TEST_F(SamplerTest, CUDADMARandom) {
  Graph* graph = new Graph();
  int32_t req_num = 2;
  graph->InitCUDAGraphFromCSR(indptr, indices, 0, GraphMode::DMA);
  auto sampler = CUDARandomSampler(graph);
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, 0);
  auto inputs = torch::tensor({0, 1, 2, 2, 2, 3}, options);
  auto res = sampler.Sample(inputs, req_num);
  CUDACheckError();
  cudaDeviceSynchronize();
  CheckRandomUniqueNbrs(std::get<0>(res), std::get<1>(res));
  delete graph;
}

TEST_F(SamplerTest, CUDADMARandomWithEdge) {
  Graph* graph = new Graph();
  int32_t req_num = 2;
  graph->InitCUDAGraphFromCSR(indptr, indices, 0, GraphMode::DMA, edge_ids);
  auto sampler = CUDARandomSampler(graph);
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, 0);
  auto inputs = torch::tensor({0, 1, 2, 2, 2, 3}, options);
  auto res = sampler.SampleWithEdge(inputs, req_num);
  CUDACheckError();
  cudaDeviceSynchronize();
  CheckRandomUniqueNbrs(std::get<0>(res), std::get<1>(res));
  auto eids = std::get<2>(res).to(device(torch::kCPU));
  auto eids_ptr = eids.data_ptr<int64_t>();
  for (int32_t i = 0; i < eids.size(0); ++i) {
    std::cout << "eid: " << eids_ptr[i] << std::endl;
  }
  delete graph;
}

TEST_F(SamplerTest, CUDAPinRandom) {
  Graph* graph = new Graph();
  int32_t req_num = 2;
  graph->InitCUDAGraphFromCSR(indptr, indices, 0, GraphMode::ZERO_COPY);
  auto sampler = CUDARandomSampler(graph);
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, 0);
  auto inputs = torch::tensor({0, 1, 2, 2, 2, 3}, options);
  auto res = sampler.Sample(inputs, req_num);
  CUDACheckError();
  cudaDeviceSynchronize();
  CheckRandomUniqueNbrs(std::get<0>(res), std::get<1>(res));
  delete graph;
}

TEST_F(SamplerTest, CUDAPinRandomWithEdge) {
  Graph* graph = new Graph();
  int32_t req_num = 2;
  graph->InitCUDAGraphFromCSR(indptr, indices, 0, GraphMode::ZERO_COPY, edge_ids);
  auto sampler = CUDARandomSampler(graph);
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, 0);
  auto inputs = torch::tensor({0, 1, 2, 2, 2, 3}, options);
  auto res = sampler.SampleWithEdge(inputs, req_num);
  CUDACheckError();
  cudaDeviceSynchronize();
  CheckRandomUniqueNbrs(std::get<0>(res), std::get<1>(res));
  auto eids = std::get<2>(res).to(device(torch::kCPU));
  auto eids_ptr = eids.data_ptr<int64_t>();
  for (int32_t i = 0; i < eids.size(0); ++i) {
    std::cout << "eid: " << eids_ptr[i] << std::endl;
  }
  delete graph;
}