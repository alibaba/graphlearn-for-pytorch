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

#include "gtest/gtest.h"

#include "graphlearn_torch/include/common.cuh"
#include "graphlearn_torch/include/graph.h"

using namespace graphlearn_torch;

class GraphTest : public ::testing::Test {
protected:
  void SetUp() override {
    row_count = 4;
    col_count = 6;
    edge_count = 8;
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
    pindptr = indptr.data_ptr<int64_t>();
    pindices = indices.data_ptr<int64_t>();
    pedge_ids = edge_ids.data_ptr<int64_t>();
  }

protected:
  torch::Tensor indptr;
  torch::Tensor indices;
  torch::Tensor edge_ids;
  const int64_t* pindptr;
  const int64_t* pindices;
  const int64_t* pedge_ids;
  int64_t row_count;
  int64_t col_count;
  int64_t edge_count;
};

TEST_F(GraphTest, CUDA) {
  auto cuda_graph = Graph();
  cuda_graph.InitCUDAGraphFromCSR(indptr, indices, 0, GraphMode::DMA);
  CUDACheckError();
  cudaDeviceSynchronize();
  const auto cu_row_ptr = cuda_graph.GetRowPtr();
  const auto cu_col_idx = cuda_graph.GetColIdx();
  int64_t cpu_row_ptr[row_count+1];
  int64_t cpu_col_idx[edge_count];
  cudaMemcpy((void*)cpu_row_ptr, (void*)cu_row_ptr,
             sizeof(int64_t) * (row_count+1), cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)cpu_col_idx, (void *)cu_col_idx,
             sizeof(int64_t) * edge_count, cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_graph.GetRowCount(), row_count);
  EXPECT_EQ(cuda_graph.GetColCount(), col_count);
  EXPECT_EQ(cuda_graph.GetEdgeCount(), edge_count);
  for (int32_t i = 0; i < indptr.size(0); ++i) {
    EXPECT_EQ(cpu_row_ptr[i], indptr[i].item<int64_t>());
  }
  for (int32_t i = 0; i < indices.size(0); ++i) {
    EXPECT_EQ(cpu_col_idx[i], indices[i].item<int64_t>());
  }
}

TEST_F(GraphTest, CUDAWithEdgeId) {
  auto cuda_graph = Graph();
  cuda_graph.InitCUDAGraphFromCSR(indptr, indices, 0, GraphMode::DMA, edge_ids);
  CUDACheckError();
  cudaDeviceSynchronize();
  const auto cu_row_ptr = cuda_graph.GetRowPtr();
  const auto cu_col_idx = cuda_graph.GetColIdx();
  const auto cu_edge_id = cuda_graph.GetEdgeId();
  int64_t cpu_row_ptr[row_count+1];
  int64_t cpu_col_idx[edge_count];
  int64_t cpu_edge_id[edge_count];
  cudaMemcpy((void*)cpu_row_ptr, (void*)cu_row_ptr,
             sizeof(int64_t) * (row_count+1), cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)cpu_col_idx, (void *)cu_col_idx,
             sizeof(int64_t) * edge_count, cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)cpu_edge_id, (void *)cu_edge_id,
             sizeof(int64_t) * edge_count, cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_graph.GetRowCount(), row_count);
  EXPECT_EQ(cuda_graph.GetColCount(), col_count);
  EXPECT_EQ(cuda_graph.GetEdgeCount(), edge_count);
  for (int32_t i = 0; i < indptr.size(0); ++i) {
    EXPECT_EQ(cpu_row_ptr[i], indptr[i].item<int64_t>());
  }
  for (int32_t i = 0; i < indices.size(0); ++i) {
    EXPECT_EQ(cpu_col_idx[i], indices[i].item<int64_t>());
  }
  for (int32_t i = 0; i < edge_ids.size(0); ++i) {
    EXPECT_EQ(cpu_edge_id[i], edge_ids[i].item<int64_t>());
  }
}

TEST_F(GraphTest, Pin) {
  auto pin_graph = Graph();
  pin_graph.InitCUDAGraphFromCSR(indptr, indices, 0, GraphMode::ZERO_COPY);
  CUDACheckError();
  const auto pin_row_ptr = pin_graph.GetRowPtr();
  const auto pin_col_idx = pin_graph.GetColIdx();
  EXPECT_EQ(pin_graph.GetRowCount(), row_count);
  EXPECT_EQ(pin_graph.GetColCount(), col_count);
  EXPECT_EQ(pin_graph.GetEdgeCount(), edge_count);
  for (int32_t i = 0; i < indptr.size(0); ++i) {
    EXPECT_EQ(pin_row_ptr[i], indptr[i].item<int64_t>());
  }
  for (int32_t i = 0; i < indices.size(0); ++i) {
    EXPECT_EQ(pin_col_idx[i], indices[i].item<int64_t>());
  }
}

TEST_F(GraphTest, PinWithEdgeId) {
  auto pin_graph = Graph();
  pin_graph.InitCUDAGraphFromCSR(indptr, indices, 0, GraphMode::ZERO_COPY, edge_ids);
  CUDACheckError();
  const auto pin_row_ptr = pin_graph.GetRowPtr();
  const auto pin_col_idx = pin_graph.GetColIdx();
  const auto pin_edge_id = pin_graph.GetEdgeId();
  EXPECT_EQ(pin_graph.GetRowCount(), row_count);
  EXPECT_EQ(pin_graph.GetColCount(), col_count);
  EXPECT_EQ(pin_graph.GetEdgeCount(), edge_count);
  for (int32_t i = 0; i < indptr.size(0); ++i) {
    EXPECT_EQ(pin_row_ptr[i], indptr[i].item<int64_t>());
  }
  for (int32_t i = 0; i < indices.size(0); ++i) {
    EXPECT_EQ(pin_col_idx[i], indices[i].item<int64_t>());
  }
  for (int32_t i = 0; i < edge_ids.size(0); ++i) {
    EXPECT_EQ(pin_edge_id[i], edge_ids[i].item<int64_t>());
  }
}

TEST_F(GraphTest, CUDAPointer) {
  auto cuda_graph = Graph();
  cuda_graph.InitCUDAGraphFromCSR(pindptr, 5, pindices, 8,
      nullptr, 0, GraphMode::DMA);
  CUDACheckError();
  cudaDeviceSynchronize();
  const auto cu_row_ptr = cuda_graph.GetRowPtr();
  const auto cu_col_idx = cuda_graph.GetColIdx();
  int64_t cpu_row_ptr[row_count+1];
  int64_t cpu_col_idx[edge_count];
  cudaMemcpy((void*)cpu_row_ptr, (void*)cu_row_ptr,
             sizeof(int64_t) * (row_count+1), cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)cpu_col_idx, (void *)cu_col_idx,
             sizeof(int64_t) * edge_count, cudaMemcpyDeviceToHost);
  EXPECT_EQ(cuda_graph.GetRowCount(), row_count);
  EXPECT_EQ(cuda_graph.GetColCount(), col_count);
  EXPECT_EQ(cuda_graph.GetEdgeCount(), edge_count);
  for (int32_t i = 0; i < indptr.size(0); ++i) {
    EXPECT_EQ(cpu_row_ptr[i], indptr[i].item<int64_t>());
  }
  for (int32_t i = 0; i < indices.size(0); ++i) {
    EXPECT_EQ(cpu_col_idx[i], indices[i].item<int64_t>());
  }
}

TEST_F(GraphTest, PinPointer) {
  auto pin_graph = Graph();
  pin_graph.InitCUDAGraphFromCSR(pindptr, 5, pindices, 8,
      nullptr, 0, GraphMode::ZERO_COPY);
  CUDACheckError();
  const auto pin_row_ptr = pin_graph.GetRowPtr();
  const auto pin_col_idx = pin_graph.GetColIdx();
  EXPECT_EQ(pin_graph.GetRowCount(), row_count);
  EXPECT_EQ(pin_graph.GetColCount(), col_count);
  EXPECT_EQ(pin_graph.GetEdgeCount(), edge_count);
  for (int32_t i = 0; i < indptr.size(0); ++i) {
    EXPECT_EQ(pin_row_ptr[i], indptr[i].item<int64_t>());
  }
  for (int32_t i = 0; i < indices.size(0); ++i) {
    EXPECT_EQ(pin_col_idx[i], indices[i].item<int64_t>());
  }
}