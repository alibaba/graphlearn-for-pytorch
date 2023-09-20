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

#ifdef WITH_VINEYARD

#include "graphlearn_torch/include/common.cuh"
#include "graphlearn_torch/include/graph.h"
#include "graphlearn_torch/v6d/vineyard_utils.h"

#include <grape/communication/communicator.h>
#include <grape/worker/comm_spec.h>

using namespace graphlearn_torch;

#endif


int main(int argc, const char** argv) {
#ifdef WITH_VINEYARD

  // This test case is for the dataset
  // https://github.com/GraphScope/gstest/tree/master/modern_graph

  torch::Tensor indptr;
  torch::Tensor indices;
  torch::Tensor edge_ids;

  grape::InitMPIComm();
  {
    grape::CommSpec comm_spec;
    comm_spec.Init(MPI_COMM_WORLD);
    std::string ipc_socket = std::string(argv[1]);
    std::string object_id_str = std::string(argv[2 + comm_spec.fid()]);

    auto g = Graph();
    std::tie(indptr, indices, edge_ids) = ToCSR(ipc_socket, object_id_str, 0, 1, true);

    std::vector<std::string> vcols = {"age", "id"};
    std::vector<std::string> ecols = {"weight"};
    auto vfeat = LoadVertexFeatures(
      ipc_socket, object_id_str, 0, vcols, graphlearn_torch::DataType::Int64);
    auto efeat = LoadEdgeFeatures(
      ipc_socket, object_id_str, 0, ecols, graphlearn_torch::DataType::Float64);
    EXPECT_EQ(vfeat.size(0), 4);
    EXPECT_EQ(efeat.size(0), 2);
    EXPECT_EQ(vfeat[0][0].item<int64_t>(), 27);
    EXPECT_EQ(vfeat[3][1].item<int64_t>(), 1);
    EXPECT_NEAR(efeat[1].item<double>(), 1.0, 1e-6);

    const auto p = edge_ids.data_ptr<int64_t>();

    EXPECT_EQ(p[0] + p[1] + p[2] + p[3], 6);

    int64_t row_count = 4;
    int64_t col_count = 2;
    int64_t edge_count = 4;

    g.InitCUDAGraphFromCSR(indptr, indices, 0, GraphMode::DMA);
    CUDACheckError();
    cudaDeviceSynchronize();

    const auto cu_row_ptr = g.GetRowPtr();
    const auto cu_col_idx = g.GetColIdx();

    int64_t cpu_row_ptr[row_count+1];
    int64_t cpu_col_idx[edge_count];
    cudaMemcpy((void*)cpu_row_ptr, (void*)cu_row_ptr,
              sizeof(int64_t) * (row_count+1), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)cpu_col_idx, (void *)cu_col_idx,
              sizeof(int64_t) * edge_count, cudaMemcpyDeviceToHost);
    EXPECT_NE(cu_row_ptr, nullptr);

    EXPECT_EQ(g.GetRowCount(), row_count);
    EXPECT_EQ(g.GetColCount(), col_count);
    EXPECT_EQ(g.GetEdgeCount(), edge_count);
    for (int32_t i = 0; i < indptr.size(0); ++i) {
      EXPECT_EQ(cpu_row_ptr[i], indptr[i].item<int64_t>());
    }
    for (int32_t i = 0; i < indices.size(0); ++i) {
      EXPECT_EQ(cpu_col_idx[i], indices[i].item<int64_t>());
    }
  }

  grape::FinalizeMPIComm();

#endif
}
