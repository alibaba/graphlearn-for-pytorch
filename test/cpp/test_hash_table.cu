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
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "gtest/gtest.h"
#include "graphlearn_torch/include/common.cuh"
#include "graphlearn_torch/include/hash_table.cuh"

using namespace graphlearn_torch;

class HashTableTest : public ::testing::Test {
protected:
  void SetUp() override {
    thrust::host_vector<int>h_vec1(10);
    input1 = h_vec1;
    // 0 1 2 0 1 2 0 1 2 0
    for (int32_t i = 0; i < 10; ++i) {
      input1[i] = i%3;
    }
    // 1 2 3 3 6
    thrust::host_vector<int>h_vec2(5);
    input2 = h_vec2;
    input2[0] = 1;
    input2[1] = 2;
    input2[2] = 3;
    input2[3] = 3;
    input2[4] = 6;
  }

protected:
  thrust::device_vector<int64_t> input1;
  thrust::device_vector<int64_t> input2;
};

TEST_F(HashTableTest, Insert) {
  cudaStream_t stream = 0;
  HostHashTable host_table(input1.size()+input2.size(), 1);
  int64_t* unique_nodes1;
  int64_t* unique_nodes2;
  int32_t unique_nodes_num = 0;
  cudaMalloc((void**)&unique_nodes1, sizeof(int64_t) * 3);
  cudaMalloc((void**)&unique_nodes2, sizeof(int64_t) * 2);
  host_table.InsertDeviceHashTable(stream,
    thrust::raw_pointer_cast(input1.data()),
    input1.size(), unique_nodes1, &unique_nodes_num);
  EXPECT_EQ(unique_nodes_num, 3);
  EXPECT_EQ(host_table.Size(), 3);
  host_table.InsertDeviceHashTable(stream,
    thrust::raw_pointer_cast(input2.data()),
    input2.size(), unique_nodes2, &unique_nodes_num);
  EXPECT_EQ(unique_nodes_num, 2);
  EXPECT_EQ(host_table.Size(), 5);
  cudaFree(unique_nodes1);
  cudaFree(unique_nodes2);
}

__global__ void Lookup(DeviceHashTable device_table,
    const int64_t* input, const int32_t input_size, int32_t* outs) {
  int32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < input_size) {
    outs[tid] = device_table.Lookup(input[tid])->value;
  }
}

TEST_F(HashTableTest, Lookup) {
  cudaStream_t stream = 0;
  HostHashTable host_table(input1.size()+input2.size(), 1);
  int64_t* unique_nodes1;
  int64_t* unique_nodes2;
  int32_t unique_nodes_num = 0;
  cudaMalloc((void**)&unique_nodes1, sizeof(int64_t) * 3);
  cudaMalloc((void**)&unique_nodes2, sizeof(int64_t) * 2);
  host_table.InsertDeviceHashTable(stream,
    thrust::raw_pointer_cast(input1.data()),
    input1.size(), unique_nodes1, &unique_nodes_num);
  EXPECT_EQ(unique_nodes_num, 3);
  EXPECT_EQ(host_table.Size(), 3);
  host_table.InsertDeviceHashTable(stream,
    thrust::raw_pointer_cast(input2.data()),
    input2.size(), unique_nodes2, &unique_nodes_num);
  EXPECT_EQ(unique_nodes_num, 2);
  EXPECT_EQ(host_table.Size(), 5);
  cudaFree(unique_nodes1);
  cudaFree(unique_nodes2);

  int32_t h_data[10];
  int32_t* d_data;
  cudaMalloc(&d_data, 10 * sizeof(int32_t));
  const dim3 grid(1);
  const dim3 block(10);
  Lookup<<<grid, block, 0, stream>>>(host_table.DeviceHandle(),
      thrust::raw_pointer_cast(input1.data()), input1.size(), d_data);
  cudaMemcpy(h_data, d_data, 10 * sizeof(int32_t), cudaMemcpyDeviceToHost);
  CUDACheckError();
  for (int32_t i = 0; i < 10; ++i) {
    EXPECT_EQ(i%3, h_data[i]);
  }

  Lookup<<<grid, block, 0, stream>>>(host_table.DeviceHandle(),
      thrust::raw_pointer_cast(input2.data()), input2.size(), d_data);
  cudaMemcpy(h_data, d_data, 5 * sizeof(int32_t), cudaMemcpyDeviceToHost);
  CUDACheckError();
  EXPECT_EQ(h_data[0], 1);
  EXPECT_EQ(h_data[1], 2);
  EXPECT_EQ(h_data[2], 3);
  EXPECT_EQ(h_data[3], 3);
  EXPECT_EQ(h_data[4], 4);
  cudaFree(d_data);
}