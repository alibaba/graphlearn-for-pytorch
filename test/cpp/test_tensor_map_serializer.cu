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

#include "gtest/gtest.h"

#include "graphlearn_torch/include/common.cuh"
#include "graphlearn_torch/include/tensor_map.h"

using namespace graphlearn_torch;

TEST(TensorMapSerializer, Serialization) {
  int current_device;
  cudaGetDevice(&current_device);

  TensorMap orignal_map;
  orignal_map.emplace("t1", torch::ones({3, 4}, torch::TensorOptions()
      .dtype(torch::kFloat)
      .device(torch::kCUDA, current_device)));
  CUDACheckError();
  orignal_map.emplace("t2", torch::empty({1}, torch::TensorOptions()
      .dtype(torch::kChar)
      .device(torch::kCPU)));
  orignal_map.emplace("t3", torch::zeros({2, 2, 2}, torch::TensorOptions()
      .dtype(torch::kInt)
      .device(torch::kCPU)));

  auto buf_size = TensorMapSerializer::GetSerializedSize(orignal_map);
  char buf[buf_size];
  TensorMapSerializer::Serialize(orignal_map, buf);

  auto loaded_map = TensorMapSerializer::Load(buf);

  auto& t1 = loaded_map.at("t1");
  EXPECT_EQ(t1.scalar_type(), torch::kFloat);
  EXPECT_EQ(t1.device().type(), torch::kCPU);
  std::vector<int64_t> t1_size_vec{3, 4};
  torch::IntArrayRef t1_sizes(t1_size_vec);
  EXPECT_EQ(t1.sizes(), t1_sizes);
  for (size_t i = 0; i < 12; i++) {
    EXPECT_EQ(t1.data_ptr<float>()[i], 1.0);
  }

  auto& t2 = loaded_map.at("t2");
  EXPECT_EQ(t2.scalar_type(), torch::kChar);
  EXPECT_EQ(t2.device().type(), torch::kCPU);
  EXPECT_EQ(t2.sizes(), torch::IntArrayRef{1});
  EXPECT_EQ(t2.nbytes(), 1);

  auto& t3 = loaded_map.at("t3");
  EXPECT_EQ(t3.scalar_type(), torch::kInt);
  EXPECT_EQ(t3.device().type(), torch::kCPU);
  std::vector<int64_t> t3_size_vec{2, 2, 2};
  torch::IntArrayRef t3_sizes(t3_size_vec);
  EXPECT_EQ(t3.sizes(), t3_sizes);
  for (size_t i = 0; i < 8; i++) {
    EXPECT_EQ(t3.data_ptr<int>()[i], 0);
  }
}
