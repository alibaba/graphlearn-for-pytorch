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
#include <vector>
#include <cuda.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "gtest/gtest.h"

#include "graphlearn_torch/include/common.cuh"
#include "graphlearn_torch/include/unified_tensor.cuh"

using namespace graphlearn_torch;

class UnifiedTensorTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto torch_cpu_tensor1 = torch::ones({128, 128}, torch::kFloat32);
    auto torch_cpu_tensor2 = torch_cpu_tensor1 * 2;

    tensors.resize(2);
    tensor_devices.resize(2);
    tensors[0] = torch_cpu_tensor1;
    tensors[1] = torch_cpu_tensor2;
    tensor_devices[0] = 0;
    tensor_devices[1] = -1;
  }

protected:
  std::vector<torch::Tensor> tensors;
  std::vector<int32_t> tensor_devices;
};

TEST_F(UnifiedTensorTest, CUDA) {
  auto unified_tensor = UnifiedTensor(0); // gpu0
  std::cout << "Start Init UnifiedTensor from torch cpu tensors." << std::endl;
  unified_tensor.InitFrom(tensors, tensor_devices);
  std::cout << "Init UnifiedTensor success." << std::endl;
  EXPECT_EQ(unified_tensor.Shape(), std::vector<int64_t>({128*2, 128}));

  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, 0);
  auto indices = torch::tensor({10, 20, 200, 210}, options);
  std::cout << "Input Indices:" << indices << std::endl;
  auto res = unified_tensor[indices];
  CUDACheckError();
  cudaDeviceSynchronize();
  std::cout << "UnifiedTensor on GPU 0 lookups tensor Success." << std::endl;
  EXPECT_EQ(res.device().type(), torch::kCUDA);
  EXPECT_EQ(res.device().index(), 0);
  EXPECT_EQ(res.size(0), 4);
  EXPECT_EQ(res.size(1), 128);
  std::cout << "Output Results:" << res << std::endl;
}
