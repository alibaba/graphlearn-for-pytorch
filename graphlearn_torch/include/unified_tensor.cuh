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

#ifndef GRAPHLEARN_TORCH_INCLUDE_UNIFIED_TENSOR_CUH_
#define GRAPHLEARN_TORCH_INCLUDE_UNIFIED_TENSOR_CUH_

#include <atomic>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace graphlearn_torch {

// CUDA Ipc shared tensor.
class SharedTensor {
public:
  SharedTensor();
  SharedTensor(int32_t device, cudaIpcMemHandle_t mem_handle,
               const std::vector<int64_t>& shape);
  std::tuple<int32_t, cudaIpcMemHandle_t, std::vector<int64_t>> ShareCUDAIpc();
  void FromCUDAIpc(
    std::tuple<int32_t, cudaIpcMemHandle_t, std::vector<int64_t>> ipc_data);

public:
  int32_t device_;
  cudaIpcMemHandle_t mem_handle_;
  std::vector<int64_t> shape_;
};

class UnifiedTensor {
public:
  UnifiedTensor(int32_t device, torch::Dtype dtype = torch::kFloat32);
  ~UnifiedTensor();
  void AppendSharedTensor(const SharedTensor& item);
  void AppendCPUTensor(const torch::Tensor& tensor);
  void InitFrom(const std::vector<torch::Tensor>& tensors,
                const std::vector<int32_t>& tensor_devices);
  torch::Tensor operator[](const torch::Tensor& indices);
  std::vector<SharedTensor> ShareCUDAIpc();

  const std::vector<int64_t> Shape() const;
  int32_t Device() const;
  int32_t Size(int32_t dim) const;
  int64_t Stride(int32_t dim) const;
  int64_t Numel() const;

  std::tuple<void**, int64_t*> GetDeviceData(int32_t device);
  void Initp2p(const std::vector<int32_t>& devices);

private:
  std::vector<int64_t> shape_;
  torch::Dtype dtype_;
  int32_t device_;
  int32_t device_count_;
  bool inited_;
  void* registered_ptr_;

  std::vector<int64_t> tensor_offsets_;
  std::vector<std::vector<int64_t>> tensor_shapes_;
  std::vector<void*> tensor_ptrs_;
  std::vector<int32_t> tensor_devices_;
  std::unordered_map<int32_t, std::tuple<void**, int64_t*>> device_data_map_;
};

} // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_INCLUDE_UNIFIED_TENSOR_CUH_
