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

#include "graphlearn_torch/include/tensor_map.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

#include "graphlearn_torch/include/common.h"

namespace graphlearn_torch {

/// Get the serialized size of a single `torch::Tensor` with its key name.
///
size_t GetTensorSerializedSize(const std::string& name,
                               const torch::Tensor& tensor) {
  return sizeof(size_t) + name.size() /* key */
      + sizeof(torch::ScalarType) /* data type */
      + sizeof(size_t) + tensor.sizes().size() * sizeof(int64_t) /* shape */
      + sizeof(size_t) + tensor.nbytes() /* data */;
}

/// Serialize a `torch::Tensor` into a buffer with a specified pointer.
/// \return The buffer pointer after writing.
///
void* SerializeTensor(const std::string& name,
                      const torch::Tensor& tensor,
                      void* buf_ptr) {
  Check(tensor.layout() != torch::kSparse,
        "Serializing a torch::Tensor with sparse layout "
        "is not supported now!");
  auto* write_ptr = static_cast<char*>(buf_ptr);
  // write key length
  *reinterpret_cast<size_t*>(write_ptr) = name.size();
  write_ptr = write_ptr + sizeof(size_t);
  // write key bytes
  memcpy(write_ptr, name.data(), name.size());
  write_ptr = write_ptr + name.size();
  // write data type
  *reinterpret_cast<torch::ScalarType*>(write_ptr) = tensor.scalar_type();
  write_ptr = write_ptr + sizeof(torch::ScalarType);
  // write shape number
  auto shape_array = tensor.sizes();
  *reinterpret_cast<size_t*>(write_ptr) = shape_array.size();
  write_ptr = write_ptr + sizeof(size_t);
  // write shape values
  auto shape_values_bytes = shape_array.size() * sizeof(int64_t);
  memcpy(write_ptr, shape_array.data(), shape_values_bytes);
  write_ptr = write_ptr + shape_values_bytes;
  // write data length
  *reinterpret_cast<size_t*>(write_ptr) = tensor.nbytes();
  write_ptr = write_ptr + sizeof(size_t);
  // write data bytes
  if (tensor.device().type() == torch::kCPU) {
    memcpy(write_ptr,
           tensor.data_ptr(),
           tensor.nbytes());
  } else if (tensor.device().type() == torch::kCUDA) {
#ifdef WITH_CUDA
    cudaMemcpyAsync(write_ptr, tensor.data_ptr(), tensor.nbytes(),
        cudaMemcpyDeviceToHost);
#endif
  } else {
    Check(false, "Only support serializing tensor on cpu or cuda device");
  }
  write_ptr = write_ptr + tensor.nbytes();

  return write_ptr;
}

/// Create a `torch::Tensor` from serialzed buffer bytes.
///
/// \return The key name, the created `torch::Tensor` and the buffer
/// pointer after reading.
///
std::tuple<std::string, torch::Tensor, void*>
LoadTensorFrom(void* buf_ptr, const torch::Deleter& d) {
  auto* read_ptr = static_cast<char*>(buf_ptr);
  // read key length
  auto key_len = *reinterpret_cast<size_t*>(read_ptr);
  read_ptr = read_ptr + sizeof(size_t);
  // create key name
  std::string key_name{read_ptr, key_len};
  read_ptr = read_ptr + key_len;
  // read data type
  auto data_type = *reinterpret_cast<torch::ScalarType*>(read_ptr);
  read_ptr = read_ptr + sizeof(torch::ScalarType);
  // read shape number
  auto shape_num = *reinterpret_cast<size_t*>(read_ptr);
  read_ptr = read_ptr + sizeof(size_t);
  // create shape values
  torch::IntArrayRef shapes{reinterpret_cast<int64_t*>(read_ptr), shape_num};
  read_ptr = read_ptr + shape_num * sizeof(int64_t);
  // read tensor data length
  auto data_len = *reinterpret_cast<size_t*>(read_ptr);
  read_ptr = read_ptr + sizeof(size_t);
  // create tensor
  auto options = torch::TensorOptions().dtype(data_type).device(torch::kCPU);
  auto tensor = torch::from_blob(read_ptr, shapes, d, options);
  read_ptr = read_ptr + data_len;

  return std::make_tuple(std::move(key_name), std::move(tensor), read_ptr);
}

/// Create a `TensorMap` from serialzed buffer bytes with a shared deleter.
///
TensorMap LoadTensorMapFrom(void* buf, const torch::Deleter& shd) {
  TensorMap map{};
  auto* read_ptr = static_cast<char*>(buf);
  auto tensor_num = *reinterpret_cast<size_t*>(read_ptr);
  read_ptr = read_ptr + sizeof(size_t);
  for (size_t i = 0; i < tensor_num; i++) {
    auto tuple = LoadTensorFrom(read_ptr, shd);
    map.emplace(std::move(std::get<0>(tuple)), std::move(std::get<1>(tuple)));
    read_ptr = static_cast<char*>(std::get<2>(tuple));
  }
  return map;
}

/// Methods of TensorMapSerializer

size_t TensorMapSerializer::GetSerializedSize(const TensorMap& map) {
  size_t total_size = sizeof(size_t);
  for (auto& it : map) {
    total_size += GetTensorSerializedSize(it.first, it.second);
  }
  return total_size;
}

void TensorMapSerializer::Serialize(const TensorMap& map, void* buf) {
  auto* write_ptr = static_cast<char*>(buf);
  *reinterpret_cast<size_t*>(write_ptr) = map.size();
  write_ptr = write_ptr + sizeof(size_t);
  for (auto& it : map) {
    write_ptr = static_cast<char*>(
        SerializeTensor(it.first, it.second, write_ptr));
  }
}

TensorMap TensorMapSerializer::Load(void* buf) {
  torch::Deleter shd = [](void*){};
  return LoadTensorMapFrom(buf, shd);
}

struct ShmDataSharedDeleter {
  std::shared_ptr<ShmData> shm_data;
  ShmDataSharedDeleter(ShmData&& shm_data)
      : shm_data(std::make_shared<ShmData>(std::move(shm_data))) {}
  void operator()(void*) {}
};

TensorMap TensorMapSerializer::Load(ShmData&& shm_data) {
  void* buf = shm_data.Data();
  ShmDataSharedDeleter shd{std::move(shm_data)};
  return LoadTensorMapFrom(buf, shd);
}

}  // namespace graphlearn_torch
