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

#include <cub/cub.cuh>

#include "graphlearn_torch/include/hash_table.cuh"
#include "graphlearn_torch/include/common.cuh"

namespace graphlearn_torch {

constexpr int32_t BLOCK_SIZE = 256;
constexpr int32_t TILE_SIZE = 256;

__global__ void InsertDeviceHashTableKernel(const int64_t* keys,
                                            const int32_t keys_num,
                                            const int32_t offset,
                                            DeviceHashTable device_table) {
  const int32_t block_start = TILE_SIZE * blockIdx.x;
  const int32_t block_end = min((blockIdx.x + 1) * TILE_SIZE, keys_num);
  for (int32_t index = threadIdx.x + block_start;
       index < block_end;
       index += BLOCK_SIZE) {
    device_table.Insert(keys[index], index + offset);
  }
}

__global__ void CountUniqueKeyKernel(const int64_t* keys,
                                     const int32_t keys_num,
                                     const int32_t offset,
                                     DeviceHashTable device_table,
                                     int32_t* out_prefix) {
  const int32_t block_start = TILE_SIZE * blockIdx.x;
  const int32_t block_end = min((blockIdx.x + 1) * TILE_SIZE, keys_num);
  for (int32_t index = threadIdx.x + block_start; index < block_end;
      index += BLOCK_SIZE) {
    const auto& kv = *(device_table.Lookup(keys[index]));
    if (kv.input_idx == (index + offset)) {
      out_prefix[index] = 1;
    }
  }
}

__global__ void FillValueKernel(const int64_t* keys,
                                const int32_t keys_num,
                                const int32_t idx_offset,
                                const int32_t value_offset,
                                DeviceHashTable device_table,
                                const int32_t* out_prefix,
                                int64_t* unique_keys) {
  const int32_t block_start = TILE_SIZE * blockIdx.x;
  const int32_t block_end = min((blockIdx.x + 1) * TILE_SIZE, keys_num);
  for (int32_t index = threadIdx.x + block_start;
       index < block_end;
       index += BLOCK_SIZE) {
    auto& kv = *(device_table.Lookup(keys[index]));
    if (kv.input_idx == (index + idx_offset)) {
      kv.value = (out_prefix[index] + value_offset);
      unique_keys[out_prefix[index]] = keys[index];
    }
  }
}

void HostHashTable::InsertDeviceHashTable(cudaStream_t stream,
                                          const int64_t* keys,
                                          const int32_t keys_num,
                                          int64_t* unique_keys,
                                          int32_t* unique_keys_num) {
  // insert (key, input_index) to device table.
  const dim3 grid((keys_num + TILE_SIZE - 1) / TILE_SIZE);
  const dim3 block(BLOCK_SIZE);
  InsertDeviceHashTableKernel<<<grid, block, 0, stream>>>(
    keys, keys_num, input_count_, device_table_);
  // compute output prefix.
  int32_t* out_prefix = static_cast<int32_t*>(
      CUDAAlloc(sizeof(int32_t) * (keys_num + 1), stream));
  cudaMemsetAsync((void*)out_prefix, 0, sizeof(int32_t) * (keys_num + 1), stream);
  CountUniqueKeyKernel<<<grid, block, 0, stream>>>(
    keys, keys_num, input_count_, device_table_, out_prefix);
  // update value in device table.
  size_t prefix_temp_size = 0;
  cub::DeviceScan::ExclusiveSum(
      nullptr, prefix_temp_size, out_prefix, out_prefix, keys_num + 1, stream);
  void* prefix_temp = CUDAAlloc(prefix_temp_size, stream);
  cub::DeviceScan::ExclusiveSum(
      prefix_temp, prefix_temp_size, out_prefix, out_prefix, keys_num + 1, stream);
  CUDADelete(prefix_temp);

  FillValueKernel<<<grid, block, 0, stream>>>(
    keys, keys_num, input_count_, size_, device_table_, out_prefix, unique_keys);
  cudaMemcpyAsync((void*)unique_keys_num, (void*)(out_prefix + keys_num),
      sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
  input_count_ += keys_num;
  size_ += *unique_keys_num;
  CUDADelete((void*)out_prefix);
}

} // namespace graphlearn_torch
