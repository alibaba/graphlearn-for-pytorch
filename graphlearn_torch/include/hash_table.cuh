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

#ifndef GRAPHLEARN_TORCH_INCLUDE_HASH_TABLE_CUH_
#define GRAPHLEARN_TORCH_INCLUDE_HASH_TABLE_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "graphlearn_torch/include/common.cuh"


namespace graphlearn_torch {

struct KeyValue {
  int64_t key;
  int32_t input_idx;
  int32_t value;
};

class HostHashTable;

class DeviceHashTable {
// DeviceHashTable is a device side GPU hash table used to map
// the input key to the index of its first occurrence.
public:
  DeviceHashTable(KeyValue* kvs = nullptr, int32_t size = 0)
      : kvs_(kvs), size_(size) {}
  friend class HostHashTable;

  inline __device__ KeyValue* Lookup(const int64_t key) {
    int32_t pos = Hash(key);
    int64_t delta = 1;
    while (kvs_[pos].key != key) {
      assert(kvs_[pos].key != kEmpty);
      pos = Hash(pos+delta);
      delta +=1;
    }
    assert(pos < size_);
    return GetMutable(pos);
  }

  inline __device__ int32_t FindValue(const int64_t key) {
    int32_t pos = Hash(key);
    int64_t delta = 1;
    while (true) {
      if (kvs_[pos].key == key) return kvs_[pos].value;
      if (kvs_[pos].key == kEmpty) return kEmpty;
      pos = Hash(pos+delta);
      delta +=1;
    }
    assert(pos < size_);
    return kEmpty;
  }

  inline __device__ void Insert(const int64_t key, const int32_t index) {
    int32_t pos = Hash(key);
    int64_t delta = 1;
    using T = unsigned long long int;
    while(true) {
      const int64_t prev =
          atomicCAS(reinterpret_cast<T*>(&GetMutable(pos)->key),
                    static_cast<T>(kEmpty), static_cast<T>(key));
      if (prev == kEmpty || prev == key) {
        atomicMin(reinterpret_cast<unsigned int*>(&GetMutable(pos)->input_idx),
                  static_cast<unsigned int>(index));
        return;
      }
      pos = Hash(pos + delta);
      delta += 1;
    }
  }

private:
  inline __device__ KeyValue* GetMutable(const int32_t pos) {
    assert(pos < this->size_);
    return this->kvs_ + pos;
  }

  inline __device__ int32_t Hash(const int64_t key) const {
    return key % size_;
  }

private:
  static constexpr int64_t kEmpty = -1;
  KeyValue* kvs_;
  int32_t size_;
};

class HostHashTable {
public:
  HostHashTable(int32_t num, int scale) : input_count_(0), size_(0) {
    const int32_t next_pow2 =
        1 << static_cast<int32_t>(1 + std::log2(num >> 1));
    capacity_ = next_pow2 << scale;
    void *ptr = CUDAAlloc(capacity_ * sizeof(KeyValue));
    cudaMemsetAsync(ptr, kEmpty, capacity_ * sizeof(KeyValue));
    device_table_ = DeviceHashTable(reinterpret_cast<KeyValue*>(ptr), capacity_);
  }

  ~HostHashTable() {
    CUDADelete(device_table_.kvs_);
  }

  HostHashTable(const HostHashTable& other) = delete;

  HostHashTable& operator=(const HostHashTable& other) = delete;

  DeviceHashTable DeviceHandle() const {
    return device_table_;
  }

  void InsertDeviceHashTable(cudaStream_t stream, const int64_t* keys,
      const int32_t keys_num, int64_t* unique_keys, int32_t* unique_keys_num);

  int32_t Size() const {
    return size_;
  }

  int32_t Capacity() const {
    return capacity_;
  }

  void Clear() {
    cudaMemsetAsync(device_table_.kvs_, kEmpty, capacity_ * sizeof(KeyValue));
    size_ = 0;
    input_count_ = 0;
  }

private:
  static constexpr int64_t kEmpty = -1;
  DeviceHashTable          device_table_;
  int32_t                  capacity_;
  int32_t                  input_count_;
  int32_t                  size_;
};

} // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_INCLUDE_HASH_TABLE_CUH_