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

#ifndef GRAPHLEARN_TORCH_INCLUDE_TENSOR_MAP_H_
#define GRAPHLEARN_TORCH_INCLUDE_TENSOR_MAP_H_

#include "graphlearn_torch/include/shm_queue.h"
#include "graphlearn_torch/include/types.h"

namespace graphlearn_torch {

/// Serialization layout of a `TensorMap`:
/// | tensor_num | serialized_tensor_1 | ... | serialized_tensor_n |
///
/// Serialization layout of a `torch::Tensor`:
/// | key_length | key_bytes | data_type | shape_num | shape_values | data_length | data_bytes |

class TensorMapSerializer {
public:
  /// Get the serialized size of a `TensorMap` with all its managed tensors.
  ///
  static size_t GetSerializedSize(const TensorMap& map);

  /// Serialize a `TensorMap` into a data buffer.
  ///
  static void Serialize(const TensorMap& map, void* buf);

  /// Create a `TensorMap` from a serialized data buffer without taking
  /// ownership of the original data.
  ///
  /// The original data should maintain its lifetime during the use of
  /// created `TensorMap`.
  ///
  static TensorMap Load(void* buf);

  /// Create a `TensorMap` from a `ShmData`, the created `TensorMap` will
  /// take ownership of this shm data and release it after use.
  ///
  static TensorMap Load(ShmData&& shm_data);
};

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_INCLUDE_TENSOR_MAP_H_
