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

#include "graphlearn_torch/include/sample_queue.h"

namespace graphlearn_torch {

void SampleQueue::Enqueue(const TensorMap& msg) {
  auto serialized_size = TensorMapSerializer::GetSerializedSize(msg);
  shmq_->Enqueue(serialized_size, [&msg] (void* buf) {
    TensorMapSerializer::Serialize(msg, buf);
  });
}

TensorMap SampleQueue::Dequeue(unsigned int timeout_ms) {
  auto shm_data = shmq_->Dequeue(timeout_ms);
  return TensorMapSerializer::Load(std::move(shm_data));
}

bool SampleQueue::Empty() {
  return shmq_->Empty();
}

}  // namespace graphlearn_torch
