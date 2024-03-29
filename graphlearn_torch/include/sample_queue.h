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

#ifndef GRAPHLEARN_TORCH_INCLUDE_SAMPLE_QUEUE_H_
#define GRAPHLEARN_TORCH_INCLUDE_SAMPLE_QUEUE_H_

#include <torch/extension.h>

#include "graphlearn_torch/include/shm_queue.h"
#include "graphlearn_torch/include/tensor_map.h"

namespace graphlearn_torch {

class SampleQueue {
public:
  SampleQueue(size_t max_msg_num, size_t buf_size) {
    shmq_ = std::make_unique<ShmQueue>(max_msg_num, buf_size);
  }

  SampleQueue(int shmid) {
    shmq_ = std::make_unique<ShmQueue>(shmid);
  }

  void PinMemory() {
    shmq_->PinMemory();
  }

  int ShmId() const {
    return shmq_->ShmId();
  }

  void Enqueue(const TensorMap& msg);
  TensorMap Dequeue(unsigned int timeout_ms = 0);
  bool Empty();

private:
  std::unique_ptr<ShmQueue> shmq_;
};

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_INCLUDE_SAMPLE_QUEUE_H_
