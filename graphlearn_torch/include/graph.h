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

#ifndef GRAPHLEARN_TORCH_INCLUDE_GRAPH_H_
#define GRAPHLEARN_TORCH_INCLUDE_GRAPH_H_

#include <torch/extension.h>

#include "graphlearn_torch/include/common.h"

namespace graphlearn_torch {

enum GraphMode {
  DMA,
  ZERO_COPY
};

class Graph {
public:
  Graph()
    : row_ptr_(nullptr), col_idx_(nullptr),
      edge_id_(nullptr), edge_weight_(nullptr),
      row_count_(0), edge_count_(0), col_count_(0),
      graph_mode_(GraphMode::ZERO_COPY), device_id_(0) {}

  Graph(int64_t* row_ptr, int64_t* col_idx,
        int64_t* edge_id, float* edge_weight,
        int64_t row_count, int64_t edge_count, int64_t col_count)
    : row_ptr_(row_ptr),
      col_idx_(col_idx),
      edge_id_(edge_id),
      edge_weight_(edge_weight),
      row_count_(row_count),
      edge_count_(edge_count),
      col_count_(col_count),
      graph_mode_(GraphMode::ZERO_COPY),
      device_id_(0) {}

  void InitCPUGraphFromCSR(const torch::Tensor& indptr,
                           const torch::Tensor& indices,
                           const torch::Tensor& edge_ids=torch::empty(0),
                           const torch::Tensor& edge_weights=torch::empty(0));
#ifdef WITH_CUDA
  virtual ~Graph();
  void LookupDegree(const int64_t* nodes,
                    int32_t nodes_size,
                    int64_t* degrees) const;
  void InitCUDAGraphFromCSR(const torch::Tensor& indptr,
                            const torch::Tensor& indices,
                            int device=0,
                            GraphMode graph_mode=GraphMode::ZERO_COPY,
                            const torch::Tensor& edge_ids=torch::empty(0));
  void InitCUDAGraphFromCSR(const int64_t* input_indptr,
                            int64_t indptr_size,
                            const int64_t* input_indices,
                            int64_t indices_size,
                            const int64_t* edge_ids=nullptr,
                            int device=0,
                            GraphMode graph_mode=GraphMode::ZERO_COPY);
#endif

  const int64_t* GetRowPtr() const {
    return row_ptr_;
  }

  const int64_t* GetColIdx() const {
    return col_idx_;
  }

  const int64_t* GetEdgeId() const {
    return edge_id_;
  }

  const float* GetEdgeWeight() const {
    return edge_weight_;
  }

  int64_t GetRowCount() const {
    return row_count_;
  }

  int64_t GetEdgeCount() const {
    return edge_count_;
  }

  int64_t GetColCount() const {
    return col_count_;
  }

  GraphMode GetGraphMode() const {
    return graph_mode_;
  }

  int32_t GetDeviceId() const {
    return device_id_;
  }

  void SetEdgeId(int32_t idx, int64_t value) {
    if (edge_id_ != nullptr && idx < edge_count_)
      edge_id_[idx] = value;
  }

  void AllocateEdgeId(int64_t len=0) {
    if (edge_id_ == nullptr && len > 0) {
      edge_id_ = new int64_t[len];
      edge_count_ = len;
    }
  }

private:
  int64_t*  row_ptr_;
  int64_t*  col_idx_;
  int64_t*  edge_id_;
  float*    edge_weight_;
  std::vector<void*>  registered_ptrs_;
  int64_t   row_count_;
  int64_t   edge_count_;
  int64_t   col_count_;
  GraphMode graph_mode_;
  int32_t   device_id_;
};

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_INCLUDE_GRAPH_H_
