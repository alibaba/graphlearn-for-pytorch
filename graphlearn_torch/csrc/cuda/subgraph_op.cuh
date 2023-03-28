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

#ifndef GRAPHLEARN_TORCH_CSRC_CUDA_SUBGRAPH_OP_CUH_
#define GRAPHLEARN_TORCH_CSRC_CUDA_SUBGRAPH_OP_CUH_

#include "graphlearn_torch/include/subgraph_op_base.h"

#include <torch/extension.h>
#include <unordered_map>
#include <vector>

#include "graphlearn_torch/include/graph.h"
#include "graphlearn_torch/include/types.h"

namespace graphlearn_torch {

class HostHashTable;

class CUDASubGraphOp : public SubGraphOp {
public:
  CUDASubGraphOp(const Graph* graph);
  ~CUDASubGraphOp();
  CUDASubGraphOp(const CUDASubGraphOp& other) = delete;
  CUDASubGraphOp& operator=(const CUDASubGraphOp& other) = delete;
  CUDASubGraphOp(CUDASubGraphOp&&) = default;
  CUDASubGraphOp& operator=(CUDASubGraphOp&&) = default;

  SubGraph NodeSubGraph(const torch::Tensor& srcs, bool with_edge=false) override;

private:
  void Reset();
  void InitNode(cudaStream_t stream, const torch::Tensor& srcs,
      int64_t* nodes, int32_t* nodes_size);
  void CSRSliceRows(cudaStream_t stream,
                    const int64_t* rows,
                    int32_t rows_size,
                    int64_t* indptr);
  void GetNbrsNumAndColMask(cudaStream_t stream,
                            const int64_t* nodes,
                            const int64_t* origin_nbrs_offset,
                            int32_t nodes_size,
                            int64_t* out_nbrs_num);
private:
  HostHashTable* host_table_;
  int32_t* col_mask_;
};

}  // namespace graphlearn_torch

#endif // GRAPHLEARN_TORCH_CSRC_CUDA_SUBGRAPH_OP_CUH_
