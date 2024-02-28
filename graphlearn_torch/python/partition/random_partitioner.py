# Copyright 2022 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from typing import List, Dict, Optional, Tuple, Union

import torch

from ..typing import NodeType, EdgeType, TensorDataType

from .base import PartitionerBase, PartitionBook


# Implementation of a random partitioner.

class RandomPartitioner(PartitionerBase):
  r""" Random partitioner for graph topology and features.

  Args:
    output_dir: The output root directory for partitioned results.
    num_parts: Number of partitions.
    num_nodes: Number of graph nodes, should be a dict for hetero data.
    edge_index: The edge index data of graph edges, should be a dict
      for hetero data.
    node_feat: The node feature data, should be a dict for hetero data.
    node_feat_dtype: The data type of node features.
    edge_feat: The edge feature data, should be a dict for hetero data.
    edge_feat_dtype: The data type of edge features.
    edge_assign_strategy: The assignment strategy when partitioning edges,
      should be 'by_src' or 'by_dst'.
    chunk_size: The chunk size for partitioning.
  """
  def __init__(
    self,
    output_dir: str,
    num_parts: int,
    num_nodes: Union[int, Dict[NodeType, int]],
    edge_index: Union[TensorDataType, Dict[EdgeType, TensorDataType]],
    node_feat: Optional[Union[TensorDataType, Dict[NodeType, TensorDataType]]] = None,
    node_feat_dtype: torch.dtype = torch.float32,
    edge_feat: Optional[Union[TensorDataType, Dict[EdgeType, TensorDataType]]] = None,
    edge_feat_dtype: torch.dtype = torch.float32,
    edge_weights: Optional[Union[TensorDataType, Dict[EdgeType, TensorDataType]]] = None,
    edge_assign_strategy: str = 'by_src',
    chunk_size: int = 10000,
  ):
    super().__init__(output_dir, num_parts, num_nodes, edge_index, node_feat,
                     node_feat_dtype, edge_feat, edge_feat_dtype, edge_weights,
                     edge_assign_strategy, chunk_size)

  def _partition_node(
    self,
    ntype: Optional[NodeType] = None
  ) -> Tuple[List[torch.Tensor], PartitionBook]:
    if 'hetero' == self.data_cls:
      assert ntype is not None
      node_num = self.num_nodes[ntype]
    else:
      node_num = self.num_nodes
    ids = torch.arange(node_num, dtype=torch.int64)
    partition_book = ids % self.num_parts
    rand_order = torch.randperm(ids.size(0))
    partition_book = partition_book[rand_order]
    partition_results = []
    for pidx in range(self.num_parts):
      mask = (partition_book == pidx)
      partition_results.append(torch.masked_select(ids, mask))
    return partition_results, partition_book

  def _cache_node(
    self,
    ntype: Optional[NodeType] = None
  ) -> List[Optional[torch.Tensor]]:
    return [None for _ in range(self.num_parts)]
