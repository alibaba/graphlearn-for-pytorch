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

from ..typing import NodeType, EdgeType, TensorDataType, PartitionBook
from ..utils import parse_size

from .base import PartitionerBase


class FrequencyPartitioner(PartitionerBase):
  r""" Frequency-based partitioner for graph topology and features.

  Args:
    output_dir: The output root directory for partitioned results.
    num_parts: Number of partitions.
    num_nodes: Number of graph nodes, should be a dict for hetero data.
    edge_index: The edge index data of graph edges, should be a dict
      for hetero data.
    probs: The node access distribution on each partition, should be a
      dict for hetero data.
    node_feat: The node feature data, should be a dict for hetero data.
    node_feat_dtype: The data type of node features.
    edge_feat: The edge feature data, should be a dict for hetero data.
    edge_feat_dtype: The data type of edge features.
    edge_weights: The edge weights, should be a dict for hetero data.
    edge_assign_strategy: The assignment strategy when partitioning edges,
      should be 'by_src' or 'by_dst'.
    cache_memory_budget: The memory budget (in bytes) for cached node features
      per partition for each node type, should be a dict for hetero data.
    cache_ratio: The proportion to cache node features per partition for each
      node type, should be a dict for hetero data.
    chunk_size: The chunk size for partitioning.

  Note that if both `cache_memory_budget` and `cache_ratio` are provided,
  the metric that caches the smaller number of features will be used.
  If both of them set to empty dict, the feature cache will be turned off.
  """
  def __init__(
    self,
    output_dir: str,
    num_parts: int,
    num_nodes: Union[int, Dict[NodeType, int]],
    edge_index: Union[TensorDataType, Dict[EdgeType, TensorDataType]],
    probs: Union[List[torch.Tensor], Dict[NodeType, List[torch.Tensor]]],
    node_feat: Optional[Union[TensorDataType, Dict[NodeType, TensorDataType]]] = None,
    node_feat_dtype: torch.dtype = torch.float32,
    edge_feat: Optional[Union[TensorDataType, Dict[EdgeType, TensorDataType]]] = None,
    edge_feat_dtype: torch.dtype = torch.float32,
    edge_weights: Optional[Union[TensorDataType, Dict[EdgeType, TensorDataType]]] = None,
    edge_assign_strategy: str = 'by_src',
    cache_memory_budget: Union[int, Dict[NodeType, int]] = None,
    cache_ratio: Union[float, Dict[NodeType, float]] = None,
    chunk_size: int = 10000,
  ):
    super().__init__(output_dir, num_parts, num_nodes, edge_index, node_feat,
                     node_feat_dtype, edge_feat, edge_feat_dtype, edge_weights,
                     edge_assign_strategy, chunk_size)

    self.probs = probs
    if self.node_feat is not None:
      if 'hetero' == self.data_cls:
        self.per_feature_bytes = {}
        for ntype, feat in self.node_feat.items():
          assert len(feat.shape) == 2
          self.per_feature_bytes[ntype] = feat.shape[1] * feat.element_size()
        assert isinstance(self.probs, dict)
        for ntype, prob_list in self.probs.items():
          assert ntype in self.node_types
          assert len(prob_list) == self.num_parts
      else:
        assert len(self.node_feat.shape) == 2
        self.per_feature_bytes = (self.node_feat.shape[1] *
                                  self.node_feat.element_size())
        assert len(self.probs) == self.num_parts

    self.blob_size = self.chunk_size * self.num_parts

    if cache_memory_budget is None:
      self.cache_memory_budget = {} if 'hetero' == self.data_cls else 0
    else:
      self.cache_memory_budget = cache_memory_budget
    if cache_ratio is None:
      self.cache_ratio = {} if 'hetero' == self.data_cls else 0.0
    else:
      self.cache_ratio = cache_ratio

  def _get_chunk_probs_sum(
    self,
    chunk: torch.Tensor,
    probs: List[torch.Tensor]
  ) -> List[torch.Tensor]:
    r""" Helper function for partitioning a certain type of node to
    calculate hotness and difference between partitions.
    """
    chunk_probs_sum = [
      (torch.zeros(chunk.size(0)) + 1e-6)
      for _ in range(self.num_parts)
    ]
    for src_rank in range(self.num_parts):
      for dst_rank in range(self.num_parts):
        if dst_rank == src_rank:
          chunk_probs_sum[src_rank] += probs[dst_rank][chunk] * self.num_parts
        else:
          chunk_probs_sum[src_rank] -= probs[dst_rank][chunk]
    return chunk_probs_sum

  def _partition_node(
    self,
    ntype: Optional[NodeType] = None
  ) -> Tuple[List[torch.Tensor], PartitionBook]:
    if 'hetero' == self.data_cls:
      assert ntype is not None
      node_num = self.num_nodes[ntype]
      probs = self.probs[ntype]
    else:
      node_num = self.num_nodes
      probs = self.probs
    chunk_num = (node_num + self.chunk_size - 1) // self.chunk_size

    res = [[] for _ in range(self.num_parts)]
    current_chunk_start_pos = 0
    current_partition_idx = 0
    for _ in range(chunk_num):
      current_chunk_end_pos = min(node_num,
                                  current_chunk_start_pos + self.blob_size)
      current_chunk_size = current_chunk_end_pos - current_chunk_start_pos
      chunk = torch.arange(current_chunk_start_pos, current_chunk_end_pos,
                           dtype=torch.long)
      chunk_probs_sum = self._get_chunk_probs_sum(chunk, probs)
      assigned_node_size = 0
      per_partition_size = self.chunk_size
      for partition_idx in range(current_partition_idx,
                                 current_partition_idx + self.num_parts):
        partition_idx = partition_idx % self.num_parts
        actual_per_partition_size = min(per_partition_size,
                                        chunk.size(0) - assigned_node_size)
        _, sorted_res_order = torch.sort(chunk_probs_sum[partition_idx],
                                         descending=True)
        pick_chunk_part = sorted_res_order[:actual_per_partition_size]
        pick_ids = chunk[pick_chunk_part]
        res[partition_idx].append(pick_ids)
        for idx in range(self.num_parts):
          chunk_probs_sum[idx][pick_chunk_part] = -self.num_parts
        assigned_node_size += actual_per_partition_size
      current_partition_idx += 1
      current_chunk_start_pos += current_chunk_size

    partition_book = torch.zeros(node_num, dtype=torch.long)
    partition_results = []
    for partition_idx in range(self.num_parts):
      partition_ids = torch.cat(res[partition_idx])
      partition_results.append(partition_ids)
      partition_book[partition_ids] = partition_idx

    return partition_results, partition_book

  def _cache_node(
    self,
    ntype: Optional[NodeType] = None
  ) -> List[Optional[torch.Tensor]]:
    if 'hetero' == self.data_cls:
      assert ntype is not None
      probs = self.probs[ntype]
      per_feature_bytes = self.per_feature_bytes[ntype]
      cache_memory_budget = self.cache_memory_budget.get(ntype, 0)
      cache_ratio = self.cache_ratio.get(ntype, 0.0)
    else:
      probs = self.probs
      per_feature_bytes = self.per_feature_bytes
      cache_memory_budget = self.cache_memory_budget
      cache_ratio = self.cache_ratio
    cache_memory_budget_bytes = parse_size(cache_memory_budget)
    cache_num_by_memory = int(cache_memory_budget_bytes /
                              (per_feature_bytes + 1e-6))
    cache_num_by_memory = min(cache_num_by_memory, probs[0].size(0))
    cache_num_by_ratio = int(probs[0].size(0) * min(cache_ratio, 1.0))
    if cache_num_by_memory == 0:
      cache_num = cache_num_by_ratio
    elif cache_num_by_ratio == 0:
      cache_num = cache_num_by_memory
    else:
      cache_num = min(cache_num_by_memory, cache_num_by_ratio)

    cache_results = [None] * self.num_parts
    if cache_num > 0:
      for partition_idx in range(self.num_parts):
        _, prev_order = torch.sort(probs[partition_idx], descending=True)
        cache_results[partition_idx] = prev_order[:cache_num]
    return cache_results
