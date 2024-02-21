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

import os
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from ..partition import (
  save_meta, save_node_pb, save_edge_pb,
  save_graph_partition, save_feature_partition,
  PartitionBook
)
from ..typing import (
  NodeType, EdgeType, TensorDataType,
  GraphPartitionData, FeaturePartitionData
)
from ..utils import convert_to_tensor, ensure_dir, index_select

from .dist_context import get_context, init_worker_group
from .rpc import (
  init_rpc, rpc_is_initialized, all_gather, barrier,
  get_rpc_current_group_worker_names,
  rpc_request_async, rpc_register, RpcCalleeBase
)


class RpcUpdatePartitionValueCallee(RpcCalleeBase):
  def __init__(self, dist_partition_mgr):
    super().__init__()
    self.dist_partition_mgr = dist_partition_mgr

  def call(self, *args, **kwargs):
    self.dist_partition_mgr._update_part_val(*args, **kwargs)
    return None


class RpcUpdatePartitionBookCallee(RpcCalleeBase):
  def __init__(self, dist_partition_mgr):
    super().__init__()
    self.dist_partition_mgr = dist_partition_mgr

  def call(self, *args, **kwargs):
    self.dist_partition_mgr._update_pb(*args, **kwargs)
    return None


class DistPartitionManager(object):
  r""" A state manager for distributed partitioning.
  """
  def __init__(self, total_val_size: int = 1, generate_pb: bool = True):
    assert rpc_is_initialized() is True
    self.num_parts = get_context().world_size
    self.cur_pidx = get_context().rank

    self._lock = threading.RLock()
    self._worker_names = get_rpc_current_group_worker_names()

    self.reset(total_val_size, generate_pb)

    val_update_callee = RpcUpdatePartitionValueCallee(self)
    self._val_update_callee_id = rpc_register(val_update_callee)
    pb_update_callee = RpcUpdatePartitionBookCallee(self)
    self._pb_update_callee_id = rpc_register(pb_update_callee)

  def reset(self, total_val_size: int, generate_pb: bool = True):
    with self._lock:
      self.generate_pb = generate_pb
      self.cur_part_val_list = []
      if self.generate_pb:
        self.partition_book = torch.zeros(total_val_size, dtype=torch.int64)
      else:
        self.partition_book = None

  def process(self, res_list: List[Tuple[Any, torch.Tensor]]):
    r""" Process partitioned results of the current corresponded distributed
    partitioner and synchronize with others.

    Args:
      res_list: The result list of value and ids for each partition.
    """
    assert len(res_list) == self.num_parts
    futs = []
    for pidx, (val, val_idx) in enumerate(res_list):
      if pidx == self.cur_pidx:
        self._update_part_val(val, pidx)
      else:
        futs.append(rpc_request_async(self._worker_names[pidx],
                                      self._val_update_callee_id,
                                      args=(val, pidx)))
      if self.generate_pb:
        futs.extend(self._broadcast_pb(val_idx, pidx))
    _ = torch.futures.wait_all(futs)

  def _broadcast_pb(self, val_idx: torch.Tensor, target_pidx: int):
    pb_update_futs = []
    for pidx in range(self.num_parts):
      if pidx == self.cur_pidx:
        self._update_pb(val_idx, target_pidx)
      else:
        pb_update_futs.append(rpc_request_async(self._worker_names[pidx],
                                                self._pb_update_callee_id,
                                                args=(val_idx, target_pidx)))
    return pb_update_futs

  def _update_part_val(self, val, target_pidx: int):
    assert target_pidx == self.cur_pidx
    with self._lock:
      if val is not None:
        self.cur_part_val_list.append(val)

  def _update_pb(self, val_idx: torch.Tensor, target_pidx: int):
    with self._lock:
      self.partition_book[val_idx] = target_pidx


class DistRandomPartitioner(object):
  r""" A distributed random partitioner for parallel partitioning with large
  scale graph and features.

  Each distributed partitioner will process a part of the full graph and
  feature data, and partition them in parallel. A distributed partitioner's
  rank is corresponding to a partition index, and the number of all distributed
  partitioners must be same with the number of output partitions. During
  partitioning, the partitioned results will be sent to other distributed
  partitioners according to their ranks. After partitioning, each distributed
  partitioner will own a partitioned graph with its corresponding rank and
  further save the partitioned results into the local output directory.

  Args:
    output_dir: The output root directory on local machine for partitioned
      results.
    num_nodes: Number of all graph nodes, should be a dict for hetero data.
    edge_index: A part of the edge index data of graph edges, should be a dict
      for hetero data.
    edge_ids: The edge ids of the input ``edge_index``.
    node_feat: A part of the node feature data, should be a dict for hetero data.
    node_feat_ids: The node ids corresponding to the input ``node_feat``.
    edge_feat: The edge feature data, should be a dict for hetero data.
    edge_feat_ids: The edge ids corresponding to the input ``edge_feat``.
    num_parts: The number of all partitions. If not provided, the value of
      ``graphlearn_torch.distributed.get_context().world_size`` will be used.
    current_partition_idx: The partition index corresponding to the current
      distributed partitioner. If not provided, the value of
      ``graphlearn_torch.distributed.get_context().rank`` will be used.
    node_feat_dtype: The data type of node features.
    edge_feat_dtype: The data type of edge features.
    edge_assign_strategy: The assignment strategy when partitioning edges,
      should be 'by_src' or 'by_dst'.
    chunk_size: The chunk size for partitioning.
    master_addr: The master TCP address for RPC connection between all
      distributed partitioners.
    master_port: The master TCP port for RPC connection between all
      distributed partitioners.
    num_rpc_threads: The number of RPC worker threads to use.
  """
  def __init__(
    self,
    output_dir: str,
    num_nodes: Union[int, Dict[NodeType, int]],
    edge_index: Union[TensorDataType, Dict[EdgeType, TensorDataType]],
    edge_ids: Union[TensorDataType, Dict[EdgeType, TensorDataType]],
    node_feat: Optional[Union[TensorDataType, Dict[NodeType, TensorDataType]]] = None,
    node_feat_ids: Optional[Union[TensorDataType, Dict[NodeType, TensorDataType]]] = None,
    edge_feat: Optional[Union[TensorDataType, Dict[EdgeType, TensorDataType]]] = None,
    edge_feat_ids: Optional[Union[TensorDataType, Dict[EdgeType, TensorDataType]]] = None,
    num_parts: Optional[int] = None,
    current_partition_idx: Optional[int] = None,
    node_feat_dtype: torch.dtype = torch.float32,
    edge_feat_dtype: torch.dtype = torch.float32,
    edge_assign_strategy: str = 'by_src',
    chunk_size: int = 10000,
    master_addr: Optional[str] = None,
    master_port: Optional[str] = None,
    num_rpc_threads: int = 16,
  ):
    self.output_dir = output_dir

    if get_context() is not None:
      if num_parts is not None:
        assert get_context().world_size == num_parts
      if current_partition_idx is not None:
        assert get_context().rank == current_partition_idx
    else:
      assert num_parts is not None
      assert current_partition_idx is not None
      init_worker_group(
        world_size=num_parts,
        rank=current_partition_idx,
        group_name='distributed_random_partitoner'
      )
    self.num_parts = get_context().world_size
    self.current_partition_idx = get_context().rank

    if rpc_is_initialized() is not True:
      if master_addr is None:
        master_addr = os.environ['MASTER_ADDR']
      if master_port is None:
        master_port = int(os.environ['MASTER_PORT'])
      init_rpc(master_addr, master_port, num_rpc_threads)

    self.num_nodes = num_nodes
    self.edge_index = convert_to_tensor(edge_index, dtype=torch.int64)
    self.edge_ids = convert_to_tensor(edge_ids, dtype=torch.int64)

    self.node_feat = convert_to_tensor(node_feat, dtype=node_feat_dtype)
    self.node_feat_ids = convert_to_tensor(node_feat_ids, dtype=torch.int64)
    if self.node_feat is not None:
      assert self.node_feat_ids is not None

    self.edge_feat = convert_to_tensor(edge_feat, dtype=edge_feat_dtype)
    self.edge_feat_ids = convert_to_tensor(edge_feat_ids, dtype=torch.int64)
    if self.edge_feat is not None:
      assert self.edge_feat_ids is not None

    if isinstance(self.num_nodes, dict):
      assert isinstance(self.edge_index, dict)
      assert isinstance(self.edge_ids, dict)
      assert isinstance(self.node_feat, dict) or self.node_feat is None
      assert isinstance(self.node_feat_ids, dict) or self.node_feat_ids is None
      assert isinstance(self.edge_feat, dict) or self.edge_feat is None
      assert isinstance(self.edge_feat_ids, dict) or self.edge_feat_ids is None
      self.data_cls = 'hetero'
      self.node_types = sorted(list(self.num_nodes.keys()))
      self.edge_types = sorted(list(self.edge_index.keys()))
      self.num_local_edges = {}
      self.num_edges = {}
      for etype, index in self.edge_index.items():
        self.num_local_edges[etype] = len(index[0])
        self.num_edges[etype] = sum(all_gather(len(index[0])).values())
    else:
      self.data_cls = 'homo'
      self.node_types = None
      self.edge_types = None
      self.num_local_edges = len(self.edge_index[0])
      self.num_edges = sum(all_gather(len(self.edge_index[0])).values())

    self.edge_assign_strategy = edge_assign_strategy.lower()
    assert self.edge_assign_strategy in ['by_src', 'by_dst']
    self.chunk_size = chunk_size

    self._partition_mgr = DistPartitionManager()

  def _partition_by_chunk(
    self,
    val: Any,
    val_idx: torch.Tensor,
    partition_fn,
    total_val_size: int,
    generate_pb = True
  ):
    r""" Partition generic values and sync with all other partitoners.
    """
    val_num = len(val_idx)
    chunk_num = (val_num + self.chunk_size - 1) // self.chunk_size
    chunk_start_pos = 0
    self._partition_mgr.reset(total_val_size, generate_pb)
    barrier()
    for _ in range(chunk_num):
      chunk_end_pos = min(val_num, chunk_start_pos + self.chunk_size)
      current_chunk_size = chunk_end_pos - chunk_start_pos
      chunk_idx = torch.arange(current_chunk_size, dtype=torch.long)
      chunk_val = index_select(val, index=(chunk_start_pos, chunk_end_pos))
      chunk_val_idx = val_idx[chunk_start_pos:chunk_end_pos]
      chunk_partition_idx = partition_fn(
        chunk_val_idx, (chunk_start_pos, chunk_end_pos))
      chunk_res = []
      for pidx in range(self.num_parts):
        mask = (chunk_partition_idx == pidx)
        idx = torch.masked_select(chunk_idx, mask)
        chunk_res.append((index_select(chunk_val, idx), chunk_val_idx[idx]))
      self._partition_mgr.process(chunk_res)
      chunk_start_pos += current_chunk_size
    barrier()
    return (
      self._partition_mgr.cur_part_val_list,
      self._partition_mgr.partition_book
    )

  def _partition_node(
    self,
    ntype: Optional[NodeType] = None
  ) -> PartitionBook:
    r""" Partition graph nodes of a specify node type in parallel.

    Args:
      ntype (str): The type for input nodes, must be provided for heterogeneous
        graph. (default: ``None``)

    Returns:
      PartitionBook: The partition book of graph nodes.
    """
    if 'hetero' == self.data_cls:
      assert ntype is not None
      node_num = self.num_nodes[ntype]
    else:
      node_num = self.num_nodes
    per_node_num = node_num // self.num_parts
    local_node_start = per_node_num * self.current_partition_idx
    local_node_end = min(
      node_num,
      per_node_num * (self.current_partition_idx + 1)
    )
    local_node_ids = torch.arange(
      local_node_start, local_node_end, dtype=torch.int64
    )

    def _node_pfn(n_ids, _):
      partition_idx = n_ids % self.num_parts
      rand_order = torch.randperm(len(n_ids))
      return partition_idx[rand_order]

    _, node_pb = self._partition_by_chunk(
      val=None,
      val_idx=local_node_ids,
      partition_fn=_node_pfn,
      total_val_size=node_num,
      generate_pb=True
    )
    return node_pb

  def _partition_graph(
    self,
    node_pbs: Union[PartitionBook, Dict[NodeType, PartitionBook]],
    etype: Optional[EdgeType] = None
  ) -> Tuple[GraphPartitionData, PartitionBook]:
    r""" Partition graph topology of a specify edge type in parallel.

    Args:
      node_pbs (PartitionBook or Dict[NodeType, PartitionBook]): The
        partition books of all graph nodes.
      etype (Tuple[str, str, str]): The type for input edges, must be provided
        for heterogeneous graph. (default: ``None``)

    Returns:
      GraphPartitionData: The graph data of the current partition.
      PartitionBook: The partition book of graph edges.
    """
    if 'hetero' == self.data_cls:
      assert isinstance(node_pbs, dict)
      assert etype is not None
      src_ntype, _, dst_ntype = etype

      edge_index = self.edge_index[etype]
      rows, cols = edge_index[0], edge_index[1]
      eids = self.edge_ids[etype]
      edge_num = self.num_edges[etype]

      if 'by_src' == self.edge_assign_strategy:
        target_node_pb = node_pbs[src_ntype]
        target_indices = rows
      else:
        target_node_pb = node_pbs[dst_ntype]
        target_indices = cols
    else:
      edge_index = self.edge_index
      rows, cols = edge_index[0], edge_index[1]
      eids = self.edge_ids
      edge_num = self.num_edges

      target_node_pb = node_pbs
      target_indices = rows if 'by_src' == self.edge_assign_strategy else cols

    def _edge_pfn(_, chunk_range):
      chunk_target_indices = index_select(target_indices, chunk_range)
      return target_node_pb[chunk_target_indices]

    res_list, edge_pb = self._partition_by_chunk(
      val=(rows, cols, eids),
      val_idx=eids,
      partition_fn=_edge_pfn,
      total_val_size=edge_num,
      generate_pb=True
    )
    current_graph_part = GraphPartitionData(
      edge_index=(
        torch.cat([r[0] for r in res_list]),
        torch.cat([r[1] for r in res_list]),
      ),
      eids=torch.cat([r[2] for r in res_list])
    )
    return current_graph_part, edge_pb

  def _partition_node_feat(
    self,
    node_pb: PartitionBook,
    ntype: Optional[NodeType] = None,
  ) -> Optional[FeaturePartitionData]:
    r""" Split node features in parallel by the partitioned node results,
    and return the current partition of node features.
    """
    if self.node_feat is None:
      return None

    if 'hetero' == self.data_cls:
      assert ntype is not None
      node_num = self.num_nodes[ntype]
      node_feat = self.node_feat[ntype]
      node_feat_ids = self.node_feat_ids[ntype]
    else:
      node_num = self.num_nodes
      node_feat = self.node_feat
      node_feat_ids = self.node_feat_ids

    def _node_feat_pfn(nfeat_ids, _):
      return node_pb[nfeat_ids]

    res_list, _ = self._partition_by_chunk(
      val=(node_feat, node_feat_ids),
      val_idx=node_feat_ids,
      partition_fn=_node_feat_pfn,
      total_val_size=node_num,
      generate_pb=False
    )
    return FeaturePartitionData(
      feats=torch.cat([r[0] for r in res_list]),
      ids=torch.cat([r[1] for r in res_list]),
      cache_feats=None,
      cache_ids=None
    )

  def _partition_edge_feat(
    self,
    edge_pb: PartitionBook,
    etype: Optional[EdgeType] = None,
  ) -> Optional[FeaturePartitionData]:
    r""" Split edge features in parallel by the partitioned edge results,
    and return the current partition of edge features.
    """
    if self.edge_feat is None:
      return None

    if 'hetero' == self.data_cls:
      assert etype is not None
      edge_num = self.num_edges[etype]
      edge_feat = self.edge_feat[etype]
      edge_feat_ids = self.edge_feat_ids[etype]
    else:
      edge_num = self.num_edges
      edge_feat = self.edge_feat
      edge_feat_ids = self.edge_feat_ids

    def _edge_feat_pfn(efeat_ids, _):
      return edge_pb[efeat_ids]

    res_list, _ = self._partition_by_chunk(
      val=(edge_feat, edge_feat_ids),
      val_idx=edge_feat_ids,
      partition_fn=_edge_feat_pfn,
      total_val_size=edge_num,
      generate_pb=False
    )
    return FeaturePartitionData(
      feats=torch.cat([r[0] for r in res_list]),
      ids=torch.cat([r[1] for r in res_list]),
      cache_feats=None,
      cache_ids=None
    )

  def partition(self):
    r""" Partition graph and feature data into different parts along with all
    other distributed partitioners, save the result of the current partition
    index into output directory.
    """
    ensure_dir(self.output_dir)
    if 'hetero' == self.data_cls:
      node_pb_dict = {}
      for ntype in self.node_types:
        node_pb = self._partition_node(ntype)
        node_pb_dict[ntype] = node_pb
        save_node_pb(self.output_dir, node_pb, ntype)

        current_node_feat_part = self._partition_node_feat(node_pb, ntype)
        if current_node_feat_part is not None:
          save_feature_partition(
            self.output_dir, self.current_partition_idx, current_node_feat_part,
            group='node_feat', graph_type=ntype
          )
        del current_node_feat_part

      for etype in self.edge_types:
        current_graph_part, edge_pb = self._partition_graph(node_pb_dict, etype)
        save_edge_pb(self.output_dir, edge_pb, etype)
        save_graph_partition(
          self.output_dir, self.current_partition_idx, current_graph_part, etype
        )
        del current_graph_part

        current_edge_feat_part = self._partition_edge_feat(edge_pb, etype)
        if current_edge_feat_part is not None:
          save_feature_partition(
            self.output_dir, self.current_partition_idx, current_edge_feat_part,
            group='edge_feat', graph_type=etype
          )
        del current_edge_feat_part

    else:
      node_pb = self._partition_node()
      save_node_pb(self.output_dir, node_pb)

      current_node_feat_part = self._partition_node_feat(node_pb)
      if current_node_feat_part is not None:
        save_feature_partition(
          self.output_dir, self.current_partition_idx,
          current_node_feat_part, group='node_feat'
        )
      del current_node_feat_part

      current_graph_part, edge_pb = self._partition_graph(node_pb)
      save_edge_pb(self.output_dir, edge_pb)
      save_graph_partition(
        self.output_dir, self.current_partition_idx, current_graph_part
      )
      del current_graph_part

      current_edge_feat_part = self._partition_edge_feat(edge_pb)
      if current_edge_feat_part is not None:
        save_feature_partition(
          self.output_dir, self.current_partition_idx,
          current_edge_feat_part, group='edge_feat'
        )
      del current_edge_feat_part

    # save meta.
    save_meta(self.output_dir, self.num_parts, self.data_cls,
              self.node_types, self.edge_types)
