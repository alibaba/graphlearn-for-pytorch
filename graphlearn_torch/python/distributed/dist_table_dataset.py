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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, eithPer express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ODPS table related distributed partitioner and dataset."""

import datetime
from multiprocessing.reduction import ForkingPickler
import numpy as np
import torch
import time
from typing import Dict, Optional, Union

try:
  import common_io
except ImportError:
  pass

from ..typing import (
  NodeType, EdgeType, TensorDataType,
)

from .dist_dataset import DistDataset, _cat_feature_cache
from .dist_random_partitioner import DistRandomPartitioner


class DistTableRandomPartitioner(DistRandomPartitioner):
  r""" A distributed random partitioner for parallel partitioning with large
  scale edge tables and node tables.

  Each distributed partitioner will process a slice of the full table,
  and partition them in parallel. After partitioning, each distributed
  partitioner will own a partitioned graph with its corresponding rank.

  Args:
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
  Returns:
    int: Number of all partitions.
    int: The current partition idx.
    GraphPartitionData/HeteroGraphPartitionData: graph partition data.
    FeaturePartitionData/HeteroFeaturePartitionData: node feature partition
      data, optional.
    FeaturePartitionData/HeteroFeaturePartitionData: edge feature partition
      data, optional.
    PartitionBook/HeteroNodePartitionDict: node partition book.
    PartitionBook/HeteroEdgePartitionDict: edge partition book.
  """
  def __init__(
    self,
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
    super().__init__('', num_nodes, edge_index, edge_ids, node_feat, node_feat_ids,
                   edge_feat, edge_feat_ids, num_parts, current_partition_idx,
                   node_feat_dtype, edge_feat_dtype, edge_assign_strategy,
                   chunk_size, master_addr, master_port, num_rpc_threads)

  def partition(self):
    r""" Partition graph and feature data into different parts along with all
    other distributed partitioners, save the result of the current partition
    index into output directory.
    """
    if 'hetero' == self.data_cls:
      node_pb_dict = {}
      node_feat_dict = {}
      for ntype in self.node_types:
        node_pb = self._partition_node(ntype)
        node_pb_dict[ntype] = node_pb

        current_node_feat_part = self._partition_node_feat(node_pb, ntype)
        if current_node_feat_part is not None:
          node_feat_dict[ntype] = current_node_feat_part

      edge_pb_dict = {}
      graph_dict = {}
      edge_feat_dict = {}
      for etype in self.edge_types:
        current_graph_part, edge_pb = self._partition_graph(node_pb_dict, etype)
        edge_pb_dict[etype] = edge_pb
        graph_dict[etype] = current_graph_part

        current_edge_feat_part = self._partition_edge_feat(edge_pb, etype)
        if current_edge_feat_part is not None:
          edge_feat_dict[etype] = current_edge_feat_part

      return (
        self.num_parts, self.current_partition_idx,
        graph_dict, node_feat_dict, edge_feat_dict, node_pb_dict, edge_pb_dict
      )
    else:
      node_pb = self._partition_node()
      node_feat = self._partition_node_feat(node_pb)
      graph, edge_pb = self._partition_graph(node_pb)
      edge_feat = self._partition_edge_feat(edge_pb)

      return (
        self.num_parts, self.current_partition_idx,
        graph, node_feat, edge_feat, node_pb, edge_pb
      )


class DistTableDataset(DistDataset):
  """ Creates `DistDataset` from ODPS tables.

  Args:
    edge_tables: A dict({edge_type : odps_table}) denoting each
      bipartite graph input table of heterogeneous graph, where edge_type is
      a tuple of (src_type, edge_type, dst_type).
    node_tables: A dict({node_type(str) : odps_table}) denoting each
      input node table.
    num_nodes: Number of all graph nodes, should be a dict for hetero data.
    graph_mode: mode in graphlearn_torch's `Graph`, 'CPU', 'ZERO_COPY'
      or 'CUDA'.
    sort_func: function for feature reordering, return feature data(2D tenosr)
      and a map(1D tensor) from id to index.
    split_ratio: The proportion of data allocated to the GPU, between 0 and 1.
    device_group_list: A list of `DeviceGroup`. Each DeviceGroup must have the
      same size. A group of GPUs with peer-to-peer access to each other should
      be set in the same device group for high feature collection performance.
    directed: A Boolean value indicating whether the graph topology is
      directed.
    reader_threads: The number of threads of table reader.
    reader_capacity: The capacity of table reader.
    reader_batch_size: The number of records read at once.
    label: A CPU torch.Tensor(homo) or a Dict[NodeType, torch.Tensor](hetero)
      with the label data for graph nodes.
    device: The target cuda device rank to perform graph operations and
      feature lookups.
    feature_with_gpu (bool): A Boolean value indicating whether the created
      ``Feature`` objects of node/edge features use ``UnifiedTensor``.
      If True, it means ``Feature`` consists of ``UnifiedTensor``, otherwise
      ``Feature`` is a PyTorch CPU Tensor, the ``device_group_list`` and
      ``device`` will be invliad. (default: ``True``)
    edge_assign_strategy: The assignment strategy when partitioning edges,
      should be 'by_src' or 'by_dst'.
    chunk_size: The chunk size for partitioning.
    master_addr: The master TCP address for RPC connection between all
      distributed partitioners.
    master_port: The master TCP port for RPC connection between all
      distributed partitioners.
    num_rpc_threads: The number of RPC worker threads to use.
  """
  def load(
    self,
    num_partitions=1,
    partition_idx=0,
    edge_tables=None,
    node_tables=None,
    num_nodes=0,
    graph_mode='ZERO_COPY',
    device_group_list=None,
    reader_threads=10,
    reader_capacity=10240,
    reader_batch_size=1024,
    label=None,
    device=None,
    feature_with_gpu=True,
    edge_assign_strategy='by_src',
    chunk_size=10000,
    master_addr=None,
    master_port=None,
    num_rpc_threads=16,
  ):
    assert isinstance(edge_tables, dict)
    assert isinstance(node_tables, dict)
    edge_index, eids, feature = {}, {}, {}
    edge_hetero = (len(edge_tables) > 1)
    node_hetero = (len(node_tables) > 1)

    print("Start Loading edge and node tables...")
    step = 0
    start_time = time.time()
    for e_type, table in edge_tables.items():
      edge_list = []
      reader = common_io.table.TableReader(table,
                                           slice_id=partition_idx,
                                           slice_count=num_partitions,
                                           num_threads=reader_threads,
                                           capacity=reader_capacity)
      while True:
        try:
          data = reader.read(reader_batch_size, allow_smaller_final_batch=True)
          edge_list.extend(data)
          step += 1
        except common_io.exception.OutOfRangeException:
          reader.close()
          break
        if step % 1000 == 0:
          print(f"{datetime.datetime.now()}: load "
                f"{step * reader_batch_size} edges.")
      rows = [e[0] for e in edge_list]
      cols = [e[1] for e in edge_list]
      eids_array = np.array([e[2] for e in edge_list], dtype=np.int64)
      edge_array = np.stack([np.array(rows, dtype=np.int64),
                             np.array(cols, dtype=np.int64)])
      if edge_hetero:
        edge_index[e_type] = eids_array
        eids[e_type] = eids
      else:
        edge_index = edge_array
        eids = eids_array
      del rows
      del cols
      del edge_list

    step = 0
    for n_type, table in node_tables.items():
      feature_list = []
      reader = common_io.table.TableReader(table,
                                           slice_id=partition_idx,
                                           slice_count=num_partitions,
                                           num_threads=reader_threads,
                                           capacity=reader_capacity)
      while True:
        try:
          data = reader.read(reader_batch_size, allow_smaller_final_batch=True)
          feature_list.extend(data)
          step += 1
        except common_io.exception.OutOfRangeException:
          reader.close()
          break
        if step % 1000 == 0:
          print(f"{datetime.datetime.now()}: load "
                f"{step * reader_batch_size} nodes.")
      ids = torch.tensor([feat[0] for feat in feature_list], dtype=torch.long)
      if isinstance(feature_list[0][1], bytes):
        float_feat= [
          list(map(float, feat[1].decode().split(':')))
          for feat in feature_list
        ]
      else:
        float_feat= [
          list(map(float, feat[1].split(':')))
          for feat in feature_list
        ]
      if node_hetero:
        feature[n_type] = torch.tensor(float_feat)
      else:
        feature = torch.tensor(float_feat)
      del float_feat
      del feature_list
    load_time = (time.time() - start_time) / 60
    print(f'Loading table completed in {load_time:.2f} minutes.')

    print("Start partitioning graph and feature...")
    p_start = time.time()
    dist_partitioner = DistTableRandomPartitioner(
        num_nodes, edge_index=edge_index, edge_ids=eids,
        node_feat=feature, node_feat_ids=ids,
        num_parts=num_partitions, current_partition_idx=partition_idx,
        edge_assign_strategy=edge_assign_strategy,
        chunk_size=chunk_size, master_addr=master_addr, master_port=master_port,
        num_rpc_threads=num_rpc_threads)
    (
      self.num_partitions,
      self.partition_idx,
      graph_data,
      node_feat_data,
      edge_feat_data,
      self.node_pb,
      self.edge_pb
    ) = dist_partitioner.partition()
    part_time = (time.time() - p_start) / 60
    print(f'Partitioning completed in {part_time:.2f} minutes.')

    # init graph
    if isinstance(graph_data, dict):
      # heterogeneous.
      edge_index, edge_ids = {}, {}
      for k, v in graph_data.items():
        edge_index[k] = v.edge_index
        edge_ids[k] = v.eids
    else:
      # homogeneous.
      edge_index = graph_data.edge_index
      edge_ids = graph_data.eids
    self.init_graph(edge_index, edge_ids, layout='COO',
                    graph_mode=graph_mode, device=device)

    # load node feature
    if node_feat_data is not None:
      node_cache_ratio, node_feat, node_feat_id2idx, node_feat_pb = \
        _cat_feature_cache(partition_idx, node_feat_data, self.node_pb)
      self.init_node_features(
        node_feat, node_feat_id2idx, None, node_cache_ratio,
        device_group_list, device, feature_with_gpu, dtype=None
      )
      self._node_feat_pb = node_feat_pb

    # load edge feature
    if edge_feat_data is not None:
      edge_cache_ratio, edge_feat, edge_feat_id2idx, edge_feat_pb = \
        _cat_feature_cache(partition_idx, edge_feat_data, self.edge_pb)
      self.init_edge_features(
        edge_feat, edge_feat_id2idx, edge_cache_ratio,
        device_group_list, device, feature_with_gpu, dtype=None
      )
      self._edge_feat_pb = edge_feat_pb

    # load whole node labels
    self.init_node_labels(label)


## Pickling Registration

def rebuild_dist_table_dataset(ipc_handle):
  ds = DistTableDataset.from_ipc_handle(ipc_handle)
  return ds

def reduce_dist_table_dataset(dataset: DistTableDataset):
  ipc_handle = dataset.share_ipc()
  return (rebuild_dist_table_dataset, (ipc_handle, ))

ForkingPickler.register(DistTableDataset, reduce_dist_table_dataset)